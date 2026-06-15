import os
import json
import random
import warnings
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

warnings.filterwarnings("ignore")

# =========================
# 1. 路徑設定
# =========================
TRAIN_PATH = r"C:\Python\ai cup\final\vpesg4k_train_1000 V1.json"
TEST_PATH = r"C:\Python\ai cup\final\vpesg4k_test_2000.json"
SUBMISSION_PATH = r"C:\Python\ai cup\final\submission.csv"

MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
MAX_LEN = 256
EPOCHS = 5
BATCH_SIZE = 8
LR = 2e-5
SEED = 42

TASKS = [
    "promise_status",
    "verification_timeline",
    "evidence_status",
    "evidence_quality"
]

WEIGHTS = {
    "promise_status": 0.20,
    "verification_timeline": 0.15,
    "evidence_status": 0.30,
    "evidence_quality": 0.35,
}

LABEL_LISTS = {
    "promise_status": ["No", "Yes"],
    "verification_timeline": [
        "N/A",
        "already",
        "within_2_years",
        "between_2_and_5_years",
        "longer_than_5_years"
    ],
    "evidence_status": ["N/A", "No", "Yes"],
    "evidence_quality": ["N/A", "Clear", "Not Clear", "Misleading"],
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# 2. 讀取資料
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "data" in data:
            data = data["data"]
        elif "records" in data:
            data = data["records"]
        else:
            data = list(data.values())

    return pd.DataFrame(data)


def normalize_timeline(x):
    x = str(x)
    if x == "more_than_5_years":
        return "longer_than_5_years"
    return x


def build_text(df):
    df = df.copy()
    df["data"] = df["data"].fillna("").astype(str)

    if "esg_type" in df.columns:
        df["esg_type"] = df["esg_type"].fillna("").astype(str)
    else:
        df["esg_type"] = ""

    df["text"] = df["data"] + " [ESG_TYPE] " + df["esg_type"]
    return df


train_df = build_text(load_json(TRAIN_PATH))
test_df = build_text(load_json(TEST_PATH))

train_df["verification_timeline"] = train_df["verification_timeline"].apply(normalize_timeline)

print("Train:", train_df.shape)
print("Test :", test_df.shape)


# =========================
# 3. Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )


def filter_train_by_task(df, task):
    if task == "promise_status":
        return df.copy()

    if task in ["verification_timeline", "evidence_status"]:
        return df[df["promise_status"] == "Yes"].copy()

    if task == "evidence_quality":
        return df[df["evidence_status"] == "Yes"].copy()

    return df.copy()


def train_one_task(task, train_data, valid_data=None, output_dir="model_tmp"):
    labels = LABEL_LISTS[task]
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    task_train = filter_train_by_task(train_data, task)
    task_train["label"] = task_train[task].fillna("N/A").astype(str).map(label2id)
    task_train = task_train.dropna(subset=["label"])
    task_train["label"] = task_train["label"].astype(int)

    train_dataset = Dataset.from_pandas(task_train[["text", "label"]])
    train_dataset = train_dataset.map(tokenize_function, batched=True)

    eval_dataset = None
    if valid_data is not None:
        task_valid = valid_data.copy()
        task_valid["label"] = task_valid[task].fillna("N/A").astype(str).map(label2id)
        task_valid = task_valid.dropna(subset=["label"])
        task_valid["label"] = task_valid["label"].astype(int)

        eval_dataset = Dataset.from_pandas(task_valid[["text", "label"]])
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer
)

    trainer.train()
    return trainer, labels


def predict_one_task(trainer, labels, df):
    pred_dataset = Dataset.from_pandas(df[["text"]])
    pred_dataset = pred_dataset.map(tokenize_function, batched=True)

    outputs = trainer.predict(pred_dataset)
    pred_ids = np.argmax(outputs.predictions, axis=1)
    return [labels[i] for i in pred_ids]


def post_process(pred_df):
    pred_df = pred_df.copy()

    no_promise = pred_df["promise_status"].eq("No")
    pred_df.loc[
        no_promise,
        ["verification_timeline", "evidence_status", "evidence_quality"]
    ] = "N/A"

    yes_no_timeline = (
        pred_df["promise_status"].eq("Yes")
        & pred_df["verification_timeline"].eq("N/A")
    )
    pred_df.loc[yes_no_timeline, "verification_timeline"] = "already"

    no_evidence = pred_df["evidence_status"].isin(["No", "N/A"])
    pred_df.loc[no_evidence, "evidence_quality"] = "N/A"

    yes_evidence_no_quality = (
        pred_df["evidence_status"].eq("Yes")
        & pred_df["evidence_quality"].eq("N/A")
    )
    pred_df.loc[yes_evidence_no_quality, "evidence_quality"] = "Not Clear"

    return pred_df


# =========================
# 4. train1000 自切 8:2 驗證
# =========================
train_part, valid_part = train_test_split(
    train_df,
    test_size=0.2,
    random_state=SEED,
    stratify=train_df["promise_status"]
)

valid_pred = pd.DataFrame({"id": valid_part["id"].values})

for task in TASKS:
    print("\n==============================")
    print("Validation training task:", task)
    print("==============================")

    trainer, labels = train_one_task(
        task=task,
        train_data=train_part,
        valid_data=valid_part,
        output_dir=f"valid_model_{task}"
    )

    valid_pred[task] = predict_one_task(trainer, labels, valid_part)

valid_pred = post_process(valid_pred)

print("\n===== Validation Score =====")
weighted_score = 0.0

for task in TASKS:
    y_true = valid_part[task].fillna("N/A").astype(str).values
    y_pred = valid_pred[task].fillna("N/A").astype(str).values

    score = f1_score(
        y_true,
        y_pred,
        labels=LABEL_LISTS[task],
        average="macro",
        zero_division=0
    )

    weighted_score += WEIGHTS[task] * score
    print(f"{task:<26} Macro-F1 = {score:.4f}")

print(f"\nWeighted Macro F1 Score = {weighted_score:.4f}")


# =========================
# 5. 用完整 train1000 重訓，預測 test2000
# =========================
final_pred = pd.DataFrame()

if "id" in test_df.columns:
    final_pred["id"] = test_df["id"].values
else:
    final_pred["id"] = range(1, len(test_df) + 1)

for task in TASKS:
    print("\n==============================")
    print("Final training task:", task)
    print("==============================")

    trainer, labels = train_one_task(
        task=task,
        train_data=train_df,
        valid_data=None,
        output_dir=f"final_model_{task}"
    )

    final_pred[task] = predict_one_task(trainer, labels, test_df)

final_pred = post_process(final_pred)

final_pred["verification_timeline"] = final_pred["verification_timeline"].replace({
    "longer_than_5_years": "more_than_5_years"
})

final_pred = final_pred[
    [
        "id",
        "promise_status",
        "verification_timeline",
        "evidence_status",
        "evidence_quality"
    ]
]

final_pred.to_csv(
    SUBMISSION_PATH,
    index=False,
    encoding="utf-8",
    lineterminator="\n"
)

print("\nSubmission saved:", SUBMISSION_PATH)
print("Rows:", len(final_pred))
print(final_pred.head())