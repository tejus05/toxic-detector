import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
CATEGORIES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MODEL_PATH = "toxic_comment_model"

class ToxicCommentDataset(Dataset):
    def __init__(self, texts, targets=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'targets': target
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

def train_model():
    df = pd.read_csv("train.csv", engine="python", on_bad_lines='skip')

    print(f"Dataset shape: {df.shape}")
    print(f"Sample data:\n{df.head()}")

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Train size: {train_df.shape}, Validation size: {val_df.shape}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = ToxicCommentDataset(
        texts=train_df['comment_text'].values,
        targets=train_df[CATEGORIES].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    val_dataset = ToxicCommentDataset(
        texts=val_df['comment_text'].values,
        targets=val_df[CATEGORIES].values,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CATEGORIES),
        problem_type="multi_label_classification"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_auc = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        model.train()
        train_losses = []

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = torch.nn.BCEWithLogitsLoss()(logits, targets)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        print(f"Train Loss: {np.mean(train_losses):.4f}")

        model.eval()
        val_losses = []
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = torch.nn.BCEWithLogitsLoss()(logits, targets)
                val_losses.append(loss.item())

                predictions = torch.sigmoid(logits).detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()

                all_targets.append(targets)
                all_predictions.append(predictions)

        all_targets = np.vstack(all_targets)
        all_predictions = np.vstack(all_predictions)

        aucs = []
        for i in range(len(CATEGORIES)):
            auc = roc_auc_score(all_targets[:, i], all_predictions[:, i])
            aucs.append(auc)
            print(f"{CATEGORIES[i]} AUC: {auc:.4f}")

        mean_auc = np.mean(aucs)
        print(f"Validation Loss: {np.mean(val_losses):.4f}")
        print(f"Mean AUC: {mean_auc:.4f}")

        if mean_auc > best_val_auc:
            best_val_auc = mean_auc
            print(f"New best model with AUC: {best_val_auc:.4f}")

            os.makedirs(MODEL_PATH, exist_ok=True)
            model.save_pretrained(MODEL_PATH)
            tokenizer.save_pretrained(MODEL_PATH)

    print(f"Training completed. Best AUC: {best_val_auc:.4f}")
    return model, tokenizer

def load_model():
    if os.path.exists(MODEL_PATH):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        return model, tokenizer
    else:
        print("No saved model found. Training new model...")
        return train_model()

def predict_toxicity(text, model, tokenizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    inputs = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()[0]

    results = {}
    for i, category in enumerate(CATEGORIES):
        results[category] = float(probabilities[i])

    return results

model, tokenizer = load_model()

test_texts = [
    "You are amazing and I love this community.",
    "You're a disgusting and horrible person.",
    "Go away, nobody wants you here.",
    "Just a normal comment, nothing wrong.",
    "I will find you and make you pay."
]

for text in test_texts:
    result = predict_toxicity(text, model, tokenizer)
    print(f"\nInput: {text}")
    for category, score in result.items():
        print(f"{category}: {score:.4f}")

# import shutil
# shutil.make_archive('toxic_comment_model', 'zip', 'toxic_comment_model')

# from google.colab import files
# files.download('toxic_comment_model.zip')

