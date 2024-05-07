from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer,
    BertModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, classification_report

from utils import time_it, get_device


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label),
        }


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits


def train(model, data_loader, optimizer, scheduler, device):
    model.train()

    for batch in tqdm(data_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()

        optimizer.step()
        scheduler.step()


def evaluate(model, data_loader, device, predict=False):
    model.eval()

    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    if predict:
        return classification_report(actual_labels, predictions)
    else:
        return accuracy_score(actual_labels, predictions)


@time_it
def run(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    test_texts,
    test_labels,
    bert_model_name,
    num_classes,
    max_length,
    batch_size,
    num_epochs,
    learning_rate,
):
    device = get_device()

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer, max_length
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, tokenizer, max_length
    )
    test_dataset = TextClassificationDataset(
        test_texts, test_labels, tokenizer, max_length
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = BERTClassifier(bert_model_name, num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_dataloader, optimizer, scheduler, device)
        accuracy = evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")

    report = evaluate(model, test_dataloader, device, predict=True)
    print(report)
    return report


if __name__ == "main":
    run()
