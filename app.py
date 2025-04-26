import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr

MODEL_PATH = "thequantumcoder/toxic-detector"
MAX_LEN = 128
CATEGORIES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

def predict_toxicity(text):
    if not text.strip():
        return {category: 0.0 for category in CATEGORIES}

    inputs = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    results = {category: float(probabilities[i]) for i, category in enumerate(CATEGORIES)}
    return results

iface = gr.Interface(
    fn=predict_toxicity,
    inputs=gr.Textbox(lines=4, placeholder="Enter text to analyze for toxicity..."),
    outputs=gr.Label(num_top_classes=len(CATEGORIES)),
    title="Toxic Comment Classification",
    description="This model identifies toxic content like threats, obscenity, insults, and identity-based hate.",
    examples=[
        ["This is a wonderful and polite message."],
        ["You are the worst and should disappear."],
        ["Get lost, you idiot."]
    ]
)

if __name__ == "__main__":
    iface.launch(share=True)