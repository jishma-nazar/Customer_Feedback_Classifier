import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch


# Load model & tokenizer

model_path = "./customer_feedback_bert"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()
model.to("cpu") 


# Prediction function
def predict_feedback(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=1)
    label_idx = torch.argmax(probs, dim=1).item()
    
    # Map index to label
    label_map = {0: "Complaint", 1: "Praise"}  # Todo: extend later to include "Suggestions"
    return label_map[label_idx], probs[0][label_idx].item()


# Streamlit UI
st.title("Customer Feedback Classifier")
st.write("Enter customer feedback and get a prediction:")

user_input = st.text_area("Feedback", height=120)

if st.button("Predict"):
    if user_input.strip() != "":
        label, confidence = predict_feedback(user_input)
        st.success(f"Prediction: **{label}** (Confidence: {confidence:.2f})")
    else:
        st.warning("Please enter some feedback text.")
