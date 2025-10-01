# Customer Feedback Classifier

End-to-end ML project to classify customer feedback as **Complaint** or **Praise** using **DistilBERT**.

## Features
- Fine-tuned DistilBERT on a **subset of 5k Amazon Polarity reviews** for faster experimentation
- Achieved **~85% validation accuracy**
- Interactive **Streamlit app** for real-time feedback classification with confidence scores

## Dataset
Uses the public Amazon Polarity dataset from Hugging Face:
[https://huggingface.co/datasets/mteb/amazon_polarity](https://huggingface.co/datasets/mteb/amazon_polarity)

> Note: Dataset is **not included** in the repo to keep it lightweight. It is downloaded automatically via the `datasets` library.
