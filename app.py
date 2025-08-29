import gradio as gr
import pandas as pd
import joblib

# Load your trained model
model = joblib.load("fraud_detections.pkl")  # replace with your compressed .pkl if you have one

# Define prediction function
def predict_fraud(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, transaction_type):
    # Derived features
    diff_org = oldbalanceOrg - newbalanceOrig
    diff_dest = newbalanceDest - oldbalanceDest

    # One-hot encode type
    type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
    type_DEBIT = 1 if transaction_type == "DEBIT" else 0
    type_PAYMENT = 1 if transaction_type == "PAYMENT" else 0
    type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0

    # Create input DataFrame
    input_df = pd.DataFrame([[
        amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest,
        diff_org, diff_dest, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER
    ]], columns=[
        "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        "diff_org", "diff_dest", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"
    ])

    # Predict
    prediction = model.predict(input_df)[0]
    return "FRAUD ⚠️" if prediction == 1 else "NOT FRAUD ✅"

# Build Gradio interface
iface = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Number(label="Transaction Amount"),
        gr.Number(label="Old Balance (Origin)"),
        gr.Number(label="New Balance (Origin)"),
        gr.Number(label="Old Balance (Destination)"),
        gr.Number(label="New Balance (Destination)"),
        gr.Dropdown(choices=["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"], label="Transaction Type")
    ],
    outputs=gr.Label(label="Prediction"),
    title="💳 Fraud Detection System",
    description="Enter transaction details to predict if it's FRAUD or NOT FRAUD."
)

# Launch locally
if __name__ == "__main__":
    iface.launch(share=True)
