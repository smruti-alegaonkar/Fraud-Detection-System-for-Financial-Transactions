import streamlit as st
import numpy as np

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="centered"
)

# Title
st.title("💳 Financial Transaction Fraud Detection")

st.write("""
This application predicts whether a financial transaction is **Fraudulent** or **Legitimate**
based on basic transaction details.
""")

st.divider()

# Input fields
st.subheader("🧾 Enter Transaction Details")

amount = st.number_input("Transaction Amount (₹)", min_value=0.0, step=100.0)
transaction_type = st.selectbox(
    "Transaction Type",
    ["Online Purchase", "ATM Withdrawal", "Bank Transfer", "POS Payment"]
)

location = st.selectbox(
    "Transaction Location",
    ["Same City", "Different City", "Different State", "International"]
)

time = st.slider("Transaction Time (Hour of Day)", 0, 23, 12)

# Dummy encoding (simple logic)
transaction_type_map = {
    "Online Purchase": 0,
    "ATM Withdrawal": 1,
    "Bank Transfer": 2,
    "POS Payment": 3
}

location_map = {
    "Same City": 0,
    "Different City": 1,
    "Different State": 2,
    "International": 3
}

# Predict button
if st.button("🔍 Detect Fraud"):
    
    # Convert inputs
    features = np.array([
        amount,
        transaction_type_map[transaction_type],
        location_map[location],
        time
    ])

    # ---- Dummy Fraud Logic ----
    if amount > 50000 and location == "International":
        prediction = 1
    else:
        prediction = 0

    st.divider()

    # Output
    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

st.divider()

st.caption("Mini Project | Fraud Detection using Streamlit")
