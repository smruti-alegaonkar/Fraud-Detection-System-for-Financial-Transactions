import streamlit as st

st.title("Online Transaction Fraud Detection")
st.write("This app predicts whether a transaction is fraudulent using a Machine Learning model.")

amount = st.number_input("Transaction Amount (₹)", min_value=1.0)
transaction_type = st.selectbox("Transaction Type", ["Online", "POS", "ATM"])
location = st.selectbox("Location", ["India", "International"])
device = st.selectbox("Device Used", ["Mobile", "Desktop"])
hour = st.slider("Transaction Hour", 0, 23)

