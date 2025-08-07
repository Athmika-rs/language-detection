import streamlit as st
import joblib
import matplotlib.pyplot as plt
from PIL import Image

# Load model and vectorizer
model = joblib.load("language_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("🌍 Language Detection App")
st.markdown("Enter any sentence and I’ll try to detect the language using a trained Random Forest model.")

user_input = st.text_area("Enter Text", "")

if st.button("Detect Language"):
    if user_input.strip() != "":
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)
        st.success(f"Predicted Language: **{prediction[0]}**")
    else:
        st.warning("Please enter some text.")

st.markdown("---")
st.subheader("📊 Confusion Matrix")
st.image("confusion_matrix.png", caption="Model Performance", use_column_width=True)
