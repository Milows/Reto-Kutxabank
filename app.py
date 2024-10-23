import streamlit as st
import joblib


rf_text_model = joblib.load('rf_text_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


st.title("Predicción de Categorías de Transacciones")


st.write("Ingrese un concepto de transacción y obtenga su categoría predicha:")


transaction_input = st.text_input("Concepto de la transacción", "compra en supermercado")


if st.button("Predecir Categoría"):
    transaction_vectorized = vectorizer.transform([transaction_input])
    
    prediction = rf_text_model.predict(transaction_vectorized)
    
    st.write(f"La categoría predicha es: {prediction[0]}")