import streamlit as st
import requests
import PyPDF2
import io
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

[Previous PDFChatbot class code from the last message]

# Streamlit app
def main():
    st.set_page_config(page_title="Pediatric Nursing Chatbot", page_icon="👩‍⚕️", layout="wide")
    
    st.title("Pediatric Nursing Resource Chatbot 👩‍⚕️")
    
    st.markdown("""
    This chatbot can answer questions about pediatric nursing based on the provided resource material.
    
    **Note:** All information is for educational purposes only and should not replace professional medical judgment.
    """)

    # Initialize session state
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            pdf_url = "https://raw.githubusercontent.com/trmann2/Pediatric-Nursing-Resources/5d5d787167317c6963c8603e4e03a377dd1e4cdb/eBook.pdf"
            try:
                st.session_state.chatbot = PDFChatbot(pdf_url)
                st.success("Chatbot initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize chatbot: {str(e)}")
                return

    # Create two columns for a better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Question input
        question = st.text_input("Ask a question about pediatric nursing:", 
                               placeholder="e.g., What are the key assessments for pediatric drowning?")

        if question:
            with st.spinner("Finding answer..."):
                response = st.session_state.chatbot.get_answer(question)
                st.markdown("### Answer:")
                st.markdown(response)
                
                st.markdown("---")
                st.markdown("*Disclaimer: This information is extracted from educational materials and should not replace professional medical judgment.*")

    with col2:
        st.markdown("### Sample Questions:")
        sample_questions = [
            "What are the normal vital signs for infants?",
            "What are the key warning signs of respiratory distress?",
            "How should you assess pain in pediatric patients?",
            "What are the signs of dehydration in children?",
            "How do you calculate pediatric medication dosages?"
        ]
        
        for q in sample_questions:
            if st.button(q):
                with st.spinner("Finding answer..."):
                    response = st.session_state.chatbot.get_answer(q)
                    st.session_state.question = q
                    st.session_state.response = response

if __name__ == "__main__":
    main()
