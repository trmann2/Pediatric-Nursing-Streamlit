import streamlit as st
import requests
import PyPDF2
import io
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

class PDFChatbot:
    def __init__(self, pdf_url):
        self.pdf_url = pdf_url
        self.text_chunks = []
        self.embeddings = None
        self.page_mapping = {}
        
        # Initialize the model and tokenizer
        @st.cache_resource
        def load_model():
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            return tokenizer, model
        
        self.tokenizer, self.model = load_model()
        self._initialize_pdf()

    def _initialize_pdf(self):
        try:
            with st.spinner("Loading PDF content..."):
                response = requests.get(self.pdf_url)
                response.raise_for_status()
                
                pdf_content = io.BytesIO(response.content)
                self._process_pdf(pdf_content)
                
                if self.text_chunks:
                    self._create_embeddings()
                else:
                    st.error("No text could be extracted from the PDF")
                    
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            raise

    def _process_pdf(self, pdf_content):
        try:
            reader = PyPDF2.PdfReader(pdf_content)
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    if not text.strip():
                        continue
                    
                    cleaned_text = self._clean_text(text)
                    chunks = self._split_into_chunks(cleaned_text)
                    
                    for chunk in chunks:
                        self.text_chunks.append(chunk)
                        self.page_mapping[len(self.text_chunks) - 1] = page_num
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            raise

    def _clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\+\/°%]', '', text)
        return text.strip()

    def _split_into_chunks(self, text, chunk_size=1000):
        words = text.split()
        chunks = []
        overlap = 100
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks

    @st.cache_data
    def _get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def _create_embeddings(self):
        with st.spinner("Processing content..."):
            self.embeddings = np.vstack([self._get_embedding(chunk) for chunk in self.text_chunks])

    def get_answer(self, question, top_k=3):
        try:
            question_embedding = self._get_embedding(question)
            similarities = cosine_similarity(question_embedding, self.embeddings)[0]
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            answer_parts = []
            
            for idx in top_indices:
                similarity_score = similarities[idx]
                if similarity_score < 0.3:
                    continue
                    
                chunk_text = self.text_chunks[idx]
                page_num = self.page_mapping[idx]
                
                answer_parts.append(f"**From page {page_num}** (confidence: {similarity_score:.2%}):\n{chunk_text}\n")
            
            if not answer_parts:
                return "I couldn't find any relevant information in the document. Please try rephrasing your question."
            
            return "\n".join(answer_parts)
            
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return "An error occurred while processing your question. Please try again."

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
