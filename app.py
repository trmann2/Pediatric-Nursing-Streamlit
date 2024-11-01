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
        self.context_mapping = {}  # Stores chapter/section information
        
        # Initialize the model and tokenizer
        @st.cache_resource
        def load_model():
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            return tokenizer, model
        
        self.tokenizer, self.model = load_model()
        self._initialize_pdf()

    def _detect_section_header(self, text):
        """Detect if text is a section header"""
        header_patterns = [
            r'^Chapter \d+[:\s]+(.*)',
            r'^\d+\.\d*\s+[A-Z][^.!?]*$',
            r'^[A-Z][A-Z\s]{3,}[A-Z]$'
        ]
        for pattern in re.compile('|'.join(header_patterns), re.MULTILINE).finditer(text):
            return pattern.group()
        return None

    def _detect_table(self, text):
        """Detect if text block contains a table"""
        table_indicators = [
            r'Table \d+[:\-]',
            r'\|\s*[\w\s]+\s*\|',
            r'[\t]{2,}',
            r'\s{4,}',
            r'(\d+[\s\t]+){3,}'
        ]
        return any(re.search(pattern, text) for pattern in table_indicators)

    def _process_table(self, text):
        """Process and format table content"""
        text = re.sub(r'\s{2,}', ' | ', text)
        text = re.sub(r'[\t]+', ' | ', text)
        return f"\nTABLE:\n{text}\n"

    def _clean_text(self, text):
        """Clean and normalize text while preserving structure"""
        if self._detect_table(text):
            return self._process_table(text)
            
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'[^\w\s\-\+\/°%.,;:()?!><=#]', '', text)
        
        text = re.sub(r'(?<![\w.])(vs\.|eq\.|approx\.|temp\.|resp\.|pt\.|hr\.|min\.|sec\.|mg\/kg|mcg\/kg)(?![\w.])', 
                     lambda m: m.group(1).replace('.', '_dot_'), text)
        
        return text.strip()

    def _split_into_chunks(self, text, chunk_size=600):
        """Split text into chunks while preserving context"""
        current_section = self._detect_section_header(text[:200]) or "General Content"
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        chunks = []
        current_chunk = ""
        
        for part in parts:
            if self._detect_table(part):
                if current_chunk:
                    chunks.append(current_chunk)
                chunks.append(part)
                current_chunk = ""
                continue
                
            header = self._detect_section_header(part)
            if header:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = part
                current_section = header
                continue
                
            if len(current_chunk) + len(part) < chunk_size:
                current_chunk += " " + part
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = part
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks, current_section

    def _process_pdf(self, pdf_content):
        try:
            reader = PyPDF2.PdfReader(pdf_content)
            current_chapter = "Unknown Chapter"
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    if not text.strip():
                        continue
                    
                    if re.search(r'copyright|all rights reserved|blank page', text.lower()):
                        continue
                    
                    cleaned_text = self._clean_text(text)
                    chunks, section = self._split_into_chunks(cleaned_text)
                    
                    if re.match(r'^Chapter \d+', section):
                        current_chapter = section
                    
                    for chunk in chunks:
                        if len(chunk.split()) > 20:
                            chunk_id = len(self.text_chunks)
                            self.text_chunks.append(chunk)
                            self.page_mapping[chunk_id] = page_num
                            self.context_mapping[chunk_id] = {
                                'chapter': current_chapter,
                                'section': section
                            }
                        
                except Exception as e:
                    st.warning(f"Skipping page {page_num} due to processing error")
                    continue
                    
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            raise

    def get_embedding(self, text):
        @st.cache_data
        def _cached_embedding(_text):
            inputs = self.tokenizer(_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).numpy()
        return _cached_embedding(text)

    def _create_embeddings(self):
        with st.spinner("Processing content..."):
            self.embeddings = np.vstack([self.get_embedding(chunk) for chunk in self.text_chunks])

    def get_answer(self, question, top_k=3, similarity_threshold=0.2):
        try:
            question_embedding = self.get_embedding(question)
            similarities = cosine_similarity(question_embedding, self.embeddings)[0]
            
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            answer_parts = []
            current_context = None
            
            for idx in top_indices:
                similarity_score = similarities[idx]
                if similarity_score < similarity_threshold:
                    continue
                    
                chunk_text = self.text_chunks[idx]
                page_num = self.page_mapping[idx]
                context = self.context_mapping[idx]
                
                if context != current_context:
                    answer_parts.append(f"\n**{context['chapter']} - {context['section']}**")
                    current_context = context
                
                if self._detect_table(chunk_text):
                    answer_parts.append(f"\n*Table from page {page_num}* (confidence: {similarity_score:.2%}):\n```\n{chunk_text}\n```")
                else:
                    answer_parts.append(f"\n*From page {page_num}* (confidence: {similarity_score:.2%}):\n{chunk_text}")
            
            if not answer_parts:
                return "I couldn't find relevant information. Try rephrasing your question or being more specific."
            
            return "\n".join(answer_parts) + "\n\n---\n*Disclaimer: This information is extracted from educational materials and should not replace professional medical judgment.*"
            
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            return "An error occurred while processing your question. Please try again."

    def _initialize_pdf(self):
        try:
            with st.spinner("Loading PDF content..."):
                response = requests.get(self.pdf_url)
                response.raise_for_status()
                
                pdf_content = io.BytesIO(response.content)
                self._process_pdf(pdf_content)
                
                if self.text_chunks:
                    st.success(f"Successfully processed PDF: Found {len(self.text_chunks)} content chunks")
                    self._create_embeddings()
                else:
                    st.error("No usable content could be extracted from the PDF")
                    
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            raise

def main():
    st.set_page_config(page_title="Pediatric Nursing Chatbot", page_icon="👩‍⚕️", layout="wide")
    
    st.title("Pediatric Nursing Resource Chatbot 👩‍⚕️")
    
    st.markdown("""
    This chatbot can answer questions about pediatric nursing based on the provided resource material.
    
    **Note:** All information is for educational purposes only and should not replace professional medical judgment.
    """)

    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            pdf_url = "https://raw.githubusercontent.com/trmann2/Pediatric-Nursing-Resources/5d5d787167317c6963c8603e4e03a377dd1e4cdb/eBook.pdf"
            try:
                st.session_state.chatbot = PDFChatbot(pdf_url)
                st.success("Chatbot initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize chatbot: {str(e)}")
                return

    col1, col2 = st.columns([2, 1])

    with col1:
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
