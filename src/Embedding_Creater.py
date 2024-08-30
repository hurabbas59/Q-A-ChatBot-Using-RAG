import os
import pickle
from dotenv import loadenv
import pdfplumber
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

loadenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class SimpleDocumentIndexer:
    def __init__(self, pdf_files, model_name="text-embedding-ada-002", store_path="faiss_store", openai_api_key=OPENAI_API_KEY):
        self.pdf_files = pdf_files  # List of PDF file paths
        self.model_name = model_name
        self.store_path = store_path
        self.openai_api_key = openai_api_key

    def read_pdf(self, pdf_path):
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text

    def process_documents(self):
        all_texts = []
        
        # Read and process each PDF file
        for pdf_file in self.pdf_files:
            text = self.read_pdf(pdf_file)
            all_texts.append(text)
        
        # Combine all texts into one document
        combined_text = "\n".join(all_texts)
        
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=550, chunk_overlap=50)
        chunked_docs = splitter.split_text(combined_text)
        
        # Embed documents and create FAISS index
        embeddings = OpenAIEmbeddings(model=self.model_name, openai_api_key=self.openai_api_key)
        faiss_store = FAISS.from_texts(texts=chunked_docs, embedding=embeddings)
        
        # Attempt to save the FAISS index
        try:
            faiss_store.save_local(self.store_path)
            print(f"FAISS store saved at: {os.path.abspath(self.store_path)}")
        except Exception as e:
            print(f"Failed to save FAISS store: {e}")

# Usage example
pdf_files = [r"C:\\Users\\USER\Desktop\\hrbot\src\\nvidia-earnings.pdf"]  # Use the absolute path
indexer = SimpleDocumentIndexer(pdf_files=pdf_files, openai_api_key=OPENAI_API_KEY)
indexer.process_documents()
