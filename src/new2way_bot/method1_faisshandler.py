import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from langchain.vectorstores import FAISS

# FAISSHandler class for managing FAISS operations
class FAISSHandler:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.faiss_store = None
        self.faiss_metadata = None
    
    def generate_embeddings(self, text):
        # Generate embeddings for the given text
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings
    
    def create_faiss_store(self, data_path):
        # Load data from a CSV file and generate embeddings
        df = pd.read_csv(data_path)
        
        # Generate embeddings for the DataFrame
        df["embedding"] = df.apply(lambda row: self.generate_embeddings(f"{row['intent']} {row['response']}"), axis=1)
        
        # Create a FAISS vector store
        document_vectors = np.array([embedding.numpy() for embedding in df["embedding"]])
        self.faiss_store = FAISS.from_embeddings(document_vectors)
        
        # Store metadata associated with each record
        self.faiss_metadata = df.to_dict("records")
        
        return self.faiss_store, self.faiss_metadata
    
    def extract_intent_from_metadata(self, metadata):
        # Extract the intent from the metadata
        return metadata["intent"]
