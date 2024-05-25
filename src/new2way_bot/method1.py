#pip install transformers torch langchain pandas
from src.modules.intent_ner_processor import IntentNERProcessor
import pandas as pd
import numpy as np
from langchain.vectorstores import FAISS
from transformers import DistilBertTokenizer, DistilBertModel
from langchain.vectorstores import FAISS
import torch

from new2way_bot.method1_model1a import Model1a
from new2way_bot.method1_faisshandler import FAISSHandler

class Method1:
    def __init__(self):
        self.model1 = Model1a()  # Instance of Model1
        self.faiss_handler = FAISSHandler()  # Instance of FAISSHandler
        
        # Initialize FAISS store and metadata
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        data_path = "/path/to/your/csvfile.csv"  # Correct the path
        self.faiss_store, self.faiss_metadata = self.faiss_handler.create_faiss_store(data_path)

    # Function to find a response in the FAISS store
    def find_in_faiss(self, query, intent):
        query_embedding = self.faiss_handler.generate_embeddings(query)
        
        # Perform a similarity search in the FAISS store
        search_results = self.faiss_store.similarity_search(query_embedding, k=1)
        
        if search_results:
            result_index = search_results[0].page_id  # Index of the closest match
            metadata = self.faiss_metadata[result_index]
            
            stored_intent = self.faiss_handler.extract_intent_from_metadata(metadata)  # Extract stored intent
            
            if stored_intent == intent:
                # If stored intent matches the intended intent, return the response
                response = metadata["response"]
                context = f"context: {response}, input: {query}"
                
                # Use Model1 to generate the chatbot response
                chatbot_response = self.model1.conversation.predict(input=context)
                return chatbot_response
            
            else:
                # If the stored intent doesn't match, use the raw user input
                chatbot_response = self.model1.conversation.predict(input=query)
                return chatbot_response
        
        # If no relevant match is found
        return "No relevant response found."
