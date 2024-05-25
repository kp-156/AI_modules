from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationEntityMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import json
import os

# Initialize a smaller text-generation pipeline
text_generation_pipeline = pipeline("text-generation", model="distilgpt2", tokenizer="distilgpt2", max_length=1024, pad_token_id=50256)
print("hi")
