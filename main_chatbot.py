from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from src.modules.intent_ner_processor import IntentNERProcessor
import pymongo
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from auto_gptq import AutoGPTQForCausalLM
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory

from test_api import test_search_endpoint
import torch
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer,GenerationConfig,TextStreamer,pipeline
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
import os
from src.new2way_bot.retriever import Retriever
# from x_search import re_rank_results
# from search import search_bm25_with_embeddings,re_rank_with_cosine_similarity
from langchain.chains import ConversationalRetrievalChain
from flask import Flask, request, jsonify
import warnings
import time
warnings.filterwarnings("ignore")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = mongo_client["chat_history_db"]
# collection = db["chat_history_collection"]
# db = mongo_client["bot"]
# collection = db["conversation_collection"]

# def Intent(input_sentence):
#     language = "en"
#     client_id = "1" #use this for hdfc
#     # client_id = "5" #use this for easemytrip
#     intent_ner_processor = IntentNERProcessor(text=input_sentence, language=language, client_id=client_id)
#     sentence_list, ambi_or_multi, intent_list = intent_ner_processor.process_intent()
#     unique_intent = None if intent_list is None else list(set(intent_list))[0]
#     # unique_intent = unique_intent.replace("_", " ")
#     # unique_intent = unique_intent.title()
#     print(type(unique_intent))
#     print("intent=======>", unique_intent)
#     return unique_intent

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class Chatbot:
    def __init__(self):
        self.model_id = "TheBloke/Nous-Hermes-13B-GPTQ"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoGPTQForCausalLM.from_quantized(self.model_id,
                                                        use_safetensors=True,
                                                        trust_remote_code=True,
                                                        device=DEVICE)

        self.generation_config = GenerationConfig.from_pretrained(self.model_id)
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=False, skip_special_tokens=True, use_multiprocessing=False)

        self.pipe = pipeline("text-generation",
                             model=self.model,
                             tokenizer=self.tokenizer,
                             max_new_tokens=256,
                            #  max_length=2048,
                             temperature=0.75,
                             top_p=0.95,
                             repetition_penalty=1.15,
                             generation_config=self.generation_config,
                             streamer=self.streamer,
                             batch_size=12,
                             do_sample=True)

        self.llm = HuggingFacePipeline(pipeline=self.pipe)

        self.template = """
        ###Instruction: You are a travel assistant chatbot designed to help customers with their Easemytrip travel-related queries.
        In order to assist customers effectively, your chatbot should be able to handle a variety of travel-related inquiries, including travel and hotel information, flight details, Easemytrip service, and general assistance.
        Your task is to develop a function or script that takes two parameters, `input` and `history`, and generates appropriate responses based on the user's queries and the conversation history. Chatbot should maintain context throughout the conversation, provide accurate and relevant information, and ensure customer satisfaction.
        Remember to strictly adhere to the conversation flow provided and end the conversation after gathering all the necessary input and chat history.

        Conversation Flow:
        - Start the conversation with a greeting or introduction.
        - Respond to the user's queries or requests for assistance.
        - Gather necessary information from the user.
        - Provide confirmation or additional details as needed.
        - End the conversation after receiving all required input.
        Make the output short simple and concise
        Strictly give responses of 70 words and no more

        Current conversation:
        {chat_history}
        ###User: [INST]{input}[/INST]\n.
        ###Response: """.strip()

        # self.prompt = PromptTemplate(input_variables=['input', 'chat_history', 'context'], template=self.template)

        self.memory = ConversationSummaryBufferMemory(
                                               memory_key='chat_history',
                                               return_messages=False,
                                               llm= self.llm
                                            )
        
        self.conversation = ConversationChain(
        llm=self.llm, 
        verbose=False, 
        memory=self.memory,
        prompt=PromptTemplate(
        input_variables=['chat_history', 'input'],
        output_parser=None, 
        # partial_variables={"context"},
        template=self.template,
        template_format='f-string',
        validate_template=False)
)


def retrieve_content_by_name(user_input, max_chars=None):
    retrieve = Retriever()
    result = retrieve.find_matching_intent_response(user_input)
    print("result from retriever: ", result)
    # db = mongo_client["bot"]
    # collection = db["travel_test"]
    # result = collection.find_one({"name": intent})
    if result is not None:
        return result[:max_chars]
    else:
        return None


def main():
    chatbot = Chatbot()
    chatbot.conversation.memory.clear()
    while True:
        user_input = input("User: ")
        # data=request.json
        # user_input=data['query']
        
        try:
            if user_input.lower() == 'bye' or  user_input.lower() == 'goodbye':
                print("I am glad to be of assistance")
                break
            # intent = Intent(user_input)
            # index_name = "easemytrip_conversation"  #use this for hdfc conversation
            
            result=retrieve_content_by_name(user_input, max_chars=1000)
            context = f"context:{result},input={user_input}"

            input_ids = chatbot.tokenizer.encode(context, return_tensors='pt')
            if input_ids.shape[1] > 2048:
                input_ids = input_ids[:, -2048:]

            # start_time = time.time()
            chatbot_response = chatbot.conversation.predict(input=context)
            # print("---retrieval")
            print("response:",chatbot_response)
            # print(" --- %s seconds ---" % (time.time() - start_time))
            # return chatbot_response
            
            torch.cuda.empty_cache()
        except:
            chatbot_response = chatbot.conversation.predict(input=user_input)
            print("respose:",chatbot_response)
            # print("---generative")
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # app.run(debug=True)
    main()
