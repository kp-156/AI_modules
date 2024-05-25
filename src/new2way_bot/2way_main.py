import sys
sys.path.append('/home/sdb/pritika/travel_chatbot/src/new2way_bot')

from src.modules.intent_ner_processor import IntentNERProcessor
import pandas as pd
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
import os
import datetime
from collections import deque
from translator import Translator
from langchain.vectorstores import FAISS
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from typing_extensions import Optional
import time




# from .method1 import Method1
# from .method2 import Method2   
import googletrans 
import ast
from src.new2way_bot.method1_model1a import TravelAssistantChatbot
from src.new2way_bot.intent_based_retriever import IntentBasedRetriever
from src.config import memory_path_for_chatbot 



# translator = Translator()

# detected_language = translator.detect_language()
# translated_user_input = translator.translate_user_input()

def Intent_and_ner(input_sentence):
    language = "en"
    client_id = "1" #use this for hdfc
    # client_id = "5" #use this for easemytrip
    intent_ner_processor = IntentNERProcessor(text=input_sentence, language=language, client_id=client_id)
    sentence_list, ambi_or_multi, intent_list = intent_ner_processor.process_intent()
    unique_intent = None if intent_list is None else list(set(intent_list))[0]
    # unique_intent = unique_intent.replace("_", " ")
    print(type(unique_intent))
    print("intent=======>", unique_intent)
    ner_ngram_str = intent_ner_processor.get_ner_ngram(IntentSentence=sentence_list[0])
    ner_ngram = ast.literal_eval(ner_ngram_str)
    ner_result = intent_ner_processor.process_ner(ner_ngram)

    return unique_intent, ner_result

    

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'



# possible_intents = ["hotel","discount","accommodation_search", "general_travel_question", 'scooter_rental_search',
#                         'transportation_rental_search', 'health_advice', 'activity_search', 'accommodation_search',
#                         'order status', 'packing_essentials_search', 'transportation_cost_comparison', 'general_travel_question',
#                         'insurance_search', 'travel_app_recommendation', 'travel_inspiration_search', 'travel_restrictions_search',
#                         'currency_search', 'transportation_search', 'transportation_booking_search', 'visa_information_search',
#                         'roadtrip_planning', 'currency_exchange_advice', 'attraction_search', 'luggage_storage_search',
#                         'travel_preparation_resources', 'currency_exchange_search', 'greeting', 'travel_season_advice',
#                         'bargaining_advice', 'visa_requirements_search', 'flight_search', 'travel_companion_search',
#                         'travel_advisory_resources', 'flight_search_resources', 'event_search', 'internet_access_search',
#                         'car_rental_search', 'flight_booking_advice', 'insurance_advice', 'service_search',
#                         'tipping_etiquette_advice', 'cultural_etiquette_advice', 'product information', 'volunteer_search',
#                         'restaurant_search', 'packing_advice', 'accessibility_resources', 'luggage_service_search',
#                         'schedule meeting', 'language_learning_resources', 'shopping_recommendation', 'safety_advice',
#                         'technical support', 'farewell', 'safety_search', 'like the Viking Age', 'cultural_norms_search',
#                         'sustainable_travel_tips', 'food_recommendation_search', 'weather_search',
#                         'humor','emotion','politics','profile','conversations','Ai','computer','movies','money','healt','greetings',
#                         'science','histroy','gossip','psychology','sports','literature','general_travel_question']




start_time = time.time()

# Main function with user interaction and response generation
def main():
    # method1_instance = Method1()
    translator = Translator()


    while True:
        user_input = input("User: ")
        if user_input.lower() in ["bye", "goodbye", "stop", "quit"]:
            print("Chatbot: I'm glad to be of assistance. Goodbye!")
            break
        
        user_language = translator.detect_language(user_input)
        translated_user_input = translator.translate_user_input(user_input, base_language="en")

        intent, ner_dict = Intent_and_ner(translated_user_input) 
        intent_based_retriever = IntentBasedRetriever()
        retrieved_response= intent_based_retriever.find_response(user_intent=intent, ner_dict=ner_dict)

        
        travel_assistence_chatbot = TravelAssistantChatbot()
        model_output = travel_assistence_chatbot.generate_response(translated_user_input, retrieved_response)
        translated_response = translator.translate_response(model_output, user_language)

        print("Chatbot response:", translated_response)


        
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Step name: {elapsed_time:.2f} seconds")


# Run the main function when the script is executed
if __name__ == "__main__":
    main()

