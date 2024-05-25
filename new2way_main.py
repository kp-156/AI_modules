import os
from src.new2way_bot.retriever import Retriever

def main():
    data_path = '/home/hariram/travel-chatbot_1/src/new2way_bot/data/travel_usecase_data.json'
    embedded_path = '/home/hariram/travel-chatbot_1/src/new2way_bot/data/travel_usecase_embedded.json'
    
    user_query = input("Please enter your query: ")

    retriever_obj = Retriever()
    matching_line = retriever_obj.find_matching_intent_response(user_query)

    if matching_line:
        print("Matching response:", matching_line)
    else:
        print("No matching response found.")

if __name__ == "__main__":
    main()
