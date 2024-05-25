from src.modules.intent_ner_processor import IntentNERProcessor
from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer
from src.modules.intent import IntentDetector
from src.config import FastSentenceTransformer

from src.utils.pd_utility import store_conversation_logic
import time
import pandas as pd
import os
from src.config import csv_file_path
import ast
import csv

language = "en"
client_id = "1"
input_sentence = "What are the best hotels in Paris"
intent_ner_processor = IntentNERProcessor(text=input_sentence, language=language, client_id=client_id)
# start_time = time.time()

# csv_file_path = '/home/sdb/pritika/travel_chatbot/data/master_dfs/2_en_master_df.csv'
# with open(csv_file_path, mode='r') as file:
#     csv_reader = csv.DictReader(file)
#     for row in csv_reader:
#         if not ast.literal_eval(row['ner_ngram']):
#             continue  
#         ner_ngram = eval(row['ner_ngram'])
#         input_sentence = row['Sentences']
#         # input_sentence = "What are some solo travel tips for a female traveler?"
#         # ner_ngram = {'gender':[0, 1], 'profession':[0, 3]}

#         print("input = ", row['Sentences'])
#         ner_result = intent_ner_processor.process_ner(ner_ngram)
#         top_values = {}
#         for tag, values in ner_result.items():
#             top_values[tag] = [val for val in values[0]]
#         print(f"\nTop NER outcomes -> {top_values} \n")

# print("--- %s seconds ---" % (time.time() - start_time))

# intent_detector = IntentDetector(language=language, client_id=client_id)
# model = SentenceTransformer(FastSentenceTransformer, device="cuda", quantize=True)
# master_df_path_file = csv_file_path.format(client_id=client_id, language=language)
# if os.path.exists(master_df_path_file):
#     master_df = pd.read_csv(master_df_path_file)
# else:
#     master_df = pd.DataFrame(columns=["Sentences", "Keywords", "Intent", "NER", "Ner_ngram_length", "Values", "Embeddings", "Keyword_Embeddings"])



def main():
    """
    Main function to process the input sentence and store the data in the database
    It goes through the following steps:
        1. Get input_sentence via API from customer (TODO: This step is pending)
        2. Call intent function to get the intent of the input sentence
        3. Call NER function to get the NER ngram of the input sentence
        4. Get the top NER result from NER ngram
        5. Update the local client dataframes, embeddings and NER cluster and sample files
        5. Pack the input_sentence as question, intent as goal_name, top NER result as ner_key and value in a dict
        6. Store the dictionary in the database (mongoDB)
    """
   
    # TODO: Currently this is hard coded. Get the input_sentence via API from the end user
  
    # df = pd.read_csv("/home/sdb/pritika/travel_chatbot/data/client_dataframes/1_en_dataframe")
    # data_entry_no = 1
    # for input_sentence in df['Sentences']:
    #     print('data_entry_no:==================== ', data_entry_no)
    #     data_entry_no += 1

    start_time = time.time()
    print("Step 1: Fetching intent from the input sentence")
    result = intent_ner_processor.process_intent()
    unique_intent = None if result["intents"] is None else result["intents"][0]
    print(
        f"Step 1 complete. selected sentences -> {result['sentences']}, selected intent -> {unique_intent}, \n"
        f"intent type (multi or ambiguous) -> {result['type']}\n"
    )
    print("For intent --- %s seconds ---" % (time.time() - start_time))

    # print("Step 2: Fetching NER from the input sentence")
    ner_ngram_str = intent_ner_processor.get_ner_ngram(IntentSentence=result["sentences"][0])
    print("Type of ner_ngram_str========>", type(ner_ngram_str))
    # print(f"Step 2 complete. NER ngram -> {ner_ngram_str}\n")

    # start_time = time.time()
    print("Step 3: Processing NER ngram to get the top NER outcomes")
    # ner_ngram = ast.literal_eval(ner_ngram_str) #ner_ngram is a string. so converting it to a dict for extracting the key pair
    ner_ngram = {'names':[1, 3],'location':[1,3]}
    ner_result = intent_ner_processor.process_ner(ner_ngram)
    top_values = {}
    for tag, values in ner_result.items():
        if isinstance(values[0], list) and len(values[0]) > 0:
            top_values[tag] = [values[0][0]]
        else:
            top_values[tag] = []
        # top_values[tag] = [val[0][0] for val in values]
    print(f"\nStep 3 complete. Top NER outcomes -> {ner_result} \n")
    # print("For ner --- %s seconds ---" % (time.time() - start_time))

    # print("Step 4: Saving the intent and NER results to local files")
    # # Update the intent and NER results in local dataframes, and embeddings if intent or NER is not None
    # if unique_intent:
    #     intent_ner_processor.update_intent_data(sentence=input_sentence, goal_name=unique_intent)

    # if top_values:
    #     intent_ner_processor.update_ner_cluster_data(top_values=top_values)
    #     intent_ner_processor.update_ner_sample_data(top_values=top_values)
    # print("Step 4 completed successfully")

    # print("Step 5: store conversation data to db")
    
    # data = {
    #     "client_id": "1",
    #     "language": "en",
    #     "question": input_sentence,
    #     "goal_name": unique_intent,
    #     "ner_key": list(top_values.keys()),
    #     "ner_ngram_length": list(ner_ngram.values()),
    #     "values": [top_values[key] for key in top_values],
    # }
    # store_conversation_logic(data=data)

  
    # print("Step 5: storing conversation data pd df")  
    # new_row = {
    #     "Sentences": input_sentence,
    #     "Keywords": [intent_detector.get_keywords(input_sentence)],
    #     "Intent": data["goal_name"],
    #     "NER": str(data["ner_key"]),
    #     "Ner_ngram_length": str(data["ner_ngram_length"]),
    #     "Values": str(data["values"]),
    #     "Embeddings": model.encode(data["question"]), 
    #     "Keyword_Embeddings": intent_detector.update_keyword_embeddings() 
    # } 
    # master_df = master_df._append(new_row, ignore_index=True)
    # master_df.to_csv(master_df_path_file, index=False)
    # print("All steps completed successfully")


if __name__ == "__main__":
    # start_time = time.time()

    main()
    # print("--- %s seconds ---" % (time.time() - start_time))
