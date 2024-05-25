import json
from src.modules.ner_utility import NER_UTILITY
from src.modules.intent_utility import INTENT_UTILITY
from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer
FastSentenceTransformer= "all-MiniLM-L6-v2"
from src.config import csv_file_path
import os
import pandas as pd
import ast

file_path = "/home/sdb/pritika/travel_chatbot/data/master_dfs/master_df_info.csv"  # Path to JSON file
language = "en"
client_id = "1"


class UpdateClusterSample:
    def __init__(self, language="en", client_id="1"):
        self.language = language
        self.client_id = client_id
        self.ner_util_obj = NER_UTILITY(language= self.language, client_id= self.client_id)
        self.int_util_obj = INTENT_UTILITY(language= self.language, client_id= self.client_id)
        self.model = SentenceTransformer(FastSentenceTransformer, device="cuda", quantize=True)
        self.master_df_path_file = csv_file_path.format(client_id=client_id, language=language)
        if os.path.exists(self.master_df_path_file):
            self.master_df = pd.read_csv(self.master_df_path_file)
        else:
            self.master_df = pd.DataFrame(columns=["Sentences", "Keywords", "Intent", "ner_ngram", "NER", "Ner_ngram_length", "Values", "Embeddings", "Keyword_Embeddings"])

    def update_and_encode_cluster_logic(self, data):
        try:
            ner_ngram = data["ner_ngram"]
            values = (data["values"])
            print("values:" , values)
            self.ner_util_obj.update_cluster_json(ner_ngram, values)
            self.ner_util_obj.encode_and_save_cluster_embeddings()
            print("update_and_encode_logic: Success")
        except Exception as e:
            print("Error in update_and_encode_logic:", str(e))

    def update_and_encode_samples_logic(self, data):
        try:
            ner_ngram = data["ner_ngram"]
            values = (data["values"])

            self.ner_util_obj.update_samples_json(ner_ngram, values)
            self.ner_util_obj.encode_and_save_sample_embeddings()
            print("update_and_encode_samples_logic: Success")
        except Exception as e:
            print("Error in update_and_encode_samples_logic:", str(e))

    def process_text_logic(self, data):
        try:
            sentence = data["question"]
            goal_name = data["goal_name"]

            self.int_util_obj.update_dataframe(sentence, goal_name)
            self.int_util_obj.create_embeddings()
            self.int_util_obj.create_keyword_embeddings()
            print("process_text_logic: Success")
        except Exception as e:
            print("Error in process_text_logic:", str(e))

    def store_conversation_logic(self, data):
        ner_ngram = data["ner_ngram"]
        ner_keys = list(ner_ngram.keys())
        ner_lengths = [len(ner_ngram[key]) for key in ner_keys]
        new_row = {
            "Sentences": data["question"],
            "Keywords": [self.int_util_obj.get_keywords(data["question"])],
            "Intent": data["goal_name"],
            "ner_ngram": data["ner_ngram"],
            "NER": str(ner_keys),
            "Ner_ngram_length": str(ner_lengths),
            "Values": str(data["values"]),
            "Embeddings": self.model.encode(data["question"]), 
            "Keyword_Embeddings": self.int_util_obj.create_keyword_embeddings() 
        } 
        self.master_df = self.master_df._append(new_row, ignore_index=True)
        self.master_df.to_csv(self.master_df_path_file, index=False)
        print("store_conversation_logic successfully implemented")


    def process_dataset(self, dataset):
        for item in dataset["data"]:
            # print("dataitem================================================>", item)
            self.process_text_logic(item)
            self.update_and_encode_samples_logic(item)
            self.update_and_encode_cluster_logic(item)
            
            # self.store_conversation_logic(item)

def load_json_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    pass
    # data = load_json_data(data_file_path)
    # updates = UpdateClusterSample(language = language, client_id= client_id)
    # for dataset in data["datasets"]:
    #     print("Processing dataset:", dataset["name"])
    #     updates.process_dataset(dataset)
    

