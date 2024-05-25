# import sys
# sys.path.append("/root/pritika/travel_chatbot/src")
import os
import json
from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer
import pickle
from src.config import (
    ner_embeddings_cluster,
    ner_embeddings_samples,
    ner_json_cluster,
    ner_json_samples,
    FastSentenceTransformer 
)
# ner_embeddings_cluster = "/home/sdb/pritika/travel_chatbot/data/ner_cluster/embeddings/queries_embeddings_{client_id}_{language}.pkl"
# ner_embeddings_samples = "/root/pritika/travel_chatbot/data/ner_samples/embeddings/encoded_data_{client_id}_{language}.pkl"
# ner_json_cluster =  "/home/sdb/pritika/travel_chatbot/data/ner_cluster/json/{client_id}_{language}.json"
# ner_json_samples = "/home/sdb/pritika/travel_chatbot/data/ner_samples/json/ner_samples_{client_id}_{language}.json"
# FastSentenceTransformer = "all-MiniLM-L6-v2"


class NER_UTILITY:
    def __init__(self, text = None, language="en", client_id="2"):
        self.client_id = client_id
        self.language = language
        self.text = text
        self.model = SentenceTransformer(FastSentenceTransformer, device="cuda", quantize=True)
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ner_embeddings_samples_path = ner_embeddings_samples.format(client_id=client_id, language=language)
        self.ner_embeddings_cluster_path = ner_embeddings_cluster.format(client_id=client_id, language=language)
        # self.embeddings_samples_dict = self.load_ner_embeddings_samples()
        # self.embeddings_cluster_dict = self.load_ner_embeddings_cluster()

        self.ner_samples_json_path = ner_json_samples.format(client_id=client_id, language=language)
        self.ner_cluster_json_path = ner_json_cluster.format(client_id=client_id, language=language)


    def load_ner_embeddings_samples(self):
        """
        This function loads the NER samples embeddings from local file based on client_id and language.
        Returns empty dictionary if the file is not found.
        :return:
        """
        try:
            with open(self.ner_embeddings_samples_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"NER samples embedding file for {self.client_id}-{self.language} not found. Please provide the file")
            return None

    def load_ner_embeddings_cluster(self):
        """
        This function loads the NER cluster embeddings from local file based on client_id and language.
        Returns empty dictionary if the file is not found.
        :return:
        """
        try:
            with open(self.ner_embeddings_cluster_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"NER cluster embedding file for {self.client_id}-{self.language} not found. Please provide the file")
            return None   

    def load_ner_sample_or_cluster_json(self, file_path):
        """
        This function loads the latest NER samples JSON data from file. Returns empty dict if file doesn't exist.
        """
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
        else:
            return None
        return data


    def update_cluster_json(self, ner_ngram, values):
        for category, val in zip(ner_ngram.keys(), values):
            if os.path.exists(self.ner_cluster_json_path):
                with open(self.ner_cluster_json_path, 'r') as file:
                    data = json.load(file)
            else:
                data = {}

            if category not in data:
                data[category] = []   
            if val not in data[category]:
                data[category].append(val)
                
                data_converted = {str(key): value for key, value in data.items()} # Converted data back to list

                with open(self.ner_cluster_json_path, 'w') as file:
                    json.dump(data_converted, file, indent=2)

            print(f"JSON file updated for language: {self.language}, client_id: {self.client_id}, ner_ngram: {ner_ngram}")


    def encode_and_save_cluster_embeddings(self):
        try:
            with open(self.ner_cluster_json_path, 'r') as f:
                data = json.load(f)
                
            embeddings_dict = {}

            for category, queries in data.items():
                query_embeddings = self.model.encode(queries, convert_to_tensor=True)


                embeddings_dict[category] = query_embeddings.tolist()
            with open(self.ner_embeddings_cluster_path, 'wb' ) as pickle_file:
                pickle.dump(embeddings_dict, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
            print("-----pkl file saved----ner cluster.utility")
        except Exception as error:
            print("---------------error in cluster embeddings------------", error)




    ####sample values

    def update_samples_json(self, ner_ngram, values):
        for category, val in zip(ner_ngram.keys(), values):
            if os.path.exists(self.ner_samples_json_path):
                with open(self.ner_samples_json_path, 'r') as file:
                    data = json.load(file)
            else:
                data = {}
                    
            if category not in data:
                data[category] = {"ngram": ner_ngram[category], "values": []}
            
            if val not in data[category]["values"]:
                data[category]["values"].append(val)

                with open(self.ner_samples_json_path, 'w') as file:
                    json.dump(data, file, indent=2)

        print(f"-----------------JSON file updated for language: {self.language}, client_id: {self.client_id}, ner_ngram: {ner_ngram}")



    def encode_and_save_sample_embeddings(self):
        
        with open(self.ner_samples_json_path, 'r') as json_file:
            data = json.load(json_file)
            

        encoded_data = {}
        for name, tokens in data.items():
            if 'values' in tokens:
                values = tokens['values']
                #print(f"Encoding tokens for '{name}': {values}")
                encoded_tokens = self.model.encode(values, convert_to_tensor=True)
                encoded_data[name] = encoded_tokens.tolist()

        with open(self.ner_embeddings_samples_path, 'wb') as pickle_file:
            pickle.dump(encoded_data, pickle_file)
            print("-----pkl file saved----ner sample.utility")


    def find_min_max_word_lengths(self, sample_list):  
        if not sample_list:
            return 0, 0
        
        min_length = float('inf')
        max_length = 0
        
        for string in sample_list:
            words = string.split()
            length = len(words)
            if length < min_length:
                min_length = length
            if length > max_length:
                max_length = length
        
        return [min_length, max_length]


    # def find_min_max_sentence_lengths(texts):
    #     min_length = float('inf')
    #     max_length = 0

    #     for text in texts:
    #         sentences = text.split()
            
    #         for sentence in sentences:
    #             words = sentence.split()
    #             length = len(words)
    #             min_length = min(min_length, length)
    #             max_length = max(max_length, length)

    #     return min_length, max_length



    # def update_ner_ngram_for_key(client_id, language, key,values):
    #     try:
    #         min_l, max_l = find_min_max_sentence_lengths(values)

    #         client = MongoClient("mongodb://your_mongodb_connection_string")
    #         db = client["your_database_name"]
    #         collection = db["your_collection_name"]

    #         conversations = collection.find({"client_id": client_id, "language": language})

    #         updated_count = 0
    #         for conversation in conversations:
    #             slots = conversation.get("slots", {})
    #             for slot_name, slot in slots.items():
    #                 if slot_name == key:
    #                     slot["ner_ngram"] = [min_l,max_l]
    #                     updated_count += 1

    #             collection.update_one({"_id": conversation["_id"]}, {"$set": {"slots": slots}})

    #         if updated_count > 0:
    #             return {"message": f"Updated {updated_count} conversations for key: {key}"}
    #         else:
    #             return {"message": f"No conversations found with slot name matching key: {key}"}

    #     except Exception as e:
    #         return {"error": str(e)}

# if __name__ == "__main__":
#     main()