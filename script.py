import pickle
import json
import pandas as pd
from src.config import ner_json_samples
from pprint import pprint

# from src.modules.ner_utility import NER_UTILITY
# language="en"
# client_id="1"
# ner_embeddings_cluster = "data/ner_cluster/embeddings/queries_embeddings_{client_id}_{language}.pkl"
# ner_embeddings_cluster_path = ner_embeddings_cluster.format(client_id=client_id, language=language)
# obj = NER_UTILITY(language="en", client_id="1")
# obj.encode_and_save_cluster_embeddings()
# obj.encode_and_save_sample_embeddings()

# # with open(ner_embeddings_cluster_path, "rb") as f:
# #     pkl = pickle.load(f)
# #     print(pkl.keys())

# def load_json_data(file_path):
#     with open(file_path, "r") as file:
#         data = json.load(file)
#     return data



"""This code is used to add more data in the master df using a well structured json dataset file_path"""
# from src.utils.pd_utility import create_new_df_for_new_usecase
# client_id = "1"
# language = "en"
# file_path = "/home/pritika/travel-chatbot-backup/data/travel_data.json"
# create_new_df_for_new_usecase(client_id, language, file_path)


"""Creating new df from an existing df"""
# df1 = "/home/sdb/pritika/travel_chatbot/data/master_dfs/master_df_info.csv"
# df2 = pd.read_csv(df1)
# df3 = df2.iloc[:, :].copy()
# print(df3)
# df3.to_csv("/home/sdb/pritika/travel_chatbot/data/new_travel_data2.csv", index = False)


"""To train for the intent and ner model. Call src/modules/app_training.py"""
# from src.modules.app_training import UpdateClusterSample
# data_file_path = "/home/sdb/pritika/travel_chatbot/data/new_travel_data2.json"  # Path to JSON file
# language = "en"
# client_id = "1"
# data = load_json_data(data_file_path)
# updates = UpdateClusterSample(language = language, client_id= client_id)
# for dataset in data["datasets"]:
#     print("Processing dataset:", dataset["name"])
#     updates.process_dataset(dataset)

"""Recursively get unique keys from JSON data."""
# def get_unique_keys(data, keys=None, unique_keys=None):   
#     if unique_keys is None:
#         unique_keys = set()
#     if keys is None:
#         keys = set()
#     if isinstance(data, dict):
#         for key, value in data.items():
#             keys.add(key)
#             get_unique_keys(value, keys, unique_keys)
#     elif isinstance(data, list):
#         for item in data:
#             get_unique_keys(item, keys, unique_keys)
#     else:
#         unique_keys.update(keys)
#         keys.clear()
#     return unique_keys
# json_data = load_json_data('/home/sdb/pritika/travel_chatbot/data/ner_cluster/json/1_en.json')
# unique_keys = get_unique_keys(json_data)
# print(len(unique_keys))
# print("Unique keys in the JSON data:")
# for key in unique_keys:
#     print(key)



"""This code is used to get the unique ner category and their length range present in the ner_sample dataset"""
# def get_unique_ner_category(json_data):
#     transformed_dict = {}
#     for key, value in json_data.items():
#         for sub_key, sub_value in value.items():
#             if isinstance(sub_value, list):
#                 transformed_dict[key] = sub_value
#             break
#     return transformed_dict

# lang, c_id = "en", "1"
# ner_sample_path = ner_json_samples.format(language=lang, client_id=c_id)
# with open(ner_sample_path, "r") as file:
#     data = json.load(file)
# ner_list = get_unique_ner_category(data)
# ner_list = dict(sorted(ner_list.items()))
# with open("/home/pritika/travel-chatbot-backup/data/ner_list.json", "w") as file:
#     json.dump(ner_list, file, indent=2)





"""Prints the keys of a dictionary loaded from a pickle file."""
# def print_dict_keys(filename):
#     try:
#         with open(filename, 'rb') as handle:
#             data = pickle.load(handle)

#         if isinstance(data, dict):
#             print("Keys:")
#             for key, values in data.items():
#                 print(key)
#                 print(values[0][0])
                
#         else:
#             print("The loaded data is not a dictionary.")
#     except FileNotFoundError:
#         print(f"Error: File '{filename}' not found.")

# print_dict_keys("/home/pritika/travel-chatbot-backup/data/transcriptionscorer/embedding_dict.pkl")

def inspect_types(obj, level=0):
    indent = "  " * level
    if isinstance(obj, dict):
        print(f"{indent}dict with {len(obj)} keys:")
        for key, value in obj.items():
            print(f"{indent}  Key: {key} -> Type: {type(value).__name__}")
            inspect_types(value, level + 1)
    elif isinstance(obj, list):
        print(f"{indent}list with {len(obj)} elements:")
        for idx, item in enumerate(obj):
            print(f"{indent}  Index {idx} -> Type: {type(item).__name__}")
            inspect_types(item, level + 1)
    elif isinstance(obj, tuple):
        print(f"{indent}tuple with {len(obj)} elements:")
        for idx, item in enumerate(obj):
            print(f"{indent}  Index {idx} -> Type: {type(item).__name__}")
            inspect_types(item, level + 1)
    elif isinstance(obj, set):
        print(f"{indent}set with {len(obj)} elements:")
        for item in obj:
            print(f"{indent}  Element -> Type: {type(item).__name__}")
            inspect_types(item, level + 1)
    else:
        print(f"{indent}{type(obj).__name__}")

with open('/home/pritika/travel-chatbot-backup/data/transcriptionscorer/keyword_importance/embeddings/1_en_phrase_importance.pkl', 'rb') as handle:
    data = pickle.load(handle)
# recursive_inspect(data)
inspect_types(data)