# from pymongo import MongoClient


# def find_min_max_sentence_lengths(texts):
#     min_length = float("inf")
#     max_length = 0
#     for text in texts:
#         sentences = text.split()
#         for sentence in sentences:
#             words = sentence.split()
#             length = len(words)
#             min_length = min(min_length, length)
#             max_length = max(max_length, length)
#     return min_length, max_length


# def update_ner_ngram_for_key(client_id, language, key, values):
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
#                     slot["ner_ngram"] = [min_l, max_l]
#                     updated_count += 1

#             collection.update_one(
#                 {"_id": conversation["_id"]}, {"$set": {"slots": slots}}
#             )

#         if updated_count > 0:
#             return {"message": f"Updated {updated_count} conversations for key: {key}"}
#         else:
#             return {
#                 "message": f"No conversations found with slot name matching key: {key}"
#             }

#     except Exception as e:
#         return {"error": str(e)}
import os
import json
from sentence_transformers import SentenceTransformer
import pickle
#import pymongo
from pymongo import MongoClient


json_base_path = '/root/pritika/travel_chatbot/data/ner_cluster/json'
embeddings_path = "/root/pritika/travel_chatbot/data/ner_cluster/embeddings"

model_path = "/root/pritika/travel_chatbot/data/ner_cluster/fine_tuned_similarity_model"
model = SentenceTransformer(model_path)

def update_json(client_id, language, ner_ngram, values):
    json_file_path = os.path.join(json_base_path, f"{client_id}_{language}.json")

    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            print(data)
    else:
        data = {}

    ner_key = tuple(ner_ngram)  # because ner_ngram is a list and lists are unhashable so converted ner_ngram into a tuple
    if ner_key in data:
        data[ner_key].extend(values)
    else:
        data[ner_key] = values

    data_converted = {str(key): value for key, value in data.items()} # Converted data back to list

    with open(json_file_path, 'w') as file:
        json.dump(data_converted, file, indent=2)

    print(f"JSON file updated for language: {language}, client_id: {client_id}, ner_ngram: {ner_ngram}")


def encode_and_save_queries_to_json(client_id,lang):
    # print(client_id, "----------abc----------------cluster")
    # print(lang, "----------abc----------------cluster")
    json_file = os.path.join(json_base_path, f"{client_id}_{lang}.json")
    output_file = os.path.join(embeddings_path, f"queries_embeddings_{client_id}_{lang}.json")
    print(json_file)
    print(output_file)

    with open(json_file, 'r') as f:
        data = json.load(f)
        #print(data,'-----------')

    embeddings_dict = {}
    model = SentenceTransformer(model_path)

    for category, queries in data.items():
        query_embeddings = model.encode(queries, convert_to_tensor=True)


        embeddings_dict[category] = query_embeddings.tolist()
    # print(embeddings_dict)
    with open(output_file, 'w') as f:
        json.dump(embeddings_dict, f)




####sample values

json_sample_path = '/root/pritika/travel_chatbot/data/ner_samples/json'
embeddings_sample_path = "/root/pritika/travel_chatbot/data/ner_samples/embeddings"

model_path = "/root/pritika/travel_chatbot/data/ner_samples/fine_tuned_similarity_model"

model = SentenceTransformer(model_path)

def update_json_samples(client_id, language, ner_ngram, values):
    json_file_path = os.path.join(json_sample_path, f"ner_samples_{client_id}_{language}.json")

    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    else:
        data = {}

    if ner_ngram in data:
        data[ner_ngram].extend(values)
    else:
        data[ner_ngram] = values

    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=2)

    # update_ner_ngram_for_key(client_id,language,ner_ngram,values)

    print(f"-----------------JSON file updated for language: {language}, client_id: {client_id}, ner_ngram: {ner_ngram}")



def encode_and_save_data(client_id,lang_code):
    # print(client_id, "----------abc----------------sample")
    # print(lang_code, "----------abc----------------sample")
    input_json_file = os.path.join(json_sample_path, f"ner_samples_{client_id}_{lang_code}.json")
    output_pickle_file = os.path.join(embeddings_sample_path, f"encoded_data_{client_id}_{lang_code}.pkl")

    print(input_json_file, '\n', output_pickle_file)


    with open(input_json_file, 'r') as json_file:
        data = json.load(json_file)


    encoded_data = {}
    for name, tokens in data.items():
        if 'values' in tokens:
            values = tokens['values']
            print(f"Encoding tokens for '{name}': {values}")
            encoded_tokens = model.encode(values, convert_to_tensor=True)
            encoded_data[name] = encoded_tokens.tolist()

    with open(output_pickle_file, 'wb') as pickle_file:
        
        pickle.dump(encoded_data, pickle_file)
        print("-----pkl file saved----ner.utility")




def find_min_max_sentence_lengths(texts):
    min_length = float('inf')
    max_length = 0

    for text in texts:
        sentences = text.split()
        
        for sentence in sentences:
            words = sentence.split()
            length = len(words)
            min_length = min(min_length, length)
            max_length = max(max_length, length)

    return min_length, max_length



def update_ner_ngram_for_key(client_id, language, key,values):
    try:
        min_l, max_l = find_min_max_sentence_lengths(values)

        client = MongoClient("mongodb://your_mongodb_connection_string")
        db = client["your_database_name"]
        collection = db["your_collection_name"]

        conversations = collection.find({"client_id": client_id, "language": language})

        updated_count = 0
        for conversation in conversations:
            slots = conversation.get("slots", {})
            for slot_name, slot in slots.items():
                if slot_name == key:
                    slot["ner_ngram"] = [min_l,max_l]
                    updated_count += 1

            collection.update_one({"_id": conversation["_id"]}, {"$set": {"slots": slots}})

        if updated_count > 0:
            return {"message": f"Updated {updated_count} conversations for key: {key}"}
        else:
            return {"message": f"No conversations found with slot name matching key: {key}"}

    except Exception as e:
        return {"error": str(e)}


  

client_id = '1'
language = 'en'
encode_and_save_queries_to_json(client_id, language)
encode_and_save_data(client_id, language)

# if __name__ == "__main__":
#     main()