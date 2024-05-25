import pandas as pd
import json
from src.config import csv_file_path, FastSentenceTransformer
from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer

from src.modules.intent_utility import INTENT_UTILITY
intent_detector = INTENT_UTILITY() 
import os
model = SentenceTransformer(FastSentenceTransformer, device="cuda", quantize=True) 
from src.config import csv_file_path

csv_file_path= "/home/pritika/travel-chatbot-backup/data/master_dfs/1_en_master_df.csv"
df = pd.read_csv(csv_file_path)
# print(df)


def create_new_df_for_new_usecase(client_id, language, file_path):
    """To create a new csv dataframe to store data.
    Change file_path, folder_path and final dataframe name before running."""
    ith = 1
    master_df_path_file = csv_file_path.format(client_id=client_id, language=language)
    if os.path.exists(master_df_path_file):
        master_df = pd.read_csv(master_df_path_file)
    else:
        master_df = pd.DataFrame(columns=["Sentences", "Keywords", "Intent", "ner_ngram", "NER", "Ner_ngram_length", "Values", "Embeddings", "Keyword_Embeddings"])

    with open(file_path, 'r') as file:
        data = json.load(file)

    dfs = []
    for dataset in data['datasets']:
        df = pd.DataFrame(dataset['data'])
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    # final_df = final_df[final_df['client_id'] == client_id]

    new_rows = []
    for index, data in final_df.iterrows():
        ner_ngram = data["ner_ngram"]
        print("data=========>", data)
        print("ner_ngram==============================>", ner_ngram)
        ner_keys = list(ner_ngram.keys())
        ner_lengths = [len(ner_ngram[key]) for key in ner_keys]

        new_row = {
            "Sentences": data["question"],
            "Keywords": [intent_detector.get_keywords(data["question"])],
            "Intent": data["goal_name"],
            "ner_ngram": data["ner_ngram"],
            "NER": str(ner_keys),
            "Ner_ngram_length": str(ner_lengths),
            "Values": str(data["values"]),
            # "Embeddings": model.encode(data["question"]),
            # "Keyword_Embeddings": intent_detector.create_keyword_embeddings()
        }
        new_rows.append(new_row)
        print(ith, " row saved in master df")
        ith += 1
    new_df = pd.DataFrame(new_rows)
    master_df = pd.concat([master_df, new_df], ignore_index=True)
    master_df.to_csv(master_df_path_file, index=False)
    
# def create_new_df_for_new_usecase(client_id, file_path, destination_folder_path, final_df_name):
#     """To create a new csv dataframe to store data. 
#     Change file_path, folder_path and final dataframe name before running."""

#     with open(file_path, 'r') as file:
#         data = json.load(file)

#     dfs = []
#     for dataset in data['datasets']:
#         df = pd.DataFrame(dataset['data'])
#         dfs.append(df)

#     final_df = pd.concat(dfs, ignore_index=True)

#     # final_df['keyword'] = final_df['question'].apply(extract_keywords))
#     # final_df['embeddings'] = final_df['question'].apply(calculate_embeddings)
#     # final_df['keyword_embeddings'] = final_df['keyword'].apply(lambda x: [calculate_embeddings(keyword) for keyword in x])

#     final_df = final_df[final_df['client_id'] == client_id]

#     # destination_folder_path = '/root/pritika/travel_chatbot/data/'
#     # final_df_name = "travel_data.csv"
#     final_df.to_csv(f"{destination_folder_path}/{final_df_name}", index=False)


def get_ner_ngram_from_df(language, client_id, sentence):
    """
    This function queries the csv dataframe for ner_ngram based on client_id, language and question as matched sentence to the user input
    If present, it returns the ner_ngram value otherwise returns an empty dict
    """
    try:
        
        row = df.loc[df['Sentences'] == sentence]
        #print("row=====>", row)
        if not row.empty and 'ner_ngram' in df.columns:
            return row["ner_ngram"].values[0]
        else:
            return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    # query = df["question"] == sentence
    # print("sentence====>",sentence)
    # print("query====>",query)
    # conversation_doc = df.loc[query]
    # print("conversation_doc = ", conversation_doc)
    # if not conversation_doc.empty and "ner_ngram" in conversation_doc:
    #     return conversation_doc["ner_ngram"].values[0]
    # else:
    #     return {}


def get_conv_data_from_goal(language, client_id, sentence):
    """
    This function queries the csv dataframe bases on client_id, language and question as matched sentence to the user input
    If present, it returns the conversation_doc otherwise returns False
    
    """

    query = (df["client_id"] == client_id) & (df["language"] == language) & (df["question"] == sentence) 
    conversation_doc = df.loc[query]
    if not conversation_doc.empty:
        return conversation_doc
    return False


def get_intent_from_df(sentence, language, client_id):

    query = (df["client_id"] == client_id) & (df["language"] == language) & (df["question"] == sentence)
    conversation_doc = df.loc[query]
    if not conversation_doc.empty:
        return conversation_doc["goal_name"].values[0]
    return False


def get_current_goal(public_id):

    query = df["public_id"] == public_id
    conversation_doc = df.loc[query]
    if not conversation_doc.empty and "current_goal_name" in conversation_doc:
        return conversation_doc["current_goal_name"].values[0]


def set_intent(goal, public_id):
    try:
        df.loc[df["public_id"] == public_id, ["current_goal_name", "new_intent"]] = [goal, True]
        print(f"Setting intent for public_id '{public_id}' updated in CSV.")
        return True
    except Exception as e:
        print(f"Setting intent for public_id '{public_id}' not found in CSV.")
        return False


def set_conversation_data(data, public_id):
    try:
        df.loc[df["public_id"] == public_id, "data"] = data
        print(f"Setting conversation for public_id '{public_id}' updated in CSV.")
        return True
    except Exception as e:
        print(f"Setting conversation for public_id '{public_id}' not found in CSV.")
        return False

def fill_ner_values(public_id, keys_and_values):
    try:
        current_document = df.loc[df["public_id"] == public_id]
        slots_to_ask = current_document["slots_to_ask"].values[0]

        updated_slots_to_ask = [
            slot for slot in slots_to_ask if slot not in keys_and_values.keys()
        ]

        df.loc[df["public_id"] == public_id, "slots_to_ask"] = updated_slots_to_ask

        goal_slots = current_document.get("goal_slots", [])
        for slot_name, slot_value in keys_and_values.items():
            if slot_name not in goal_slots:
                goal_slots.append({slot_name: slot_value[0]})

        df.loc[df["public_id"] == public_id, "goal_slots"] = goal_slots

        return True

    except Exception as e:
        print(f"fill_ner_values An error occurred: {str(e)}")
        return False


def set_slot_keys(public_id, data):
    try:
        df.loc[df["public_id"] == public_id, "slots_to_ask"] = data

        print(
            f"Setting slot ner_object for  public_id '{public_id}' updated in CSV."
        )
        return True

    except Exception as e:
        print(f"set_slot_keys An error occurred: {str(e)}")
        return False


def set_ner_true(public_id):
    print("Setting ner_object to -> True")
    try:
        df.loc[df["public_id"] == public_id, "ner_done"] = True

        print(
            f"Setting slot ner_object for  public_id '{public_id}' updated in CSV."
        )
        return True
    except Exception as e:
        print(f"set_ner_true An error occurred: {str(e)}")
        return False


def get_slots_to_ask(public_id):
    try:
        slots_to_ask = df.loc[df["public_id"] == public_id, "slots_to_ask"].values[0]

        return slots_to_ask if slots_to_ask else []

    except Exception as e:
        print(f"get_slots_to_ask An error occurred: {str(e)}")
        return []


def get_slots_to_ask_(public_id):
    try:
        slots_to_ask = df.loc[df["public_id"] == public_id, "slots_to_ask"].values[0]

        return slots_to_ask if slots_to_ask else []

    except Exception as e:
        print(f"get_slots_to_ask_ An error occurred: {str(e)}")
        return []


def store_conversation_logic(data):
    try:
        print("store_conversation_logic: Success")
    except Exception as e:
        print(f"store_conversation_logic An error occurred: {str(e)}")


