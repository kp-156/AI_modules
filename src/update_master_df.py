import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import json
import pickle
import os
import numpy as np
from config import (
    client_dataframe_path,
    client_embedding_path,
    client_keyword_embedding_path,
    ner_model_cluster,
    ner_model_samples,
    ner_embeddings_cluster,
    ner_embeddings_samples,
    ner_json_cluster,
    ner_json_samples
)
# from modules.intent import IntentDetector
# from modules.ner import NER

class ClientData:
    def __init__(self, client_id, language, folder_path, text):
        self.client_id = client_id
        self.language = language
        self.folder_path = folder_path
        self.model = ner_model_cluster

        # self.intent_detector = IntentDetector(language=language, client_id=client_id)
        # self.ner_object = NER(text=text, language=language, client_id=client_id)

        self.client_dataframe_file_path = client_dataframe_path.format(client_id=client_id, language=language)
        self.dataframe = self.load_client_dataframe()

        self.client_embedding_file_path = client_embedding_path.format(client_id=client_id, language=language)
        self.embeddings_dict = self.load_embeddings()  # This is actually a list

        self.client_keyword_embedding_file_path = client_keyword_embedding_path.format(client_id=client_id, language=language)
        self.keyword_embeddings = self.load_embeddings_keywords()

        self.ner_samples_json_path = ner_json_samples.format(client_id=client_id, language=language)
        self.ner_cluster_json_path = ner_json_cluster.format(client_id=client_id, language=language)
        self.ner_embeddings_samples_path = ner_embeddings_samples.format(client_id=client_id, language=language)
        self.ner_embeddings_cluster_path = ner_embeddings_cluster.format(client_id=client_id, language=language)


    def load_embeddings(self):
        """
        This function loads the embeddings from the a local embeddings based on client_id and language.
        Returns empty list if the file is not found.
        Note: Currently the embeddings file is saved as a list of lists. And mapping to sentences is assumed through
        index in a separate file (client dataframe csv). This approach error prone.
        So, suggested approach is to store the sentences as keys and embeddings as values in a dict
        :return:
        """
        try:
            with open(self.client_embedding_file_path, "rb",) as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Embeddings file for {self.client_id}-{self.language} not found. Please provide the file.")
            return []

    def load_embeddings_keywords(self):
        """
        This function loads the keyword embeddings from the a local pre-trained model based on client_id and language.
        Returns empty dictionary if the keyword embedding file is not present
        :return:
        """
        try:
            with open(self.client_keyword_embedding_file_path, "rb", ) as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Keyword embeddings file for {self.client_id}-{self.language} not found. Please provide the file.")
            return {}

    def load_client_dataframe(self):
        """
        This function loads the client dataframe from the a local csv file based on client_id and language.
        :return:
        """
        if os.path.exists(self.client_dataframe_file_path):
            df = pd.read_csv(self.client_dataframe_file_path)
        else:
            df = pd.DataFrame(columns=["Sentences", "Keywords", "Intent"])
        return df
    
    def get_keywords(self, sentence):
        """ This function extracts keywords from a sentence using TF-IDF.
            Args: sentence (str): The sentence to extract keywords from.
            Returns: list: A list of top keywords for the sentence."""
        
        vectorizer = TfidfVectorizer(max_features=10)  
        tfidf_vector = vectorizer.fit_transform([sentence])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_vector.toarray()[0]
        sorted_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, score in sorted_keywords]
            
    def load_ner_cluster_json(self):
        """
        This function loads the latest NER cluster JSON data from file. Returns empty dict if file doesn't exist.
        """
        if os.path.exists(self.ner_cluster_json_path):
            with open(self.ner_cluster_json_path, "r") as file:
                data = json.load(file)
        else:
            print("Could not load existing NER cluster JSON. Returning empty dictionary")
            data = {}
        return data

    def load_ner_sample_json(self):
        """
        This function loads the latest NER samples JSON data from file. Returns empty dict if file doesn't exist.
        """
        if os.path.exists(self.ner_samples_json_path):
            with open(self.ner_samples_json_path, "r") as file:
                data = json.load(file)
        else:
            print("Could not load existing NER samples JSON. Returning empty dictionary")
            data = {}
        return data

    def update_dataframe(self, sentence, goal_name):
        """
        This function appends the new sentence, keyword, and intent data
        into a pandas dataframe.
        """
        keywords = self.get_keywords(sentence)

        new_data = pd.DataFrame(
            {"Sentences": [sentence], "Keywords": [keywords], "Intent": [goal_name]}
        )
        self.dataframe = pd.concat([self.dataframe, new_data], ignore_index=True)


    def update_embeddings(self):
        """
        This function reads the dataframe, generates its embeddings
        from the sentence transformer package, and adds them to the dataframe.
        """
        embeddings = []
        for sentence in self.dataframe["Sentences"]:
            embedding = self.model.encode(sentence, convert_to_tensor=True)
            embedding = embedding.cpu().numpy()
            embeddings.append(embedding)
        embeddings_array = np.array(embeddings)
        self.dataframe["Embeddings"] = embeddings_array


    def update_keyword_embeddings(self):
        """
        This function reads the dataframe, generates keyword embeddings
        from the sentence transformer package, and adds them to the dataframe.
        """
        keyword_embeddings = {}
        for index, row in self.dataframe.iterrows():
            keywords = eval(row["Keywords"])
            for k in keywords:
                embedding = self.model.encode(k)
                keyword_embeddings[k] = embedding.tolist() 
        self.dataframe["Keyword Embeddings"] = keyword_embeddings


    def update_ner_cluster_json(self, ner_ngram, values):
        data = self.ner_object.load_ner_cluster_json()
        ner_key = tuple(ner_ngram)  # converted ner_ngram into a tuple because ner_ngram is a list which is unhashable

        if ner_key in data:
            data[ner_key].extend(values)
        else:
            print(f"Adding new key to NER cluster JSON: {ner_key}")
            data[ner_key] = values
        data_converted = {str(key): value for key, value in data.items()}  # Converted data back to list? TODO: Check

        ner_key = str(tuple(ner_ngram))  
        ner_data_frame = pd.DataFrame.from_dict(data, orient='index', columns=['Values'])
        ner_data_frame.index.name = 'NER Key'
        self.dataframe = pd.concat([self.dataframe, ner_data_frame], axis=1)


    def update_ner_cluster_embeddings(self):
        data = self.ner_object.load_ner_cluster_json()
        embeddings_dict = {}
        model = SentenceTransformer(ner_model_cluster)

        for category, sentences in data.items():
            query_embeddings = model.encode(sentences, convert_to_tensor=True)
            embeddings_dict[category] = query_embeddings.tolist()

        self.dataframe['NER Cluster Embeddings'] = embeddings_dict


    def update_ner_sample_json(self, ner_ngram, values):
        data = self.ner_object.load_ner_sample_json()

        if ner_ngram in data:
            data[ner_ngram].extend(values)
        else:
            data[ner_ngram] = values

        ner_samples_frame = pd.DataFrame.from_dict(data, orient='index', columns=['Values'])
        ner_samples_frame.index.name = 'NER Ngram'
        self.dataframe = pd.concat([self.dataframe, ner_samples_frame], axis=1)

    def update_ner_sample_embeddings(self):
        data = self.ner_object.load_ner_sample_json()
        encoded_data = {}
        model = SentenceTransformer(ner_model_samples)

        for name, tokens in data.items():
            if "values" in tokens:
                values = tokens["values"]
                print(f"Encoding tokens for '{name}': {values}")
                encoded_tokens = model.encode(values, convert_to_tensor=True)
                encoded_data[name] = encoded_tokens.tolist()

        self.dataframe['NER Sample Embeddings'] = encoded_data


    def save_data(self):
        """
        This function saves the combined dataframe as a CSV file.
        """
        self.dataframe.to_csv(self.client_dataframe_file_path, index=False)



    def run(self, sentence, goal_name):
        """
        This function performs all the updates and saves the data.
        """
        self.update_dataframe(sentence, goal_name)
        self.update_embeddings()
        self.update_keyword_embeddings()
        self.update_ner_cluster_embeddings()
        self.update_ner_sample_embeddings()
        self.save_data()

def main():
    client_id = "1"
    language = "en"
    folder_path = "/root/pritika/travel_chatbot/data/master_dfs"
    text = "I want to book a flight."
    goal_name = "booking_flight"
    
    client_data = ClientData(client_id, language, folder_path, text)
    client_data.run(text, goal_name)

if __name__ == "__main__":
  main()