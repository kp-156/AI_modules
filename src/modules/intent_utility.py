# import sys
# sys.path.append("/root/pritika/travel_chatbot/src")
# sys.path.append("/root/pritika/travel_chatbot/src/modules")

import os
import pandas as pd
import torch
import numpy as np
import pickle
from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from src.config import(
    client_dataframe_path,
    client_embedding_path,
    client_keyword_embedding_path,
    FastSentenceTransformer
)
# from intent import IntentDetector
# client_dataframe_path = "/home/sdb/pritika/travel_chatbot/data/client_dataframes/{client_id}_{language}_dataframe"
# client_embedding_path = "/home/sdb/pritika/travel_chatbot/data/client_embeddings/{client_id}_{language}_embeddings.pkl"
# client_keyword_embedding_path = "/home/sdb/pritika/travel_chatbot/data/client_keyword_embeddings/{client_id}_{language}_keywords.pkl"
# FastSentenceTransformer = "all-MiniLM-L6-v2"


class INTENT_UTILITY:
    def __init__(self, language="en", client_id="1"):
        self.language = language
        self.client_id = client_id
        self.device = torch.device('cuda:0')
        # self.int_obj = IntentDetector()

        self.client_dataframe_file_path = client_dataframe_path.format(client_id=client_id, language=language)
        self.dataframe = self.load_client_dataframe()

        self.client_embedding_file_path = client_embedding_path.format(client_id=client_id, language=language)
        self.embeddings_dict = self.load_embeddings()  # This is actually a list

        self.client_keyword_embedding_file_path = client_keyword_embedding_path.format(client_id=client_id, language=language)
        self.keyword_embeddings = self.load_embeddings_keywords()


        self.model = SentenceTransformer(FastSentenceTransformer, device="cuda", quantize=True)

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
            with open(self.client_embedding_file_path, "rb", ) as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Embeddings file for {self.client_id}-{self.language} not found. Please provide the file.***")
            return {}

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
            print(f"Keyword embeddings file for {self.client_id}-{self.language} not found. Please provide the file.***")
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
        """
        This function gets the keywords of the sentence using the API.
        :param sentence:
        :return:

        NOTE: the url is not returning the keyword. the problem is in the API.
        """
        # start_time = time.time()
        data = {"text": sentence, "language": self.language}
        # try:
        #     response = requests.post(process_text_api_url, json=data).json()["keywords"]
        #     print('Process text API response', response)
        #     # print("For keywords --------- %s seconds ----------" % (time.time() - start_time))
        #     return response
        
        # except Exception as e:
        # print(f"An error occurred in get_keywords function while calling process text API:\n{str(e)}")
        # print("Returning all the words of the sentence")
        vectorizer = TfidfVectorizer(max_features=10)  
        tfidf_vector = vectorizer.fit_transform([sentence])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_vector.toarray()[0]
        sorted_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
        # print("For keywords --------- %s seconds ----------" % (time.time() - start_time))
        return [keyword for keyword, score in sorted_keywords]
    

    def update_dataframe(self, sentence, goal_name):
        
        if os.path.exists(self.client_dataframe_file_path):
            df = pd.read_csv(self.client_dataframe_file_path)
        else:
            df = pd.DataFrame(columns=["Sentences", "Keywords","Intent"])
        keywords = self.get_keywords(sentence)
        new_data = pd.DataFrame({"Sentences": [sentence], "Keywords": [keywords], "Intent": [goal_name]} )
        df = pd.concat([df, new_data], ignore_index=True)

        #df = df.append({"Sentences": sentence, "Keywords": keywords}, ignore_index=True)
        df.to_csv(self.client_dataframe_file_path, index=False)
        print("dataframe updated for, ", self.client_dataframe_file_path)

    def create_embeddings(self):
        client_df = pd.read_csv(self.client_dataframe_file_path)

        embeddings = []
        for sentence in client_df["Sentences"]:
            embedding = self.model.encode(sentence, convert_to_tensor=True)  
            embedding = embedding.cpu().numpy()
            embeddings.append(embedding)
        embeddings_array = np.array(embeddings)

        with open(self.client_embedding_file_path, "wb") as f:
            pickle.dump(embeddings_array, f)
        print(f"Embeddings saved to {self.client_embedding_file_path}")
        


    def create_keyword_embeddings(self):
        client_df = pd.read_csv(self.client_dataframe_file_path)

        keyword_embeddings = {}
        for index, row in client_df.iterrows():
            keywords = eval(row["Keywords"])
            for k in keywords:
                embeddings = self.model.encode(k)
                keyword_embeddings[k] = embeddings
        
        with open(self.client_keyword_embedding_file_path, "wb") as file:
            pickle.dump(keyword_embeddings, file)
        print(f"Keyword embeddings saved to {self.client_keyword_embedding_file_path}")
        return keyword_embeddings
       

# create_embeddings("en", "1")