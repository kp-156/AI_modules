import pickle
from sentence_transformers import util
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import requests
import torch
import numpy as np
import os
import sys
import ast
import time
from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer
from src.utils.all_utils.common_util import LoadAndSaveFiles, IntentNerClassFunctions

# print(sys.path)
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add the project root directory to the Python path
from ..config import (
    client_dataframe_path,
    client_embedding_path,
    client_keyword_embedding_path,
    process_text_api_url,
    embeddings_api_url,
    csv_file_path,
    FastSentenceTransformer
)


script_dir = os.path.dirname(os.path.realpath(__file__))


class IntentDetector:
    """
    This module provides some functions to detect the intent of a given sentence based on local embeddings or
    embeddings from the API.
    """

    def __init__(self, language="en", client_id="1"):
        self.language = language
        self.client_id = client_id
        self.device = torch.device('cuda:0')
        # self.device = "cpu"

        self.client_dataframe_file_path = client_dataframe_path.format(client_id=client_id, language=language)
        self.dataframe = self.load_client_dataframe()

        self.client_embedding_file_path = client_embedding_path.format(client_id=client_id, language=language)
        self.embeddings_dict = self.load_embeddings()  # This is actually a list

        self.client_keyword_embedding_file_path = client_keyword_embedding_path.format(client_id=client_id, language=language)
        self.keyword_embeddings = self.load_embeddings_keywords()

        self.util_obj = IntentNerClassFunctions(language=self.language, client_id=self.client_id)
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
            return LoadAndSaveFiles.load_pickle(self.client_embedding_file_path)
        except FileNotFoundError:
            print(f"Embeddings file for {self.client_id}-{self.language} not found. Please provide the file.")
            return {}

        # try:
        #     df = pd.read_csv(self.master_df)
        #     if "Embeddings" in df.columns:
        #         embeddings_list = df["Embeddings"].tolist()
        #         return embeddings_list
        #     else:
        #         print(f"Embeddings column not found in DataFrame at {self.master_df}.")
        #         return []
        # except FileNotFoundError:
        #     print(f"DataFrame file not found at {self.master_df}. Please provide the correct file path.")
        #     return []

    def load_embeddings_keywords(self):
        """
        This function loads the keyword embeddings from the a local pre-trained model based on client_id and language.
        Returns empty dictionary if the keyword embedding file is not present
        :return:
        """
        
        try:
            return LoadAndSaveFiles.load_pickle(self.client_keyword_embedding_file_path)
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

    def get_top_sentence_candidates(self, input_sentence, num_candidates=5):
        """
        This function returns the list of top sentence candidates (from the API) based on the cosine similarity between
        the input sentence and the sentences in the dataframe.

        If there are no local embeddings, it will return an empty dictionary.
        If there are no embeddings from the API, it will return an empty dictionary.
        :param input_sentence:
        :param num_candidates:
        :return:
        """
        if len(self.embeddings_dict) == 0:
            print("Could not find top sentence candidates because of empty embedding dict")
            return {}

        input_embedding = self.util_obj.get_input_embedding_from_api(input_sentence)
        if input_embedding is None:
            print("Could not find top sentence candidates because of input embedding from API is not available")
            return {}

        input_embedding_tensor = input_embedding.to(self.device)
        embeddings_dict_tensor = torch.tensor(self.embeddings_dict, dtype=torch.float).to(self.device)
        similarity_scores = util.pytorch_cos_sim(input_embedding_tensor, embeddings_dict_tensor)
        similarity_scores = similarity_scores.squeeze()

        if similarity_scores.ndim == 0:
            similarity_scores = similarity_scores.reshape(1)

        # Sort the candidates by their score and select num_candidates with highest score
        top_candidates_indices = similarity_scores.argsort()[-num_candidates:]

        # Covert the filtered tensor to normal list
        top_candidates_indices = [index.item() for index in top_candidates_indices]

        top_candidates = {}
        for index in top_candidates_indices:
            sentence = self.dataframe.loc[index, "Sentences"]
            intent = self.dataframe.loc[index, "Intent"]
            score = similarity_scores[index].item()
            top_candidates[sentence] = {"score": score, "intent": intent}

        top_candidates = dict(sorted(top_candidates.items(), key=lambda x: x[1]['score'], reverse=True))
        return top_candidates
    

    def extract_keywords_for_sentences(self, sentences_to_find):
        """
        Returns a dict with keywords as values of top five sentences as keys
        """
        try:
            sentence_to_keywords = dict(zip(self.dataframe["Sentences"], self.dataframe["Keywords"]))
            results = {}
            for sentence in sentences_to_find:
                keywords = eval(sentence_to_keywords.get(sentence))
                results[sentence] = keywords

            return results
        except Exception as e:
            print(f"An error occurred in extract_keywords_for_sentences: {str(e)}")
            return {}     

    def match_input_keywords_with_sentences(self, input_sentence_keywords, top_candidate_keywords_dict):
        """
        This function matches the keywords of the input sentence with keywords of the top candidate sentences
        based on keyword embeddings.
        :param input_sentence_keywords:
        :param top_candidate_keywords_dict:
        :return:
        """
        input_keywords = input_sentence_keywords
        sentence_match_scores = {}
        #print("---------------match_input_keywords_with_sentences------------------------")
        # input_keyword_embeddings = [keyword_embeddings[keyword] for keyword in input_keywords]
        input_keyword_embeddings = [self.util_obj.get_input_embedding_from_api(keyword) for keyword in input_keywords]

        for sentence, keywords in top_candidate_keywords_dict.items():
            sentence_keywords_embeddings = [
                torch.from_numpy(self.keyword_embeddings[keyword]).to(self.device)
                for keyword in keywords
            ]

            match_score = 0
            for input_embedding in input_keyword_embeddings:
                if input_embedding is not None:
                    similarity_scores = np.array([
                        self.util_obj.calculate_cosine_similarity(input_embedding, sentence_embedding)
                        for sentence_embedding in sentence_keywords_embeddings
                        if sentence_embedding is not None
                    ])
                    match_score += np.sum(similarity_scores > 0.7)

            sentence_match_scores[sentence] = match_score
        return sentence_match_scores

    def match_sentences_with_keywords(self, input_sentence_keywords, sentences_list):
        """
        This function matches the keywords of the input sentence with keywords of the top candidate sentences
        based on API embeddings
        based on keyword embeddings.
        :param input_sentence_keywords:
        :param sentences_list:
        :return:
        """
        input_keywords = input_sentence_keywords
        matching_sentences = []

        input_keyword_embeddings = [self.util_obj.get_input_embedding_from_api(keyword) for keyword in input_keywords]
        for sentence in sentences_list:
            sentence_keywords = self.extract_keywords_for_sentences([sentence]).get(sentence, [])
            if not sentence_keywords:
                continue
            #print("---------------match_sentences_with_keywords------------------------")
            sentence_keyword_embeddings = [self.util_obj.get_input_embedding_from_api(keyword) for keyword in sentence_keywords]
            all_keywords_match = all(
                any(
                    self.util_obj.calculate_cosine_similarity(input_embedding, sentence_embedding) > 0.65
                    for sentence_embedding in sentence_keyword_embeddings
                    if sentence_embedding is not None
                )
                for input_embedding in input_keyword_embeddings
                if input_embedding is not None
            )

            if all_keywords_match:
                matching_sentences.append(sentence)

        return matching_sentences

    def get_intent(self, keys):
        df = self.dataframe
        op = []
        for key in keys:
            op.extend(
                df[df["Sentences"].str.lower() == key.lower()]["Intent"].to_list()
            )
        return op

    # def process_long_text_intent(self, text, threshold=0.5):
    #     top_candidates = self.get_top_sentence_candidates(text)
    #     # print("Top 5 Sentences are -->",top_candidates,"\n")
    #     top_candidates_sentences = [
    #         k for k, v in top_candidates.items() if v > threshold
    #     ]
    #     # print("Top Sentences after threshold  are -->",top_candidates_sentences,"\n")
    #     if not top_candidates_sentences:
    #         return None, None

    #     elif len(top_candidates_sentences) == 1:
    #         return top_candidates_sentences[0], None
    #     else:
    #         top_candidates_keywords = self.extract_keywords_for_sentences(top_candidates_sentences)
    #         input_sentence_keywords = self.util_obj.get_keywords(text, self.language)
    #         result_intent = self.match_input_keywords_with_sentences(text, input_sentence_keywords, top_candidates_keywords)
    #         multi_intent = self.get_intent(result_intent)
    #         # print("Multi Intent result is -->",multi_intent,"\n")

    #         max_value = max(result_intent.values())
    #         max_keys = [key for key, value in result_intent.items() if value == max_value]
    #         # print("Intent found is -->",max_keys)

    #         intent = self.get_intent(max_keys)

    #         return list(set(multi_intent))

    def update_dataframe(self, sentence, goal_name):
        """
        This function saves the appends the new sentence, keyword and intent data
        into the existing client dataframe and updates the dataframe csv file
        """
        keywords = self.util_obj.get_keywords(sentence, self.language)
        print("Intent. Update dataframe: Keyword are ->", keywords)

        new_data = pd.DataFrame(
            {"Sentences": [sentence], "Keywords": [keywords], "Intent": [goal_name]}
        )
        combined_df = pd.concat([self.dataframe, new_data], ignore_index=True)
        combined_df.to_csv(self.client_dataframe_file_path, index=False)

    def update_embeddings(self):
        """
        This function reads the latest client dataframe, generates its embeddings
        from the sentence transformer package, and saves the updated embeddings to the file system.
        """
        client_df = pd.read_csv(self.client_dataframe_file_path)

        
        # model = SentenceTransformer(self.model_path)
        embeddings = []

        for sentence in client_df["Sentences"]:
            embedding = self.model.encode(sentence, convert_to_tensor=True)  
            embedding = embedding.cpu().numpy()
            embeddings.append(embedding)
        embeddings_array = np.array(embeddings)

        with open(self.client_embedding_file_path, "wb") as f:
            pickle.dump(embeddings_array, f)
        print(f"Embeddings saved to {self.client_embedding_file_path}")
        
    def update_keyword_embeddings(self):
        """
        This function reads the latest client dataframe, generates its keyword embeddings
        from the sentence transformer package, and saves the updated keyword embeddings to the file system.
        """
        client_df = pd.read_csv(self.client_dataframe_file_path)

     
        keyword_embeddings = {}

        for index, row in client_df.iterrows():
            keywords = eval(row["Keywords"])
            for k in keywords:
                embeddings = self.model.encode(k)
                keyword_embeddings[k] = embeddings
        
        # return keyword_embeddings
        with open(self.client_keyword_embedding_file_path, "wb") as file:
            pickle.dump(keyword_embeddings, file)
        print(f"Keyword embeddings saved to {self.client_keyword_embedding_file_path}")
       
