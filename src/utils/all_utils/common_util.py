import pymongo
import os
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
import time
import pickle
import json
import csv
import requests
from sentence_transformers import SentenceTransformer, util
import sys
import re
import traceback
from src.config import(
    embeddings_api_url,
    process_text_api_url,
    splitter_path
)
# import standard_phrases as sp

class EXECUTORS_UTIL:

    #logger.py
    def execute_and_log(self, func, *args, **kwargs):
        """
        Executes the given function with the provided arguments and logs the result.

        Args:
            func (function): The function to be executed.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict: A dictionary containing the execution result, success status, and output.

        """
        result = {'success': False, 'result': None, 'output': None}
        try:
            result['output'] = func(*args, **kwargs)
            result['success'] = True
            result['result'] = result['output']
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            result['result'] = {
                'exception_type': str(exc_type),
                'exception_value': str(exc_value),
                'traceback': traceback.format_exception(exc_type, exc_value, exc_traceback)
            }
        return result
    
    #executor.py
    def replace_keys_in_nested_dict(self, dict_, old_char, new_char):
        """
        Recursively replaces keys in a nested dictionary.

        Args:
            dict_ (dict): The input dictionary.
            old_char (str): The character to be replaced in the keys.
            new_char (str): The character to replace the old character with.

        Returns:
            dict: The modified dictionary with replaced keys.
        """
        new_dict = {}
        for key, value in dict_.items():
            new_key = key.replace(old_char, new_char)
            if isinstance(value, dict):
                new_value = self.replace_keys_in_nested_dict(value, old_char, new_char)
            else:
                new_value = value
            new_dict[new_key] = new_value
        return new_dict

    #Data Processing:

    def normalize_text(self, text):
        """
        Normalize the given text by converting it to lowercase and removing any non-alphanumeric characters.

        Args:
            text (str): The text to be normalized.

        Returns:
            str: The normalized text.

        """
        normalized_text = text.lower()
        normalized_text = re.sub(r'[^\w\s]', '', normalized_text)
        return normalized_text


class LoadAndSaveFiles:
    
    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    @staticmethod
    def load_pickle(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    @staticmethod
    def load_csv(file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            data = [row for row in reader]
        return data

    @staticmethod
    def save_json(data, file_path):
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    @staticmethod 
    def save_pickle(data, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    @staticmethod
    def save_csv(data, file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)



class IntentNerClassFunctions:

    def __init__(self, language, client_id):
        self.language = language
        self.client_id = client_id
        self.device = torch.device('cuda:0')


    def get_input_embedding_from_api(self, input_sentence):
        """
        Retrieves the input embedding from an API.

        Args:input_sentence (str): The input sentence to be sent to the API.

        Returns: torch.Tensor or None: The input embedding as a torch.Tensor if successful, None otherwise.
        """
        api_url = embeddings_api_url
        response = requests.post(api_url, json={'texts': input_sentence})

        if response.status_code == 200:
            input_embedding = torch.tensor(response.json()['embeddings'], dtype=torch.float).to(self.device)
            return input_embedding
        else:
            for one_text in input_sentence: 
                embeddings = self.get_embeddings_for_user_input(one_text)
                if embeddings is not None:
                    input_embeddings = torch.tensor(embeddings, dtype=torch.float).to(self.device)
                    return input_embeddings
        return None

    def get_embeddings_for_user_input(self, texts):
        try:
            if isinstance(texts, str):
                texts = [texts]

            embeddings = self.model.encode(texts, convert_to_tensor=True)
            embeddings = embeddings.tolist()  
            return embeddings
        
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return None
        
    def get_keywords(self, sentence, language):
        """
        Retrieves the keywords from a given sentence using a specified language.

        Args:   sentence (str): The sentence from which to extract keywords.
                language (str): The language code used for keyword extraction.

        Returns: dict: A dictionary containing the extracted keywords.
        """
        data = {"text": sentence, "language": language}
        try:
            response = requests.post(process_text_api_url, json=data).json()["keywords"]
            print('Process text API response', response)
            return response
        
        except Exception as e:
            vectorizer = TfidfVectorizer(max_features=10)  
            tfidf_vector = vectorizer.fit_transform([sentence])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_vector.toarray()[0]
            sorted_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
            return [keyword for keyword, score in sorted_keywords]

    def generate_ngrams(self, sentence, n):
        """
        Generate ngrams from the given sentence based on the specified size of ngram 'n',
        e.g., sentence "I want to visit Japan" and n = 2 would give us: ["I want", "want to", "to visit", "visit Japan"]
        """
        words = sentence.split()
        ngrams = []
        if n <= 0:
            return []
        if n > len(words):
            return [" ".join(words)]
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i: i + n])
            ngrams.append(ngram)
        return ngrams

    def extract_max_score_entry(self, data):
        """
        Extracts the entry with the maximum score for each category from the given data.

        Args:
            data (dict): A dictionary containing sentence information categorized by category.

        Returns:
            dict: A dictionary containing the entry with the maximum score for each category.
                The keys are the categories and the values are dictionaries containing the category,
                score, and sentence of the entry with the maximum score.
        """
        category_scores = {}

        for sentence, info in data.items():
            category = info['category']
            score = info['score']

            if category not in category_scores or score > category_scores[category]['score']:
                category_scores[category] = {'sentence': sentence, 'score': score}

        result = {category: {'category': category, 'score': info['score'], 'sentence': info['sentence']} for category, info in category_scores.items()}
        return result


    def calculate_cosine_similarity(self, embedding1, embedding2):
        """
        Calculates the cosine similarity between two embeddings.

        Parameters:
            embedding1 (torch.Tensor): The first embedding.
            embedding2 (torch.Tensor): The second embedding.

        Returns:
            float: The cosine similarity between the two embeddings.
        """
        return util.pytorch_cos_sim(embedding1, embedding2).item()
  

    def get_criteria_keys(self):
        """
        Retrieves the criteria keys from a JSON file.

        Returns:
            A list of criteria keys.
        """
        raw_filename = transcription_scorer_raw_data_json
        with open(raw_filename, 'r') as raw_file:
            data = json.load(raw_file)

        keys = list(data.keys())
        print(f"Criteria points are ---> {keys}")
        return keys

    def calculate_similarity_score_for_category(self, text):
        """
        Calculates the similarity scores for each category based on the input text.

        Args:
            text (str): The input text for which similarity scores need to be calculated.

        Returns:
            dict: A dictionary containing the similarity scores for each category.
                  The keys are the category names and the values are the corresponding similarity scores.
        """
        similarity_scores = {}
        input_embedding = self.get_input_embedding_from_api(text)
        criteria_points = self.get_criteria_keys()

        for category in criteria_points:
            embeddings_cluster_dict = LOADFILES_UTILS.load_embeddings_cluster()
            embeddings = embeddings_cluster_dict.get(category, [])
            embeddings = torch.tensor(embeddings, dtype=torch.float).to(self.device)
            print(f"Embeddings is{embeddings.shape} None for category {category}")
            similarities = util.pytorch_cos_sim(input_embedding, embeddings)
            sum_similarity_score = torch.sum(similarities)
            mean_similarity_score = sum_similarity_score / len(embeddings)

            similarity_scores[category] = mean_similarity_score.item()

        return similarity_scores

    def process_text_with_category(self, splitter_path, ners):
        """
        Process the text with category.

        Args:
            splitter_path (str): The path to the splitter.
            ners (list): List of named entities.

        Returns:
            dict: A dictionary containing the processed text with their corresponding category.

        """
        splitter_keys = self.get_splitter(lang=self.language, splitter_path=splitter_path)
        split_phrases = self.phrase_splitter(phrase_splitter_key=splitter_keys)
        results = {}

        for text in split_phrases:
            similarity_scores = self.calculate_similarity_score_for_category(text=text, ners=ners)
            results[text] = similarity_scores

        result = {sentence: max(scores, key=scores.get) for sentence, scores in results.items()}
        return result
        

    def trim_str(self, some_list):
        """
        Trims the given list if its string representation exceeds 499 characters.
        
        Parameters:
        some_list (list): The list to be trimmed.
        
        Returns:
        str: The trimmed list as a string.
        """
        if len(str(some_list)) > 499:
            some_list.pop()
            if some_list[0] != 'trimmed':
                return self.trim_str(['trimmed']+some_list)
            else:
                return self.trim_str(some_list)
        else:
            return str(some_list)

    def get_ngrams(self, text, n=4, step=1):
        """
        Returns n-grams for a given string as a list of strings.

        Parameters:
        text (str): The input string.
        n (int): The number of words in each n-gram. Default is 4.
        step (int): The step size between consecutive n-grams. Default is 1.

        Returns:
        list: A list of n-grams as strings.
        """
        words = text.split()
        ngrams = [' '.join(words[i:i+n]) for i in range(0, len(words) - n + 1, step) if len(words[i:i+n]) == n]

        return ngrams

    def data_to_insert_in_key_table(self, call_id, category, sub_category, list_of_list):
        """
        Converts a list of lists into a list of tuples with additional information.

        Args:
            call_id (int): The ID of the call.
            category (str): The category of the data.
            sub_category (str): The sub-category of the data.
            list_of_list (list): A list of lists containing the data.

        Returns:
            list: A list of tuples with the following structure:
                [(call_id, category, sub_category, list_item[0], list_item[1]) for list_item in list_of_list]
        """
        return [(call_id, category, sub_category, list_item[0], list_item[1]) for list_item in list_of_list]

    def get_primary_intent(self, outer_dict):
        """
        Returns the key with the highest score from the given dictionary.

        Args:
            outer_dict (dict): The dictionary containing the intents and their scores.

        Returns:
            str: The key with the highest score.

        """
        max_key = None
        max_score = 0 # float('-inf')  # Initialize with negative infinity

        for key, value in outer_dict.items():
            if 'score' in value:
                score = value['score']
                if score > max_score:
                    max_score = score
                    max_key = key
        return max_key
