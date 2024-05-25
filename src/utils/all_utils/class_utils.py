import torch
from sentence_transformers import SentenceTransformer, util
import requests
import json
from z_all_utils.load_files_utils import load_embeddings_cluster
import sys
sys.path.append('/root/pradheep/transcription_scorer/api_tsc')

import standard_phrases as sp

def phrase_splitter(self,text, phrase_splitter_key):
        for key in phrase_splitter_key:
            text = text.replace(' ' + key.strip() + ' ', '<split>' + ' ' + key.strip() + ' ')
        return text.split('<split>')
    
def get_splitter(self,splitter_path, language):
        with open(splitter_path, 'r', encoding='utf-8') as json_file:
                splitter_json = json.load(json_file)
        split = splitter_json[language]
        return split


def get_input_embedding_from_api(self, input_sentence):
        api_url = get_input_embedding_from_api
        response = requests.post(api_url, json={'texts': input_sentence})

        if response.status_code == 200:
                input_embedding = torch.tensor(response.json()['embeddings'], dtype=torch.float).to(self.device)
                return input_embedding
        else:
                print("Error: Unable to get input embedding from the API")
                return None

def calculate_cosine_similarity(self,embedding1, embedding2):
        return util.pytorch_cos_sim(embedding1, embedding2).item()


def calculate_similarity_score_for_category(self, text):
        similarity_scores = {}
        input_embedding = self.get_input_embedding_from_api(text)
        criteria_points = self.get_criteria_keys()

        for category in criteria_points:
            embeddings_cluster_dict = load_embeddings_cluster()
            embeddings = embeddings_cluster_dict.get(category, [])
            embeddings = torch.tensor(embeddings, dtype=torch.float).to(self.device)
            print(f"Embeddings is{embeddings.shape} None for category {category}")
            similarities = util.pytorch_cos_sim(input_embedding, embeddings)
            sum_similarity_score = torch.sum(similarities)
            mean_similarity_score = sum_similarity_score / len(embeddings)

            similarity_scores[category] = mean_similarity_score.item()

        return similarity_scores

def process_text_with_category(self, splitter_path, ners):
        splitter_keys = self.get_splitter(lang=self.language, splitter_path=splitter_path)
        split_phrases = self.phrase_splitter(phrase_splitter_key=splitter_keys)
        results = {}
        #embeddings_dict = self.load_embeddings(embeddings_file_path)
        #print("Split phrases are ->",split_phrases,"\n")


        for text in split_phrases:
            similarity_scores = calculate_similarity_score_for_category(text=text, ners=ners)
            results[text] = similarity_scores

        result = {sentence: max(scores, key=scores.get) for sentence, scores in results.items()}
        return result

def trim_str(some_list):
    if len(str(some_list)) > 499:
        some_list.pop()
        if some_list[0] != 'trimmed':
            return trim_str(['trimmed']+some_list)
        else:
            return trim_str(some_list)
    else :
        return str(some_list)

def get_ngrams(text, n=4, step=1):
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

def data_to_insert_in_key_table(call_id, category, sub_category, list_of_list):
    return [(call_id, category, sub_category, list_item[0], list_item[1]) for list_item in list_of_list]

def get_primary_intent(outer_dict):
    max_key = None
    max_score = 0 # float('-inf')  # Initialize with negative infinity

    for key, value in outer_dict.items():
        if 'score' in value:
            score = value['score']
            if score > max_score:
                max_score = score
                max_key = key
    return max_key
