import os
import json
from sentence_transformers import util
from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer
import re
import pickle
import torch
import numpy as np
import requests
from fuzzywuzzy import process
import time
from src.utils.all_utils.common_util import LoadAndSaveFiles, IntentNerClassFunctions

# import nltk
# nltk.download('wordnet')

from src.config import (
    ner_embeddings_samples,
    ner_embeddings_cluster,
    ner_json_samples,
    ner_json_cluster,
    stop_words_file_path,
    splitter_path,
    trigger_word_path,
    embeddings_api_url,
)

class NER:
    """
    This call provides functions to get NER from text
    """
    def __init__(self, text, language, client_id):
        self.client_id = client_id
        self.language = language
        self.text = text
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda", quantize=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.stopwords_list = self.load_stopwords(language=self.language)
        self.punctuation_list = self.load_stopwords(language="punctuation")
        self.splitter_keys = self.load_splitter_file()
        self.trigger_words_list = self.load_trigger_words()

        self.ner_embeddings_samples_path = ner_embeddings_samples.format(client_id=client_id, language=language)
        self.ner_embeddings_cluster_path = ner_embeddings_cluster.format(client_id=client_id, language=language)
        self.embeddings_samples_dict = self.load_ner_embeddings_samples()
        self.embeddings_cluster_dict = self.load_ner_embeddings_cluster()

        self.ner_samples_json_path = ner_json_samples.format(client_id=client_id, language=language)
        self.ner_cluster_json_path = ner_json_cluster.format(client_id=client_id, language=language)

        self.util_obj = IntentNerClassFunctions(language=self.language, client_id=self.client_id)


    def load_splitter_file(self):
        try:
            splitter_json= LoadAndSaveFiles.load_json(file_path=splitter_path)
            return splitter_json[self.language]
        except FileNotFoundError:
            print(f"Error: Split-words file is not specified for language {self.language}. Please provide the file")
            return None

    def load_stopwords(self, language = None):
        try:
            stopwords_dict = LoadAndSaveFiles.load_json(file_path=stop_words_file_path)  
            return stopwords_dict[language]
        except FileNotFoundError:
            print(f"Error: stopwords file is not specified for language {language}. Please provide the file")
            return None
    
    def load_trigger_words(self, language = None):
        try:
            return LoadAndSaveFiles.load_json(file_path=trigger_word_path)
        except FileNotFoundError:
            print(f"Error: trigger word file is not specified for language {language}. Please provide the file")
            return None

    def load_ner_embeddings_samples(self):
        """
        This function loads the NER samples embeddings from local file based on client_id and language.
        Returns empty dictionary if the file is not found.
        :return:
        """
        try:
            return LoadAndSaveFiles.load_pickle(self.ner_embeddings_samples_path)
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
            return LoadAndSaveFiles.load_pickle(self.ner_embeddings_cluster_path)
        except FileNotFoundError:
            print(f"NER cluster embedding file for {self.client_id}-{self.language} not found. Please provide the file")
            return None   

    def load_ner_sample_or_cluster_json(self, file_path):
        """
        This function loads the latest NER samples JSON data from file. Returns empty dict if file doesn't exist.
        """
        if os.path.exists(file_path):
            return LoadAndSaveFiles.load_json(file_path=file_path)
        else:
            return None

    def phrase_splitter(self, phrase_splitter_key):
        """
        Split the input text based on the phrase_splitter_keys and returns the list of texts after splitting
        """
        for key in phrase_splitter_key:
            self.text = self.text.replace(" " + key.strip() + " ", " "  + "<split>" + key.strip() +" ")
            
        splitted_phrases = self.text.split("<split>")
        text = [s.strip() for s in splitted_phrases]
        return text

    
    def generate_ngrams(self, sentence, n):
        """
        Generate ngrams from the given sentence based on the specified size of ngram 'n',
        e.g., sentence "I want to visit Japan" and n = 2 would give us: ["I want", "want to", "to visit", "visit Japan"]
        """
        words = sentence.split()
        # print("words for generating ngrams--------->", words)
        ngrams = []
        if n <= 0:
            return []
        if n > len(words):
            return [" ".join(words)]
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i: i + n])
            ngrams.append(ngram)
        return ngrams

    def calculate_similarity_score_for_category(self, text, ner_ngram_keys, categories_dict):
        """
        Calculate the similarity score for the given text and NER ngram key (i.e., NER categories)
        Embeddings for the input text are obtained from the API.
        Embeddings for the NER ngrams are obtained from the pre-trained embeddings file.
        """
        
        similarity_scores = {}
        
        input_embeddings = self.util_obj.get_input_embedding_from_api([text])
        if input_embeddings is None:
            print("Failed to fetch input embeddings.")
            return similarity_scores


        # print(f"Input embeddings: {input_embeddings.shape}")
        for category in ner_ngram_keys:
            embeddings = categories_dict.get(category, [])
            # print(f"Processing category: {category}")
            if not embeddings:
                print(f"Embeddings for NER category {category} are empty or not found.")
                continue

            # print(f"Category embeddings: {len(embeddings)}")
            # Create a tensor from the embeddings and find cosine similarity with input text embeddings
            embeddings = torch.tensor(embeddings, dtype=torch.float).to(self.device)
            similarities = util.pytorch_cos_sim(input_embeddings, embeddings)
            sum_similarity_score = torch.sum(similarities)
            mean_similarity_score = sum_similarity_score / len(embeddings)

            # Store the similarity score for each NER category i.e., NER ngram key
            similarity_scores[category] = mean_similarity_score.item()
            #print(similarity_scores)

        return similarity_scores

    def process_text_with_category(self, split_phrases, ner_ngram_keys):
        """
        This functions aims to match the phrases of the input sentence with most relevant NER ngram category.
            1. Splits the input text based on split keys into a list of phrases
            2. Find similarity score for each phrase and adds them to a dictionary
            3. Select the most similar key for each phrase based on similarity score and return it
        """        
        results = {}

        for phrase_text in split_phrases:  
            similarity_scores = self.calculate_similarity_score_for_category( text=phrase_text, ner_ngram_keys=ner_ngram_keys, categories_dict=self.embeddings_cluster_dict)
            # print(f"similarity_scores for phrase '{phrase_text}': {similarity_scores}")
            results[phrase_text] = similarity_scores

        final_result = {
            phrase_text: max(scores, key=scores.get) for phrase_text, scores in results.items()
        }
        print(f"Final selected similar key for each phrase: {final_result}")
        return final_result
    
    def process_text_for_split_words(self, ner_ngram_keys, text):
        s = time.time()
        input = text
        split_words = input.split()
        split_words = list(set(self.remove_candidates_with_only_stopwords(split_words)))
        temp = []
        for word in split_words:
            temp.append(self.remove_punctuation_from_terminals(word))
        split_words = list(set(temp))
        results = {}
        for phrase_text in split_words:  
            similarity_scores = self.calculate_similarity_score_for_category( 
                text=phrase_text, 
                ner_ngram_keys=ner_ngram_keys, 
                categories_dict=self.embeddings_samples_dict )
            # print(f"similarity_scores for phrase '{phrase_text}': {similarity_scores}")
            results[phrase_text] = similarity_scores
        
        _result = {
            phrase_text: [max(scores, key=scores.get), scores[max(scores, key=scores.get)]]
            for phrase_text, scores in results.items()
        }
        final_result = {}
        for phrase_text, val in _result.items():
            if val[0] in final_result:
                final_result[val[0]].append([phrase_text, val[1]])
            else:
                final_result[val[0]] = [[phrase_text, val[1]]]

        print(f"\n\n*******************************************************")
        print(final_result)
        print(f"*******************************************************")

        print("--- %s seconds ---" % (time.time() - s))
        return final_result

    def remove_stopwords_from_candidates(self, candidates_string):
        """
        This function takes a string of candidate phrases separated by spaces
        and removes stopwords from each individual phrase
        """
        filtered_candidates = []
        stopwords = self.stopwords_list
        # triggers = self.trigger_words_list
        #print("candidates_string = ", candidates_string.lower())
        candidates = [candidates_string.lower().split()]
        #print("Candidates====>", candidates)
        for candidate in candidates:
            filtered_words = [word for word in candidate if word not in stopwords]
            filtered_candidate = " ".join(filtered_words)
            if filtered_candidate:
                filtered_candidates.append(filtered_candidate)
        #print("filtered_candidates=========>", filtered_candidates)
        return filtered_candidates

    def remove_candidates_with_only_stopwords(self, candidates):
        """
        This function takes a list of candidate phrases and remove the phrase if it consists of stop-words only
        """
        # print("candidates====>", candidates)
        filtered_candidates = []
        stopwords = self.stopwords_list

        for candidate in candidates:
            words = candidate.split()
            if not all(word.lower() in stopwords for word in words):
                filtered_candidates.append(candidate)
        return filtered_candidates
    
    def remove_punctuation_from_terminals(self, text):
        for punctuation in self.punctuation_list:
            pattern = re.compile(r'(^{}|{}$)'.format(re.escape(punctuation), re.escape(punctuation)))
            text = pattern.sub('', text)
            # print("text after removing punctuations: ", text)
        return text

    def remove_trigger_words(self, split_phrases, category):
        """
        Remove trigger words from split phrases using fuzzy matching.

        Args:
            split_phrases (list): List of phrases to filter.
            category (str): Category of trigger words.
            threshold (int): Threshold for fuzzy matching score (default is 80).

        Returns:
            list: Filtered candidates.
        """
        filtered_candidates = []
        threshold=80
        triggerwords = [word.lower() for word in self.trigger_words_list[category]]
        # print("triggerwords", triggerwords)
        for candidate in split_phrases:
            filtered_words = []
            for word in candidate.split():
                # Use fuzzy matching to find the closest trigger word
                match, score = process.extractOne(word.lower(), triggerwords)
                if score < threshold:
                    filtered_words.append(word)
            filtered_words = " ".join(filtered_words)
            if filtered_words:
                filtered_candidates.append(filtered_words)
        # print("filtered_candidates===>", filtered_candidates)
        return filtered_candidates 
    

    def remove_splitter_words(self, split_phrases):
        filtered_candidates = []
        splitterwords = self.splitter_keys
        for candidate in split_phrases:
            filtered_words = []
            for word in candidate.split():
                if word.lower() not in splitterwords:
                      filtered_words.append(word) 
            filtered_words = " ".join(filtered_words)
            if filtered_words:
                filtered_candidates.append(filtered_words)
        #print("filtered_candidates===>", filtered_candidates)
        return filtered_candidates      
    
    

    
    def get_max_scores(self, data):
        """
        Gets the scores for a category greater than threshold from the data. 
        And if none of the scores are greater than threshold, it gets the highest score.

        """

        threshold = 0.7
        selected_scores = {}

        for text_list, scores, category in data:
            selected_values = []
            selected_scores_list = []
            max_score_index = None
            for i, score in enumerate(scores):
                if score > threshold:
                    selected_values.append(text_list[i])
                    selected_scores_list.append(score)
                elif max_score_index is None or score > scores[max_score_index]:
                    max_score_index = i

            if selected_values:
                selected_scores[category] = [selected_values, selected_scores_list]
            elif max_score_index is not None:
                selected_scores[category] = [[text_list[max_score_index]], [scores[max_score_index]]]
        return selected_scores  


    def process_text_ner_2(self, text, category, ngram_length=(0, 1)):
        """
        Process the text with the given category and ngram_length
        """
        all_text = []
        for i in range(ngram_length[0]-1, ngram_length[1]):
            all_text.extend(self.util_obj.generate_ngrams(text, i + 1))

        all_text.append(text)
        all_text = list(set(all_text))
        if len(all_text)>=1:
            user_embeddings = self.util_obj.get_input_embedding_from_api(list(all_text))
            samples_embeddings = torch.tensor(self.embeddings_samples_dict[category], dtype=torch.float).to(self.device)
            cosine_similarities = util.pytorch_cos_sim(user_embeddings, samples_embeddings)

            # top 3 matches
            best_match_indices = np.argsort(-np.max(cosine_similarities.cpu().numpy(), axis=1))[:3]
            best_match_scores = np.max(cosine_similarities.cpu().numpy(), axis=1)[best_match_indices]
            best_match_sentences = [all_text[i] for i in best_match_indices]
            #print("best_match_sentences======>", best_match_sentences)
            return best_match_sentences, best_match_scores

    def update_ner_cluster_json(self, top_values):
        """
        This function reads the latest NER cluster JSON data, appends the ner_gram and its values to the existing keys
        or adds new keys and saves the updated data to the original JSON file
        """
        for category, value in top_values.items():
            data = self.load_ner_sample_or_cluster_json(self.ner_cluster_json_path) #load ner cluster json
            if data is None:
                print("Could not load existing NER cluster JSON. Returning empty dictionary")
                data = {}

            if category not in data:
                data[category] = []  

            # Append the values for ner_key if it already exists otherwise add a new key to the data
            data[category].append(value)
            
            data_converted = {str(key): value for key, value in data.items()}  # Converted data back to list? TODO: Check

            with open(self.ner_cluster_json_path, "w") as file:
                json.dump(data_converted, file, indent=2)
                print(f"NER cluster JSON file updated for ner_ngram: {category}")

        print(f"NER cluster JSON file updated for {self.language}-{self.client_id}")

    def update_ner_cluster_embeddings(self):
        """
        This function reads the latest NER cluster JSON data, finds their embeddings
         and saves the updated embeddings to the cluster embeddings file
        """
        data = self.load_ner_sample_or_cluster_json(self.ner_cluster_json_path) #load ner cluster json
        if data is None:
            print("Could not load existing NER cluster JSON. Returning empty dictionary")
            data = {}
        embeddings_dict = {}
        
        for category, sentences in data.items():
            query_embeddings = self.model.encode(sentences, convert_to_tensor=True)
            embeddings_dict[category] = query_embeddings.tolist()

        with open(self.ner_embeddings_cluster_path, "w") as f:
            json.dump(embeddings_dict, f)
        print(f"NER cluster embeddings file {self.ner_embeddings_cluster_path} is updated")
        

    def update_ner_sample_json(self, top_values):
        """
        This function reads the latest NER samples JSON data, appends the ner_gram and its values to the existing keys
        or adds new keys and saves the updated data to the original JSON file
        """
        for category_name, value in top_values.items():
            data = self.load_ner_sample_or_cluster_json(self.ner_samples_json_path) #load ner sample json 
            if data is None:
                print("Could not load existing NER samples JSON. Returning empty dictionary")
                data = {}
            
            if category_name not in data:
                data[category_name] = {"ngram": [], "values": []}

            data[category_name]["values"].append(value)
            # Append the values for ner_ngram if it already exists otherwise add a new key to the data
            with open(self.ner_samples_json_path, "w") as file:
                json.dump(data, file, indent=2)
                print(f"NER samples JSON file updated for ner_ngram: {category_name}")

        print(f"NER samples JSON file updated for {self.language}-{self.client_id}")

        # TODO: Check if the below is needed
        # update_ner_ngram_for_key(client_id,language,ner_ngram,values)

    def update_ner_sample_embeddings(self):
        """
        This function reads the latest NER samples JSON data, finds the embeddings of each token
         and saves the updated embeddings to the samples embeddings file
        """
        data = self.load_ner_sample_or_cluster_json(self.ner_samples_json_path) #load ner sample json
        if data is None:
            print("Could not load existing NER samples JSON. Returning empty dictionary")
            data = {}
        encoded_data = {}
  
        for name, tokens in data.items():
            if "values" in tokens:
                values = tokens["values"]
                print(f"Encoding tokens for '{name}': {values}")
                encoded_tokens = self.model.encode(values, convert_to_tensor=True)
                encoded_data[name] = encoded_tokens.tolist()

        with open(self.ner_embeddings_samples_path, "wb") as pickle_file:
            pickle.dump(encoded_data, pickle_file)
        #print(f"NER samples embeddings file {self.ner_embeddings_samples_path} is updated")

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
