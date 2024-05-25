from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer
from sentence_transformers import util
import time
import pickle
import sys
from src.modules.ner import NER
import json
from src.modules.packtypes import TranscriptionPack, Pack
import torch
import os

from src.utils.all_utils.common_util import LoadAndSaveFiles
from src.config import (
    tcs_parameter_embeddings,
    tcs_parameter_json,
    tcs_phrase_importance,
    tcs_phrase_importance_embeddings,
    tcs_scores,
    FastSentenceTransformer
    
)


class TranscriptionScorer():
    model = None
    def __init__(self, text, language, client_id, scores_id) -> None:
        self.language = language
        self.client_id = client_id
        self.model = SentenceTransformer(FastSentenceTransformer, device = 'cuda')
        
        self.parameter_embeddings = tcs_parameter_embeddings.format(client_id=client_id, language=language)
        self.parameter_json = tcs_parameter_json.format(client_id=client_id, language=language)
        self.phrase_importance_json = tcs_phrase_importance.format(client_id=client_id, language=language)
        self.phrase_importance_embeddings = tcs_phrase_importance_embeddings.format(client_id=client_id, language=language)
        self.tcs_scores_json_file = tcs_scores.format(scores_id=scores_id, language=language)
        
        self.ner_object = NER(text=text, language=language, client_id=client_id)
        self.scores = LoadAndSaveFiles.load_json(file_path=self.tcs_scores_json_file)

        

    def get_keyword_type(self, ngram_embedding):
        # with open(self.phrase_importance, "rb") as f:
        #     encoded_data = pickle.load(f)
        encoded_data = LoadAndSaveFiles(self.phrase_importance_json)
        for category, embeddings in encoded_data.items():
            more_important_embeddings = embeddings['more_important_embeddings']
            less_important_embeddings = embeddings['less_important_embeddings']

        for idx, embed in enumerate(more_important_embeddings):
            embed = embed.to('cuda:0')
            similarity = util.pytorch_cos_sim(embed, ngram_embedding.to('cuda:0'))[0][0]
            if similarity.item() > 0.6:
                return 'more_important'

        for embed in less_important_embeddings:
            embed = embed.to('cuda:0')
            similarity = util.pytorch_cos_sim(embed, ngram_embedding.to('cuda:0'))[0][0]
            if similarity.item() > 0.6:
                return 'less_important'
    
        return 'more_important'


    def get_transcription_score(self, doc:TranscriptionPack = None, transcript = None):
    # def get_transcription_score(self, doc:Pack = None, transcript = None):
    #     if 'transcription scorer' in doc.cat:
    #         list_ = pickle.loads(doc.lst)
    #     for dict_ in list_:
    #         dict_['speker1_full_text']
        if doc:
            transcript = doc.speaker1_full_text
        transcript = transcript.lower()

        # score = {
        #     'standard_opening' : [0,5],
        #     'purpose_of_the_call' : [0,5],
        #     'objection_handling' : [0,15],
        #     'probing_skills' : [0,10],
        #     'persuation' : [0,10],
        #     'listening_skill': [0,10],
        #     'rate_of_speech' : [0,10],
        #     'empathetic_phrases' : [0,10],
        #     'confident_and_fumbling_enthusiasm' : [0,10],
        #     'standardized_closing' : [0,5],
        #     'product_knowledge_and_information' : [0,10],
        #     'legal_flag' : [0,5],
        #     'abusive_flag' : [0,5]
        # }

        t_score = dict({key: value for key, value in self.scores.items()})
        total = 0
        
        EDICT = LoadAndSaveFiles.load_pickle(self.parameter_embeddings)
        ngrams = self.ner_object.generate_ngrams(n = 4, sentence = transcript)

        # print(ngrams)

        phrases = ngrams
        embedding = []
        for phrase in phrases:
            embedding.append(self.model.encode(phrase, convert_to_tensor=True).to('cuda:0'))

        ngram_dict = dict(zip(phrases, embedding))

        s = time.time()
        ignore_in_general = ['total', 'rate_of_speech']
        for key in t_score:
            print('scoring for key : ', key)
            if key not in ignore_in_general:
                for ngram in ngram_dict:
                    ngram_embedding = ngram_dict[ngram]
                    for keyword, keyword_embedding in EDICT[key]:
                        ngram_embedding = ngram_embedding.to(keyword_embedding.device)
                        similarity = util.pytorch_cos_sim(keyword_embedding, ngram_embedding)[0][0]
                        # print(similarity.item(),ngram, keyword)
                        if similarity.item() > 0.4:
                            print(key, " :  ", ngram, "     score:", similarity.item())
                            keyword_type = self.get_keyword_type(ngram_embedding=ngram_embedding)
                            print(keyword, ": ", keyword_type)
                            weight = 1.0  
                            if keyword_type == "more_important":
                                weight = 1.5  
                            elif keyword_type == "less_important":
                                weight = 0.5 

                            score_increment = 3 
                            if similarity.item() > 0.6:
                                score_increment = 5  

                            weighted_score_increment = score_increment * weight
                            print("weighted_score_increment========>", weighted_score_increment)

                            t_score[key][0] += weighted_score_increment
                            t_score[key].append([keyword, ngram])
                            if t_score[key][0] > t_score[key][1]:
                                t_score[key][0] = t_score[key][1]
                            break
    
        e = time.time()

        runtime = e - s
        print(f"key phrase runtime: {runtime:.2f} seconds")

        total = 0
        for key in t_score:
            if key != 'total' and key != 'abusive_flag' and key != 'legal_flag':
                total += t_score[key][0]
        t_score['total'] = total
        print("t_score:", t_score)

        # self.print_score_report(t_score=t_score, transcript=transcript)
        if doc:
            doc.transcription_score = t_score
            return doc
        else:
            return t_score
    
        

    def print_score_report(self, t_score, transcript):
        """
        Prints a human-readable report of the transcription score.

        Args:
            t_score: A dictionary containing category scores and details.
            transcript: The full transcript text.
        """
        print("**Transcription Score Report**")
        print(f"Transcript: {transcript}\n")

        print("{:<25s}{:>10s}{:>15s}{:>20s}".format("Category", "Score", "Max Score", "Matched Phrases"))
        print("-" * 80)


        sorted_categories = sorted(t_score.items(), key=lambda x: x[1][0], reverse=True)


        for category, (score, details) in sorted_categories:
            if category == 'total':
                continue  

            max_possible_score = score[1] - score[0]
            matched_phrases_str = ", ".join([f"{p[0]} ({p[2]:.2f})" for p in details])

            print("{:<25s}{:>10d}/{:>10d}{:>15.2f}{:>20s}".format(
                category, score[0], max_possible_score, score[2], matched_phrases_str))


    def merge_keywords_json(self, data):

        merged_data = {}
        for item in data:
            category = item["category"]
            merged_keywords = item["more_important_keywords"] + item["less_important_keywords"]
            merged_data[category] = merged_keywords
        return merged_data

    def create_embeddings_for_parameters(self):
        try:
            merged_data = LoadAndSaveFiles.load_json(self.parameter_json)
        except FileNotFoundError:
            return f"File {self.parameter_json} not found"

        embeddings = {}
        for category, keywords in merged_data.items():
            keyword_embeddings = []
            for keyword in keywords:
                embedding = self.model.encode(keyword, convert_to_tensor=True).to('cuda:0')
                keyword_embeddings.append((keyword, embedding))
            embeddings[category] = keyword_embeddings
        LoadAndSaveFiles.save_pickle(embeddings, self.parameter_embeddings)


    def update_json_with_new_categories(self, new_data):
        try:
            existing_data = LoadAndSaveFiles.load_json(self.parameter_json)
        except FileNotFoundError:
            existing_data = []

        existing_categories = {item['category'] for item in existing_data}
        for new_item in new_data:
            if new_item['category'] not in existing_categories:
                existing_data.append(new_item)

        LoadAndSaveFiles.save_json(existing_data, self.parameter_json)
        # with open(json_path, 'w') as file:
        #     json.dump(existing_data, file, indent=4)


    def encode_phraseimportance_data(self, data, model):
        result = {}
        for item in data:
            category = item['category']
            more_important_keywords = item['more_important_keywords']
            less_important_keywords = item['less_important_keywords']

            more_important_embeddings = self.model.encode(more_important_keywords, convert_to_tensor=True).to('cuda:0')
            less_important_embeddings = self.model.encode(less_important_keywords, convert_to_tensor=True).to('cuda:0')

            result[category] = {
                'more_important_embeddings': more_important_embeddings,
                'less_important_embeddings': less_important_embeddings
            }
        self.save_phraseimportance_embeddings(embeddings_dict=result)
        

    def save_phraseimportance_embeddings(self, embeddings_dict):
        file_path = self.phrase_importance_embeddings
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        tensor_paths_dict = {}

        for category, embeddings in embeddings_dict.items():
            category_dir = os.path.join(os.path.dirname(file_path), category)
            os.makedirs(category_dir, exist_ok=True)
            
            more_important_path = os.path.join(category_dir, 'more_important_embeddings.pt')
            less_important_path = os.path.join(category_dir, 'less_important_embeddings.pt')
            
            torch.save(embeddings['more_important_embeddings'], more_important_path)
            torch.save(embeddings['less_important_embeddings'], less_important_path)
            
            tensor_paths_dict[category] = {
                'more_important_embeddings': more_important_path,
                'less_important_embeddings': less_important_path
            }
        LoadAndSaveFiles.save_pickle(tensor_paths_dict, file_path)


# if __name__ == "__main__":
#     agent = "Hello! I hope you're doing well today. I wanted to discuss an exciting opportunity we have for you. We've reviewed your profile and financial history, and we're pleased to offer you a fantastic loan with very competitive terms. This could help you achieve some of your financial goals. \
#     I understand your hesitation, but I'd like to highlight some of the benefits of this loan offer. The interest rate is quite favorable, and the repayment options are flexible to ensure it fits into your budget. Plus, this could be a great way to consolidate any higher-interest debt you might have. \
#     I completely understand your concerns. It's important to make financial decisions that align with your goals and circumstances. However, let me share some ways this loan could actually help you. By taking advantage of the funds now, you could potentially invest in opportunities that yield higher returns than the loan's interest rate. \
#     I respect your decision, and it's essential to prioritize financial stability. If circumstances change or if you reconsider in the future, please don't hesitate to reach out. Our loan offer will still be available, and I'm here to address any questions you might have. \
#     Absolutely, I'm here to help. Whether you're ready to take advantage of this loan offer or if you simply need more information down the line, feel free to give me a call. Have a wonderful day, and take care! \
#     Goodbye, and have a great day! "
#     t = TranscriptionScorer()

#     doc = t.get_transcription_score(doc=TranscriptionPack(speaker1_full_text=agent))
#     print("doc:", doc)
#     print(json.dumps(doc.transcription_score, indent = 2))