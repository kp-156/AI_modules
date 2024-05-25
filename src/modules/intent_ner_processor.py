from src.modules.intent import IntentDetector
from src.modules.ner import NER
from src.utils.pd_utility import get_ner_ngram_from_df
from src.utils.all_utils.common_util import IntentNerClassFunctions


class IntentNERProcessor:
    def __init__(self, text, language, client_id, threshold=0.5):
        self.client_id = client_id
        self.language = language
        self.text = text
        self.threshold = threshold
        self.intent_detector = IntentDetector(language=language, client_id=client_id)
        self.ner_object = NER(text=text, language=language, client_id=client_id)
        self.util_obj = IntentNerClassFunctions(language=self.language, client_id=self.client_id)


    def process_intent(self):
        """
        This function processes the user text and finds the intent i.e., goal
        and whether the intent is ambiguous or multiple
        :return:
        """
        result = {}
        # Intent Detection
        original_top_candidates_with_intent = self.intent_detector.get_top_sentence_candidates(input_sentence=self.text)
        print("Top 5 similar sentence candidates and intent are:", original_top_candidates_with_intent, "\n")
        top_candidates_sentences = [
            sentence for sentence, value in original_top_candidates_with_intent.items() if value['score'] > self.threshold ]
        print("Top Sentences after filtering on threshold are:", top_candidates_sentences, "\n")

        if not top_candidates_sentences:
            print("No similar sentence found above threshold. Returning None")
            result['sentences'] = None
            result['type'] = None
            result['intents'] = None
        elif len(top_candidates_sentences) == 1:
            print("Single similar sentence found above threshold. It's neither ambiguous not multi-intent")
            top_candidates_intents = [
                value['intent'] for sentence, value in original_top_candidates_with_intent.items()
                if sentence in top_candidates_sentences
            ]
            result['sentences'] = top_candidates_sentences
            result['type'] = None
            result['intents'] = top_candidates_intents
        else:
            top_candidates_keywords = self.intent_detector.extract_keywords_for_sentences(top_candidates_sentences)
            input_sentence_keywords = self.util_obj.get_keywords(self.text, self.language)
            result_intent = self.intent_detector.match_input_keywords_with_sentences(
                input_sentence_keywords=input_sentence_keywords,
                top_candidate_keywords_dict=top_candidates_keywords
            )
            print("Multi Intent result is -->", result_intent)
            max_value = max(result_intent.values())
            max_keys = [key for key, value in result_intent.items() if value == max_value]
            max_key_intents = [
                value['intent'] for sentence, value in original_top_candidates_with_intent.items()
                if sentence in max_keys
            ]
            print("Intent found is -->", max_keys)
            if len(max_keys) > 1:
                result['sentences'] = max_keys
                result['type'] = "multi"
                result['intents'] = max_key_intents
            else:
                result['sentences'] = max_keys
                result['type'] = None
                result['intents'] = max_key_intents
        return result




    def process_ner(self, ner_ngram):
        """
        This function carries out the following steps on the input text and ner_ngrams
        1.
        """
        # NER Processing
        print("ner_ngram: ", ner_ngram)
        if isinstance(ner_ngram, str):
            ner_ngram = eval(ner_ngram)

        ner_ngram_keys = list(ner_ngram.keys())
        #print("ner_ngram_keys:", ner_ngram_keys)
        split_phrases = self.ner_object.phrase_splitter(phrase_splitter_key=self.ner_object.splitter_keys)
        split_phrases = list(set(self.ner_object.remove_candidates_with_only_stopwords(split_phrases)))
        partial_res = self.ner_object.process_text_with_category(split_phrases=split_phrases,ner_ngram_keys=ner_ngram_keys)

        
        
        updated_partial_res = {}
        for entity, category in partial_res.items():
            # print("entity------------------------->", entity)
            entity_without_stopwords = self.ner_object.remove_stopwords_from_candidates(entity)
            # print("entity_without_stopwords------------------------->", entity_without_stopwords)
            if len(entity_without_stopwords)>=1:
                word_list = entity_without_stopwords[0].split()
                word_list = self.ner_object.remove_trigger_words(split_phrases = word_list, category=category)
                if len(word_list)==0:
                    continue
                entity_without_triggerwords = " ".join(word_list)
                entity_without_punctuations = self.ner_object.remove_punctuation_from_terminals(entity_without_triggerwords)
                # print("entity_without_punctuations: ", entity_without_punctuations)
                updated_partial_res[entity_without_punctuations] = category

        #     print("updated_partial_res = ", updated_partial_res)

        partial_res = updated_partial_res
        # print("partial_res:", partial_res)

        result = []
        for phrase, ner in partial_res.items():
            if phrase:
                
                res_item, score = self.ner_object.process_text_ner_2(text=phrase, category=ner, ngram_length=ner_ngram[ner]) 
                if res_item is not None and score is not None:
                    result.append((list(set(res_item)), list(set(score)), ner))
                else: 
                    continue

        print("result from process_text_ner_2:=================>", result)
        data_res = self.ner_object.get_max_scores(result)
        print("data_res:=================>", data_res)

        # self.ner_object.process_text_for_split_words(ner_ngram_keys=ner_ngram_keys, text = self.text)
        top_values = {}
        for tag, values in data_res.items():
            if isinstance(values[0], list) and len(values[0]) > 0:
                top_values[tag] = [values[0][0]]
            else:
                top_values[tag] = []

        print("final result:=================>", top_values)
        return top_values
        
        

    def get_ner_ngram(self, IntentSentence):
        """
        This function gets the NER ngram for the input text from the CSV DF. If not found, it returns an empty dict.
        """
        result = get_ner_ngram_from_df(self.language, self.client_id, IntentSentence)
        return result

    def update_intent_data(self, sentence, goal_name):
        """
        This function reads the local client dataframe, embeddings and keyword embeddings
        and updates them with new intent result
        """
        try:
            #self.intent_detector.update_dataframe(sentence, goal_name)
            self.intent_detector.update_embeddings()
            self.intent_detector.update_keyword_embeddings()
            print("Successfully updated the local client dataframe, embeddings and keyword embeddings with new data.")
        except Exception as e:
            print("Error in updating the local data with intent results:\n", str(e))

    def update_ner_cluster_data(self, top_values):
        """
        This function reads the local NER cluster JSON and embeddings and updates them with new NER result
        """
        try:
            self.ner_object.update_ner_cluster_json(top_values)
            #self.ner_object.update_ner_cluster_embeddings()
            print("Successfully updated the NER cluster json and embeddings with new data.")
        except Exception as e:
            print("Error in updating the NER cluster json and embeddings with NER results:\n", str(e))

    def update_ner_sample_data(self, top_values):
        """
        This function reads the local NER cluster JSON and embeddings and updates them with new NER result
        """
        try:
            self.ner_object.update_ner_sample_json(top_values)
            #self.ner_object.update_ner_sample_embeddings()
            print("Successfully updated the NER sample json and embeddings with new data.")
        except Exception as e:
            print("Error in updating the NER sample json and embeddings with NER results:\n", str(e))
