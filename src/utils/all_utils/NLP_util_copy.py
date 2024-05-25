import csv
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import regex as re
import gensim
import csv
import json
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.ldamodel import LdaModel


class NLPUtilities:
    def __init__(self):
        self.stopwords = Stopwords()
        self.regex = Regex()
        self.pos = POS()
        self.stemming = Stemming()
        self.tokenization = Tokenization()
        self.tfidf = TFIDF()

    def remove_stopwords(self, words):
        return self.stopwords.remove_stopwords(words)

    def search_regex(self, pattern, text):
        return self.regex.search(pattern, text)

    def findall_regex(self, pattern, text):
        return self.regex.findall(pattern, text)

    def pos_tag(self, words):
        return self.pos.pos_tag(words)

    def get_tag(self, tag):
        return self.pos.get_tag(tag)

    def stem(self, words):
        return self.stemming.stem(words)

    def tokenize(self, text):
        return self.tokenization.tokenize(text)

    def get_tfidf(self, corpus, dictionary):
        self.tfidf = TFIDF(corpus, dictionary)
        return self.tfidf.get_tfidf()

class CSVToJson:
    def __init__(self, file_input_name, file_output_name):
        self.file_input_name = file_input_name
        self.file_output_name = file_output_name

    def convert_csv_to_json(self):
        with open(self.file_input_name, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            json_data = [row for row in csv_reader]

        with open(self.file_output_name, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

    def convert_json_to_csv(self):
        with open(self.file_input_name, 'r') as json_file:
            json_data = json.load(json_file)

        with open(self.file_output_name, 'w', newline='') as csv_file:
            fieldnames = json_data[0].keys()
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(json_data)

class Stopwords:
    def __init__(self, language='english'):
        self.language = language
        self.stop_words = set(stopwords.words(self.language))

    def remove_stopwords(self, words):
        return [word for word in words if word not in self.stop_words]

class Regex:
    def __init__(self, pattern):
        self.pattern = re.compile(pattern)

    def search(self, text):
        return self.pattern.search(text)

    def findall(self, text):
        return self.pattern.findall(text)

class POS:
    def __init__(self):
        self.tag_map = {'N': 'noun', 'V': 'verb', 'R': 'adverb', 'J': 'adjective'}

    def pos_tag(self, words):
        return nltk.pos_tag(words)

    def get_tag(self, tag):
        return self.tag_map.get(tag[0])

class Stemming:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def stem(self, words):
        return [self.stemmer.stem(word) for word in words]

class Tokenization:
    def __init__(self, language='english'):
        self.language = language

    def tokenize(self, text):
        return word_tokenize(text, self.language)

class TFIDF:
    def __init__(self, corpus, dictionary):
        self.corpus = corpus
        self.dictionary = dictionary

    def get_tfidf(self):
        tfidf = gensim.models.TfidfModel(self.corpus)
        return tfidf[self.corpus]


class TopicModeling:
    def __init__(self, corpus, num_topics):
        self.corpus = corpus
        self.num_topics = num_topics

        self.dictionary = Dictionary(corpus)
        
        self.corpus_bow = [self.dictionary.doc2bow(text) for text in corpus]

        self.lda_model = LdaModel(self.corpus_bow, num_topics=self.num_topics, id2word=self.dictionary, passes=15)

    def get_top_words_per_topic(self):
        top_words_per_topic = []
        for topic_id in range(self.num_topics):
            top_words = self.lda_model.show_topic(topic_id, topn=10)
            top_words_per_topic.append([word[0] for word in top_words])
        return top_words_per_topic

    def get_document_topics(self):
        document_topics = []
        for doc in self.corpus_bow:
            doc_topics = self.lda_model.get_document_topics(doc)
            document_topics.append(doc_topics)
        return document_topics





def main():
    nlp_utilities = NLPUtilities()

    # Remove stopwords
    words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
    print('Original words:', words)
    print('Words without stopwords:', nlp_utilities.remove_stopwords(words))


if __name__=='__main':
    main()