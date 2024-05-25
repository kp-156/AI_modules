import os
import json
import numpy as np
import ast
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from src.modules.intent_ner_processor import IntentNERProcessor
from src.config import chatbot_data_path, chatbot_data_embeddings_path

chatbot_data_path = "/home/pritika/travel-chatbot-backup/src/new2way_bot/data/travel_usecase_data.json"
chatbot_data_embeddings_path = "/home/pritika/travel-chatbot-backup/src/new2way_bot/data/travel_usecase_embedded.json"

class Retriever:
    def __init__(self):
        self.data_path = chatbot_data_path
        self.embedded_data_path = chatbot_data_embeddings_path
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Load pre-embedded data or create new embeddings
        self.embedded_data = self.embed_json_data(self.data_path, self.embedded_data_path)

        # Load existing FAISS index or create a new one
        self.faiss_index = self.create_faiss_index(self.embedded_data)

    def find_intent_and_ner(self, input_sentence):
        language = "en"
        client_id = "1"
        intent_ner_processor = IntentNERProcessor(text=input_sentence, language=language, client_id=client_id)
        result = intent_ner_processor.process_intent()
        print(result)
        unique_intent =  "None" if result["intents"] is None else result["intents"][0]

        print("intent=======>", unique_intent)
        if unique_intent != "None":
            ner_ngram_str = intent_ner_processor.get_ner_ngram(IntentSentence=result["sentences"][0])
            ner_ngram = ast.literal_eval(ner_ngram_str)
            ner_result = intent_ner_processor.process_ner(ner_ngram)
            final_ner = {key: value[0] for key, value in ner_result.items()}
            print("ner_result=======>", final_ner)
            return unique_intent, final_ner
        else:
            return "None", None

    def embed_json_data(self, json_path, save_path):
        """
        Loads JSON data, performs sentence-level embedding, and saves the results.

        Args:
          json_path: Path to the JSON file containing intents and responses.
          save_path: Path to save the embedded data.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        embedded_data = []
        for entry in data['data']:
            intent = entry['intent']
            responses = entry['responses']

            # Embed each sentence (intent and responses)
            intent_embedding = self.embed_model.encode(intent)
            response_embeddings = [self.embed_model.encode(sentence) for sentence in responses]

            # Create a dictionary for the embedded data
            embedded_entry = {
                "intent": intent,
                "intent_embedding": intent_embedding.tolist(),
                "responses": responses,
                "response_embeddings": [embedding.tolist() for embedding in response_embeddings]
            }

            embedded_data.append(embedded_entry)

        # Save the embedded data
        with open(save_path, 'w') as f:
            json.dump(embedded_data, f)

        print(f"Embedded data saved to: {save_path}")

        return embedded_data

    def create_faiss_index(self, embedded_data):
        embeddings = [np.array(entry["intent_embedding"]) for entry in embedded_data]
        faiss_index = faiss.IndexFlatL2(embeddings[0].shape[0])
        faiss_index.add(np.array(embeddings))
        return faiss_index

    def calculate_similarity(self, embedding1, embedding2):
        """
        Calculates the cosine similarity between two embeddings.

        Args:
          embedding1: First embedding vector.
          embedding2: Second embedding vector.

        Returns:
          similarity_score: Cosine similarity score between the two embeddings.
        """
        similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return similarity_score

    def find_response(self, user_query_embedding, response_sentences):
        """
        Finds the most relevant response to the user query among the response sentences.

        Args:
          user_query_embedding: Embedding of the user query.
          response_sentences: List of response sentences.

        Returns:
          most_relevant_response: The most relevant response sentence.
        """
        most_similar_index = 0
        highest_similarity = -1  # Initialize with lowest possible value

        # Iterate over response sentences to find the most similar one
        for i, sentence in enumerate(response_sentences):
            sentence_embedding = np.array(self.embed_model.encode(sentence))
            similarity_score = self.calculate_similarity(user_query_embedding, sentence_embedding)
            if similarity_score > highest_similarity:
                highest_similarity = similarity_score
                most_similar_index = i

        return response_sentences[most_similar_index]

    def find_matching_intent_response(self, user_query):
        """
        Finds the response line based on the user intent, NER dictionary, and user query.

        Args:
          user_intent: The user's intent.
          user_query: The user's query.
          ner_dict: The NER dictionary extracted from the user query.

        Returns:
          matching_line: The matching response line.
        """
        matching_line = None

        user_intent, ner_dict = self.find_intent_and_ner(user_query)

        if user_intent == "None":
            print("No intent matching")
        else:
            for entry in self.embedded_data:
                if entry["intent"] == user_intent:
                    matching_intent_entry = entry
                    break
            else:
                print("Intent not found in data. Please ask related to Travel")
                return None  # Return None on intent not found

            filtered_sentences = matching_intent_entry["responses"]
            for entity, value_list in ner_dict.items():
                filtered_sentences = [sentence for sentence in filtered_sentences if any(val in sentence for val in value_list)]

            # Embed user query
            user_query_embedding = np.array(self.embed_model.encode(user_query))

            # Find the most relevant line from the filtered sentences
            matching_line = self.find_response(user_query_embedding, filtered_sentences)

        return matching_line



# #  Example usage


# # Initialize the retriever
# retriever = Retriever()

# # Sample user query
# user_query = "is there any Chinese cuisine, for 4 stars?"

# # Find matching intent response
# matching_line = retriever.find_matching_intent_response(user_query)
# print("Matching line:", matching_line)
