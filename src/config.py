# Client dataframes
client_dataframe_path = "data/client_dataframes/{client_id}_{language}_dataframe"
client_embedding_path = "data/client_embeddings/{client_id}_{language}_embeddings.pkl"
client_keyword_embedding_path = "data/client_keyword_embeddings/{client_id}_{language}_keywords.pkl"

# Stop words file path
stop_words_file_path = "./data/stopwords.json"

# Sentence splitter file path
splitter_path = "./data/splitter.json"

trigger_word_path = "./data/trigger_words.json"

# Pre-trained NER embedding file paths (cluster and sample)
ner_embeddings_cluster = "data/ner_cluster/embeddings/queries_embeddings_{client_id}_{language}.pkl"
ner_embeddings_samples = "data/ner_samples/embeddings/encoded_data_{client_id}_{language}.pkl"
ner_list_dict = "./data/ner_list.json"

# Fine-tuned NER model (cluster and sample)
ner_model_cluster = "data/ner_cluster/fine_tuned_similarity_model"
ner_model_samples = "data/ner_samples/fine_tuned_similarity_model"
FastSentenceTransformer = "all-MiniLM-L6-v2"

# NER JSON data (cluster and sample)
ner_json_cluster = "data/ner_cluster/json/{client_id}_{language}.json"
ner_json_samples = "data/ner_samples/json/ner_samples_{client_id}_{language}.json"


# Embeddings API URL
embeddings_api_url = "http://127.0.0.1:4657/embeddings"

# Process text API URL
# TODO: This is not working currently. Didn't get resolution on this one.
process_text_api_url = "http://127.0.0.1:4658/keywords"


# Path to csv dataframe
csv_file_path = "/data/master_dfs/{client_id}_{language}_master_df.csv"


# Retriever for Travel Chatbot
chatbot_data_path = "/home/pritika/travel-chatbot-backup/src/new2way_bot/data/travel_usecase_data.json"
chatbot_data_embeddings_path = "/home/pritika/travel-chatbot-backup/src/new2way_bot/data/travel_usecase_embedded.json"

#Transcriptionscorer
tcs_parameter_json = "/data/transcriptionscorer/merged_category_data/json/{client_id}_{language}_parameters.json"
tcs_parameter_embeddings = "/data/transcriptionscorer/merged_category_data/embeddings/{client_id}_{language}_parameters_dict.pkl"
tcs_phrase_importance = "/data/transcriptionscorer/keyword_importance/json/{client_id}_{language}_phrase_importance.json"
tcs_phrase_importance_embeddings = "/data/transcriptionscorer/keyword_importance/embeddings/{client_id}_{language}_phrase_importance.pkl"
tcs_scores = "/data/transcriptionscorer/score_with_ids/{scores_id}_{language}_scores.json"

 
# {
#     "language": "en",
#     "client_id": "12345",
#     "datasets": [
#         {
#             "data": [
#                 {
#                     "question": "How do I reset my password?",
#                     "goal_name": "reset_password"
#                 },
#                 {
#                     "question": "Where can I find my order history?",
#                     "goal_name": "order_history"
#                 }
#             ]
#         },
#         {
#             "data": [
#                 {
#                     "question": "What is the return policy?",
#                     "goal_name": "return_policy"
#                 }
#             ]
#         }
#     ]
# }

# curl -X POST http://164.52.200.229:9485/score_transcription \
# -H "Content-Type: application/json" \
# -d '{
#     "text": "Hello! I hope you are doing well today. I wanted to discuss an exciting opportunity...",
#     "language": "en",
#     "client_id": "1"
# }'

