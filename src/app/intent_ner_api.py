from src.modules.intent_ner_processor import IntentNERProcessor
from src.modules.intent_utility import INTENT_UTILITY
from src.modules.ner_utility import NER_UTILITY
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
from src.config import ner_list_dict
with open(ner_list_dict, 'r') as file:
    ner_dict = json.load(file)

app = Flask(__name__)
CORS(app)

@app.route('/find_ner', methods=['POST'])
def ner_api():
    try:
        data = request.json
        if 'text' not in data or 'ner_ngram' not in data:
            return jsonify({'error': 'Invalid input data'}), 400
        language = data.get('language', 'en')
        client_id = data.get('client_id', '1')

        print("data['ner_ngram']", data['ner_ngram'])
        with open(ner_list_dict, 'r') as f:
            ner_list = json.load(f)

        ner_category = {}
        for ngram in data['ner_ngram']:
            for key, values in ner_list.items():
                if ngram == key:
                    if key in ner_category:
                        ner_category[key].extend(values) 
                    else:
                        ner_category[key] = values
                    
        print("ner_category: ", ner_category)
        ner = IntentNERProcessor(text=data['text'], language=language, client_id=client_id)
        result = ner.process_ner(ner_category)

        return jsonify({'response': result})
    
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/find_intent', methods=['POST'])
def intent_api():
    try:
        data = request.json
        if 'text' not in data:
            return jsonify({'error': 'Invalid input data'}), 400
        input_sentence = data['text']
        language = data.get('language', 'en')
        client_id = data.get('client_id', '1')

        intent_ner_processor = IntentNERProcessor(text=input_sentence, language=language, client_id=client_id)
        result = intent_ner_processor.process_intent()
        unique_intent = None if result["intents"] is None else result["intents"][0]

        if unique_intent is not None:
            formatted_intent = unique_intent.replace('_', ' ')
        else:
            formatted_intent = "Please ask domain related question"
        
        print(
        f"selected sentences -> {result['sentences']}\n"
        f"selected intent -> {formatted_intent}\n"
        f"intent type (multi or ambiguous) -> {result['type']}\n"
        )

        return jsonify({'response': formatted_intent.title()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/add_ner', methods=['POST'])
def add_ner():
    try:
        data = request.json
        required_keys = ['tag', 'values', 'example_phrases', 'language', 'client_id']
        if not all(key in data for key in required_keys):
            return jsonify({'error': 'Invalid input data'}), 400

        add_new_ner = NER_UTILITY(language=data["language"], client_id=data["client_id"])

        sample_range = add_new_ner.find_min_max_word_lengths(data["values"])
        logging.info(f"Sample word length range for tag {data['tag']}: {sample_range}")

        ner_ngram = {data['tag']: sample_range}

        for category, values in ner_ngram.items():
            if category in ner_dict:
                ner_dict[category] = list(set(ner_dict[category] + values))
            else:
                ner_dict[category] = values

        with open(ner_list_dict, 'w') as file:
            json.dump(ner_dict, file, indent=2)

        add_new_ner.update_cluster_json(ner_ngram=data["ner_ngram"], values=data["example_phrases"])
        add_new_ner.encode_and_save_cluster_embeddings()
        add_new_ner.update_samples_json(ner_ngram=data["ner_ngram"], values=data["values"])
        add_new_ner.encode_and_save_sample_embeddings()
        
        return jsonify({'message': 'NER data processed successfully'}), 200
    
    except Exception as e:
        logging.error(f"Error in add_ner: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/add_intent', methods=['POST'])
def add_intent():
    try:
        data = request.json
        if not all(key in data for key in ("language", "client_id", "datasets")):
            return jsonify({'error': 'Missing required fields in the request'}), 400

        add_new_intents = INTENT_UTILITY(language=data["language"], client_id=data["client_id"])

        for dataset in data["datasets"]:
            for item in dataset.get("data", []):
                try:
                    sentence = item["question"]
                    goal_name = item["goal_name"]

                    add_new_intents.update_dataframe(sentence, goal_name)
                    add_new_intents.create_embeddings()
                    add_new_intents.create_keyword_embeddings()
                    print("process_text_logic: Success")
                except KeyError as e:
                    print(f"Missing key in item: {e}")
                except Exception as e:
                    print(f"Error in process_text_logic: {e}")

        return jsonify({'message': 'Intent data processed successfully'}), 200

    except Exception as e:
        logging.error(f"Error in add_intent: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='164.52.200.229', port=9484)