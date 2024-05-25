from flask import Flask, request, jsonify
from src.modules.transcriptionscorer import TranscriptionScorer, TranscriptionPack
import json
from flask_cors import CORS
import logging
from src.config import (
    tcs_parameter_embeddings,
    tcs_parameter_json,
    tcs_phrase_importance,
    tcs_phrase_importance_embeddings,
    tcs_scores
)

app = Flask(__name__)
CORS(app, resources={r"/score_transcription": {"origins": "*"}})

logging.basicConfig(level=logging.DEBUG)

@app.before_request
def log_request_info():
    app.logger.debug('Headers: %s', request.headers)
    app.logger.debug('Body: %s', request.get_data())

@app.route('/score_transcription', methods=['POST'])
def score_transcription():
    data = request.json
    if not data or 'text' not in data or 'language' not in data or 'client_id' not in data or 'scores' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    text = data['text']
    language = data['language']
    client_id = data['client_id']
    scores = data['scores']

    scorer = TranscriptionScorer(text=text, language=language, client_id=client_id, scores=scores)
    doc = scorer.get_transcription_score(doc=TranscriptionPack(speaker1_full_text=text))
    
    response = {
        'transcription_score': doc.transcription_score
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='164.52.200.229', port=9485)