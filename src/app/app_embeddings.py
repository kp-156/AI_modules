from flask import Flask, request, jsonify
from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer

app = Flask(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda", quantize=True)

@app.route('/embeddings', methods=['POST'])
def get_embeddings():
    try:
        data = request.get_json()
        texts = data['texts'] if 'texts' in data else []

        if isinstance(texts, str):
            texts = [texts]

        embeddings = model.encode(texts, convert_to_tensor=True)
        embeddings = embeddings.tolist()  

        response = {
            'embeddings': embeddings
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port="4657",debug=True)

