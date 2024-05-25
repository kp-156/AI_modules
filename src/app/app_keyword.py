from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import time

app = Flask(__name__)

@app.route('/keywords', methods=['POST'])
def extract_keywords():
    try:
        data = request.get_json()
        sentence = data['sentence'] if 'sentence' in data else ""

        if not isinstance(sentence, str):
            return jsonify({'error': 'Input must be a string'})

        start_time = time.time()
        vectorizer = TfidfVectorizer(max_features=10)  
        tfidf_vector = vectorizer.fit_transform([sentence])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_vector.toarray()[0]
        sorted_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
        print("For keywords --------- %s seconds ----------" % (time.time() - start_time))
        keywords = [keyword for keyword, score in sorted_keywords]

        response = {
            'keywords': keywords
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

# with app.conent():


if __name__ == '__main__':
    app.run(port="4658", debug=True)