from flask import Flask, request, jsonify

app = Flask(__name__)


language_iso_mapping = {
    "ar": "arabic",
    "as": "assamese",
    "bn": "bengali",
    "en": "english",
    "fr": "french",
    "gu": "gujarati",
    "hi": "hindi",
    "kn": "kannada",
    "mr": "marathi",
    "pa": "punjabi",
    "ta": "tamil",
    "ur": "urdu",
}

language_stop_words = {}


def load_language_stop_words(lang):
    file_name = language_iso_mapping[lang]
    with open(f"{file_name}.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().splitlines()
        language_stop_words[lang] = list(set(stop_words))
    return language_stop_words


@app.route("/process_text", methods=["POST"])
def process_text():
    data = request.get_json()
    text = data.get("text", "")
    language_code = data.get("language", "en")

    words = text.split()
    result = load_language_stop_words(language_code)

    resul = []
    for word in words:
        if word.lower() in result[language_code]:
            continue
        else:
            resul.append(word)

    response = {"keywords": resul}

    return jsonify(response)


if __name__ == "__main__":
    app.run(port=8534, debug=True)
