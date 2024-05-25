from flask import Flask, request, jsonify

#from chatbot.dialog_flow import main

from src.modules.intent_ner_processor import IntentNERProcessor

# app = Flask(__name)


# @app.route('/chatbot', methods=['POST'])
def chatbot(data):
    # data = request.get_json()

    client_id = data.get("client_id", None)
    text = data.get("text", None)
    public_id = data.get("public_id", None)

    if not client_id or not text:
        return {"error": "Please send text and client id"}

    language = get_language(text)
    intent_ner_processor = IntentNERProcessor(text, language, client_id)
    intent, multi = intent_ner_processor.process_intent()
    print("intent ->", intent)

    if intent:
        int_name = get_intent(intent, language, client_id)
        if int_name:
            set_intent(int_name, public_id)

        data = get_conv_data_from_goal(language, client_id, intent)

        if data:
            set_conversation_data(data, public_id)

        set_slot_keys(public_id, list(data["slots"].keys()))

        # ngram
        ner_ngram = get_ner_ngram(language, client_id, intent)

        ner_result = intent_ner_processor.process_ner(ner_ngram)
        print("NER Result ->", ner_result)

        if ner_result:
            fill_ner_values(public_id, ner_result)

    # dialog
    try:
        if ner_result:
            response = main(public_id, text, ner=True)
        else:
            response = main(public_id, text)
        return response["message"]

    except Exception as e:
        print("Error occured in dialog manager", str(e))
        return {"error": "error occured in dialog manager"}


# print(
#     chatbot(
#         data={
#             "client_id": "4",
#             "public_id": "edrfghbjn",
#             "text": "i want to purchase a product with name vikas jangra at f 32 sector 20 noida up 201301",
#         }
#     )
# )
# print(chatbot(data = {"client_id" : "1", "public_id" : "edrfghbjn", "text" : "12354567"}))


# if __name__ == '__main__':
#    app.run(port="5079",debug=True)
