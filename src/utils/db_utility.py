import pymongo

mongo_host = "localhost"
mongo_port = 27017
database_name = "bot1"
collection_name = "travel_bot"

client = pymongo.MongoClient(mongo_host, mongo_port)
db = client[database_name]
collection = db[collection_name]


def get_ner_ngram_from_db(language, client_id, sentence):
    """
    This function queries the database for ner_ngram based on client_id, language and goal_name as sentence
    If present, it returns the ner_ngram value otherwise returns an empty dict
    """
    query = {"client_id": client_id, "language": language, "question": sentence}
    conversation_doc = collection.find_one(query)
    print(f"Conversation doc for sentence {sentence}: {conversation_doc}")

    if conversation_doc and "ner_ngram" in conversation_doc:
        return conversation_doc["ner_ngram"]
    else:
        return {}


def get_conv_data_from_goal(language, client_id, sentence):
    """
    This function queries the database bases on client_id, language and goal_name as sentence
    If present, it returns the conversation_doc otherwise returns False
    """
    try:
        query = {"client_id": client_id, "language": language, "goal_name": sentence}
        conversation_doc = collection.find_one(query)
        if conversation_doc:
            return conversation_doc
    except Exception as e:
        print(f" get_conv_data_from_goal Error: {str(e)}")
    return False


def get_intent_from_db(sentence, language, client_id):
    try:
        print(f"Fetching intent i.e., goal from DB for sentence: {sentence}")
        query = {"client_id": client_id, "language": language, "question": sentence}
        conversation_doc = collection.find_one(query)

        if conversation_doc:
            print(conversation_doc.keys())
            return conversation_doc["goal_name"]
        else:
            print("NO GOAL FOUND")
            return False
    except Exception as e:
        print(f"Error while fetching intent from DB: {str(e)}")
        return False


def get_current_goal(public_id):
    query = {"public_id": public_id}
    conversation_doc = collection.find_one(query)
    if "current_goal_name" in conversation_doc:
        return conversation_doc["current_goal_name"]


def set_intent(goal, public_id):
    try:
        result = collection.update_one(
            {"public_id": public_id},
            {"$set": {"current_goal_name": goal, "new_intent": True}},
            upsert=True,
        )

        if result.modified_count > 0:
            print(f"Setting intent for  public_id '{public_id}' updated in MongoDB.")
            return True
        else:
            print(f"Setting intent for public_id '{public_id}' not found in MongoDB.")
            return False
    except Exception as e:
        print(f"set_intent An error occurred: {str(e)}")
        return False


def set_conversation_data(data, public_id):
    try:
        result = collection.update_one(
            {"public_id": public_id}, {"$set": {"data": data}}
        )

        if result.modified_count > 0:
            print(
                f"Setting conversation  for  public_id '{public_id}' updated in MongoDB."
            )
            return True
        else:
            print(
                f"Setting conversation  for public_id '{public_id}' not found in MongoDB."
            )
            return False
    except Exception as e:
        print(f"set_conversation_data An error occurred: {str(e)}")
        return False


def fill_ner_values(public_id, keys_and_values):
    try:
        current_document = collection.find_one({"public_id": public_id})
        slots_to_ask = current_document.get("slots_to_ask")

        updated_slots_to_ask = [
            slot for slot in slots_to_ask if slot not in keys_and_values.keys()
        ]

        collection.update_one(
            {"public_id": public_id}, {"$set": {"slots_to_ask": updated_slots_to_ask}}
        )

        goal_slots = current_document.get("goal_slots", [])
        for slot_name, slot_value in keys_and_values.items():
            if slot_name not in goal_slots:
                goal_slots.append({slot_name: slot_value[0]})

        collection.update_one(
            {"public_id": public_id}, {"$set": {"goal_slots": goal_slots}}
        )

        return True

    except Exception as e:
        print(f"fill_ner_values An error occurred: {str(e)}")
        return False


def set_slot_keys(public_id, data):
    try:
        result = collection.update_one(
            {"public_id": public_id}, {"$set": {"slots_to_ask": data}}
        )

        if result.modified_count > 0:
            print(
                f"Setting slot ner_object for  public_id '{public_id}' updated in MongoDB."
            )
            return True
        else:
            print(
                f"Setting slot ner_object for public_id '{public_id}' not found in MongoDB."
            )
            return False
    except Exception as e:
        print(f" set_slot_keys An error occurred: {str(e)}")
        return False


def set_ner_true(public_id):
    print("Setting ner_object to -> True")
    try:
        result = collection.update_one(
            {"public_id": public_id}, {"$set": {"ner_done": True}}
        )

        if result.modified_count > 0:
            print(
                f"Setting slot ner_object for  public_id '{public_id}' updated in MongoDB."
            )
            return True
        else:
            print(
                f"Setting slot ner_object for public_id '{public_id}' not found in MongoDB."
            )
            return False
    except Exception as e:
        print(f" set_slot_keys An error occurred: {str(e)}")
        return False


def get_slots_to_ask(public_id):
    try:
        slots_to_ask = collection.find_one(
            {"public_id": public_id}, {"_id": 0, "slots_to_ask": 1}
        )

        if slots_to_ask is not None and "slots_to_ask" in slots_to_ask:
            return slots_to_ask["slots_to_ask"]

        else:
            print(f"Slots to ask for public_id '{public_id}' not found in MongoDB.")
            return []
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []


def get_slots_to_ask_(public_id):
    try:
        slots_to_ask = collection.find_one(
            {"public_id": public_id}, {"_id": 0, "slots_to_ask": 1}
        )

        if slots_to_ask is not None and "slots_to_ask" in slots_to_ask:
            return slots_to_ask["slots_to_ask"]

        else:
            print(f"Slots to ask for public_id '{public_id}' not found in MongoDB.")
            return []
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []


def store_conversation_logic(data):
    try:
        collection.insert_one(data)
        print("store_conversation_logic: Success")
    except Exception as e:
        print("Error in store_conversation_logic:", str(e))
