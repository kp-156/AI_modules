import pymongo
import json 
from pymongo import MongoClient
import requests
from config import mongo_client
# mongo_host = 'localhost'
# mongo_port = 27017
# database_name = 'bot'
# collection_name = 'conversation'

# client = pymongo.MongoClient(mongo_host, mongo_port)

def get_conv_data_from_goal(language, client_id, sentence, mongo_host, mongo_port, database_name, collection_name):
    """
    Retrieves conversation data from the specified MongoDB collection based on the given parameters.

    Args:
        language (str): The language of the conversation.
        client_id (str): The client ID associated with the conversation.
        sentence (str): The sentence or question of the conversation.
        mongo_host (str): The host address of the MongoDB server.
        mongo_port (int): The port number of the MongoDB server.
        database_name (str): The name of the MongoDB database.
        collection_name (str): The name of the MongoDB collection.

    Returns:
        dict or None: The conversation document if found, or None if not found.

    Raises:
        Exception: If there is an error while retrieving the conversation data.

    """
    try:
        client = pymongo.MongoClient(mongo_host, mongo_port)
        db = client[database_name]
        collection = db[collection_name]
        query = {"client_id": client_id, "language": language, "question": sentence}
        conversation_doc = collection.find_one(query)

        if conversation_doc:
            return conversation_doc

    except Exception as e:
        print(f" get_conv_data_from_goal Error: {str(e)}")
        return None



def get_intent(sentence, language, client_id, mongo_host, mongo_port, database_name, collection_name):
    """
    Retrieves the intent goal name for a given sentence.

    Args:
        sentence (str): The input sentence to find the intent goal for.
        language (str): The language of the sentence.
        client_id (str): The client ID.
        mongo_host (str): The MongoDB host.
        mongo_port (int): The MongoDB port.
        database_name (str): The name of the MongoDB database.
        collection_name (str): The name of the MongoDB collection.

    Returns:
        str: The intent goal name for the given sentence.

    Raises:
        Exception: If an error occurs while retrieving the intent.

    """
    try:
        print("Text to find goal ->", sentence)
        client = pymongo.MongoClient(mongo_host, mongo_port)
        db = client[database_name]
        collection = db[collection_name]
        query = {"client_id": client_id, "language": language, "question": sentence}
        conversation_doc = collection.find_one(query)

        if conversation_doc:
            conversation = conversation_doc
            print(conversation.keys())
        else:
            print("NO GOAL FOUND")
        return conversation["goal_name"]

    except Exception as e:
        print(f"get_intent Error: {str(e)}")
        return False




def set_intent(goal, public_id, mongo_host, mongo_port, database_name, collection_name):
    """
    Sets the intent for a given public_id in MongoDB.

    Args:
        goal (str): The goal to set for the public_id.
        public_id (str): The unique identifier for the document in MongoDB.
        mongo_host (str): The hostname or IP address of the MongoDB server.
        mongo_port (int): The port number of the MongoDB server.
        database_name (str): The name of the database in MongoDB.
        collection_name (str): The name of the collection in MongoDB.

    Returns:
        bool: True if the intent was successfully set, False otherwise.
    """
    try:
        client = pymongo.MongoClient(mongo_host, mongo_port)
        db = client[database_name]
        collection = db[collection_name]
        result = collection.update_one(
            {"public_id": public_id},
            {"$set": {"current_goal_name": goal}},
            upsert=True
        )

        if result.modified_count > 0:
            print(f"Setting intent for public_id '{public_id}' updated in MongoDB.")
            return True
        else:
            print(f"Setting intent for public_id '{public_id}' not found in MongoDB.")
            return False
    except Exception as e:
        print(f"set_intent An error occurred: {str(e)}")
        return False





def set_conversation_data(data, public_id, mongo_host, mongo_port, database_name, collection_name):
    """
    Sets the conversation data for a given public_id in MongoDB.

    Args:
        data (dict): The conversation data to be set.
        public_id (str): The public_id associated with the conversation.
        mongo_host (str): The MongoDB host.
        mongo_port (int): The MongoDB port.
        database_name (str): The name of the MongoDB database.
        collection_name (str): The name of the MongoDB collection.

    Returns:
        bool: True if the conversation data was successfully set, False otherwise.
    """
    try:
        client = pymongo.MongoClient(mongo_host, mongo_port)
        db = client[database_name]
        collection = db[collection_name]

        result = collection.update_one(
            {"public_id": public_id},
            {"$set": {"data": data}}
        )

        if result.modified_count > 0:
            print(f"Setting conversation for public_id '{public_id}' updated in MongoDB.")
            return True
        else:
            print(f"Setting conversation for public_id '{public_id}' not found in MongoDB.")
            return False
    except Exception as e:
        print(f"set_conversation_data An error occurred: {str(e)}")
        return False
    

def fill_ner_values(public_id, keys_and_values, mongo_host, mongo_port, database_name, collection_name):
    """
    Updates the slots_to_ask and goal_slots fields in the MongoDB collection for a given public_id.

    Args:
        public_id (str): The public_id of the document to update.
        keys_and_values (dict): A dictionary containing the slot names as keys and their corresponding values.
        mongo_host (str): The hostname of the MongoDB server.
        mongo_port (int): The port number of the MongoDB server.
        database_name (str): The name of the MongoDB database.
        collection_name (str): The name of the MongoDB collection.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    try:
        client = pymongo.MongoClient(mongo_host, mongo_port)
        db = client[database_name]
        collection = db[collection_name]

        current_document = collection.find_one({"public_id": public_id})
        slots_to_ask = current_document.get("slots_to_ask")

        updated_slots_to_ask = [slot for slot in slots_to_ask if slot not in keys_and_values.keys()]

        collection.update_one(
            {"public_id": public_id},
            {"$set": {"slots_to_ask": updated_slots_to_ask}}
        )

        goal_slots = current_document.get("goal_slots", [])
        for slot_name, slot_value in keys_and_values.items():
            goal_slots.append({slot_name : slot_value})

        collection.update_one(
            {"public_id": public_id},
            {"$set": {"goal_slots": goal_slots}}
        )

        return True

    except Exception as e:
        print(f"fill_ner_values An error occurred: {str(e)}")
        return False


def set_slot_keys(public_id, data, mongo_host, mongo_port, database_name, collection_name):
    """
    Sets the slot keys for a given public_id in MongoDB.

    Args:
        public_id (str): The public_id for which the slot keys need to be set.
        data (dict): The slot keys data to be set.
        mongo_host (str): The MongoDB host address.
        mongo_port (int): The MongoDB port number.
        database_name (str): The name of the MongoDB database.
        collection_name (str): The name of the MongoDB collection.

    Returns:
        bool: True if the slot keys were successfully set, False otherwise.
    """
    try:
        client = pymongo.MongoClient(mongo_host, mongo_port)
        db = client[database_name]
        collection = db[collection_name]

        result = collection.update_one(
            {"public_id": public_id},
            {"$set": {"slots_to_ask": data}}
        )

        if result.modified_count > 0:
            print(f"Setting slot ner for public_id '{public_id}' updated in MongoDB.")
            return True
        else:
            print(f"Setting slot ner for public_id '{public_id}' not found in MongoDB.")
            return False
    except Exception as e:
        print(f"set_slot_keys: An error occurred: {str(e)}")
        return False




def get_ner_ngram(language, client_id, sentence, mongo_host, mongo_port, database_name, collection_name):
    """
    Retrieves the named entity recognition (NER) n-gram for a given language, client ID, and sentence.

    Args:
        language (str): The language of the sentence.
        client_id (str): The ID of the client.
        sentence (str): The input sentence.
        mongo_host (str): The host address of the MongoDB server.
        mongo_port (int): The port number of the MongoDB server.
        database_name (str): The name of the MongoDB database.
        collection_name (str): The name of the MongoDB collection.

    Returns:
        dict: A dictionary containing the NER n-gram for each slot in the sentence.
    """
    client = pymongo.MongoClient(mongo_host, mongo_port)
    db = client[database_name]
    collection = db[collection_name]
    query = {"client_id": client_id, "language": language, "question": sentence}
    conversation_doc = collection.find_one(query)

    if conversation_doc:
        slots_dict = conversation_doc.get("slots", {})
        result = {}

        for k, v in slots_dict.items():
            if "ner_ngram" in v:
                result[k] = v["ner_ngram"]

    return result


class DIALOGFLOW_CLASS:

    def __init__(self, database_name, collection_name):
        client = MongoClient(mongo_client)  
        db = client[database_name]  
        self.collection = db[collection_name]  

            
            
    def get_conv_data(self, public_id):
        """
        Retrieve conversation data from MongoDB based on the provided public_id.

        Args:
            public_id (str): The public_id of the conversation.

        Returns:
            dict or None: The conversation data as a dictionary if found, None otherwise.
        """
        try:
            conversation_json = self.collection.find_one({"public_id": public_id}, {"_id": 0})

            if conversation_json is not None:
                return conversation_json["data"]
            else:
                print(f"Conversation with public_id '{public_id}' not found in MongoDB.")
                return None
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None


    def get_slots_to_ask(self, public_id):
        """
        Retrieves the slots to ask for a given public ID from MongoDB.

        Args:
            public_id (str): The public ID to retrieve slots for.

        Returns:
            list: A list of slots to ask for the given public ID. If no slots are found, an empty list is returned.
        """
        try:
            slots_to_ask = self.collection.find_one({"public_id": public_id}, {"_id": 0, "slots_to_ask": 1})

            if slots_to_ask is not None and "slots_to_ask" in slots_to_ask:
                return slots_to_ask["slots_to_ask"]
            
            else:
                print(f"Slots to ask for public_id '{public_id}' not found in MongoDB.")
                return []
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return []


    def set_slots_to_ask(self, public_id, slots_to_ask):
        """
        Sets the slots to ask for a given public ID in MongoDB.

        Args:
            public_id (str): The public ID associated with the slots.
            slots_to_ask (list): The list of slots to ask.

        Returns:
            bool: True if the slots were successfully updated in MongoDB, False otherwise.
        """
        try:
            result = self.collection.update_one(
                {"public_id": public_id},
                {"$set": {"slots_to_ask": slots_to_ask}}
            )

            if result.modified_count > 0:
                print(f"Slots to ask for public_id '{public_id}' updated in MongoDB.")
                return True
            else:
                print(f"Slots to ask for public_id '{public_id}' not found in MongoDB.")
                return False
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return False

    def get_current_goal_name(self, public_id):
        """
        Retrieves the current goal name for a given public ID from MongoDB.

        Args:
            public_id (str): The public ID associated with the conversation.

        Returns:
            str or None: The current goal name if found in MongoDB, None otherwise.
        """
        try:
            conversation_json = self.collection.find_one({"public_id": public_id}, {"_id": 0, "current_goal_name": 1})

            if conversation_json is not None and "current_goal_name" in conversation_json:
                return conversation_json["current_goal_name"]
            else:
                print(f"Current goal name for public_id '{public_id}' not found in MongoDB.")
                return None
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None



    def set_slot_and_value(self, public_id, slot_name, slot_value):
        """
        Sets a slot and its corresponding value for a given public_id in MongoDB.

        Args:
            public_id (str): The public_id associated with the slot.
            slot_name (str): The name of the slot.
            slot_value (str): The value to be set for the slot.

        Returns:
            bool: True if the slot and value are successfully added to MongoDB, False otherwise.
        """
        try:
            result = self.collection.update_one(
                {"public_id": public_id},
                {"$push": {"goal_slots": {slot_name: slot_value}}}
            )

            if result.modified_count > 0:
                print(f"Slot '{slot_name}' with value '{slot_value}' added for public_id '{public_id}' in MongoDB.")
                return True
            else:
                print(f"Public_id '{public_id}' not found in MongoDB.")
                return False
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return False



    def get_goal_slots_and_value(self, public_id):
        """
        Retrieves the goal slots and their values for a given public ID from MongoDB.

        Args:
            public_id (str): The public ID to search for in MongoDB.

        Returns:
            list: A list of goal slots for the given public ID. If the public ID is not found, an empty list is returned.

        """
        try:
            document = self.collection.find_one({"public_id": public_id})

            if document:
                goal_slots = document.get("goal_slots", [])
                return goal_slots
            else:
                print(f"Public_id '{public_id}' not found in MongoDB.")
                return []

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return []





    def clear_data(self, public_id):
        """
        Clears specific fields in the MongoDB document associated with the given public_id.

        Args:
            public_id (str): The public_id of the document to clear the fields for.

        Raises:
            Exception: If an error occurs while updating the MongoDB document.

        Returns:
            None
        """
        try:
            self.collection.update_one({"public_id": public_id}, {"$unset": {"slots_to_ask": ""}})
            print(f"Slots to ask for public_id '{public_id}' cleared in MongoDB.")

            self.collection.update_one({"public_id": public_id}, {"$unset": {"current_goal_name": ""}})
            print(f"Current goal name for public_id '{public_id}' cleared in MongoDB.")

            self.collection.update_one({"public_id": public_id}, {"$unset": {"goal_slots": ""}})
            print(f"Slots field for public_id '{public_id}' cleared in MongoDB.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")



    def get_exhaustive_from_api(url, method, data, response):
        """
        Sends a request to the specified API endpoint and returns the data from the response.

        Args:
            url (str): The URL of the API endpoint.
            method (str): The HTTP method to use for the request.
            data (dict): The data to be sent with the request.
            response (str): The key to extract the data from the response.

        Returns:
            The data extracted from the response if the request is successful and the response contains the specified key.
            None if an error occurs during the request or the specified key is not found in the response.
        """
        try:
            result = requests.request(url, method, json.dumps(data))
            if result.status == 200:
                try:
                    dat = result[response]
                    return dat
                except KeyError:
                    print(f"Key '{response}' not found in the response.")
                    return None
            else:
                print(f"An error occurred in {url} with method {method} and data as {str(data)}")
                return None
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None
        


