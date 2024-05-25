from src.modules.intent import IntentDetector
from src.modules.ner_utility import NER_UTILITY


def update_intent_embeddings():
    """
    This function re-writes the embedding and keyword embedding files based on the latest state of client dataframe
    """
    language = "en"
    client_id = "1"
    intent_detector = IntentDetector(language=language, client_id=client_id)
    intent_detector.update_embeddings()
    intent_detector.update_keyword_embeddings()


def update_ner_embeddings():
    """
    This function re-writes the embeddings files based on the latest state of ner_sample/json and ner_cluster/json
    """
    language="en"
    client_id="1"
    obj = NER_UTILITY(language=language, client_id=client_id)
    obj.encode_and_save_cluster_embeddings()
    obj.encode_and_save_sample_embeddings()

if __name__ == "__main__":
    # update_intent_embeddings()
    update_ner_embeddings()
