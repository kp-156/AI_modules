from typing import Dict, Optional
from jina import Flow, requests, Executor
from docarray import BaseDoc, DocList
from benedict import benedict
from benedict.serializers import JSONSerializer
import json
from logger_ import execute_and_log, replace_keys_in_nested_dict
from src.modules.intent_ner_processor import IntentNERProcessor
from jina import Document, Client

class Pack(BaseDoc):
    item: bytes = JSONSerializer().encode(benedict(keyattr_dynamic=True))

class NerExec(Executor):
    
    @requests
    async def process(self, docs: DocList[Pack], **kwargs) -> DocList[Pack]:
        for doc in docs:
            packet = benedict(JSONSerializer().decode(doc.item))

            if 'ner' in packet['tasks']:
                lang = packet['tasks', 'ner', 'input', 'lang']
                c_id = packet['tasks', 'ner', 'input', 'client_id']
                input_ = {
                    'language': lang,
                    'text': packet['tasks', 'ner', 'input', 'text'],
                    'client_id': c_id,
                    'ner_ngram': packet['tasks', 'ner', 'input', 'ner_ngram']
                }
                intent_ner_obj = IntentNERProcessor(text=input_['text'], language=input_['language'], client_id=input_['client_id'])
                ner_ngram = execute_and_log(intent_ner_obj.process_ner, **input_)
                ner_ngram = replace_keys_in_nested_dict(ner_ngram, '.', '<period>')
                result = intent_ner_obj.process_ner(ner_ngram)
                packet['tasks', 'ner'] = result
            doc.item = JSONSerializer().encode(packet)
        return docs

def test_executor():
    flow = Flow(port=11001).add(uses=NerExec)

    item = benedict(keyattr_dynamic=True)
    item['tasks', 'ner'] = True
    item['tasks', 'ner', 'input', 'lang'] = 'en'
    item['tasks', 'ner', 'input', 'client_id'] = '1'
    item['tasks', 'ner', 'input', 'text'] = 'I am John Doe and I live at 123 Elm Street'
    item['tasks', 'ner', 'input', 'ambig_or_multi'] = None
    item['tasks', 'ner', 'input', 'ner_ngram'] = {'names': [1, 3], 'addresses': [3, 8]}

    pack = Pack(item=JSONSerializer().encode(item))

    with flow:
        client = Client(port=11001)
        response = client.post(on='/', inputs=[pack], return_type=DocList[Pack])

        assert len(response) == 1
        result_packet = benedict(JSONSerializer().decode(response[0].item))

        assert 'tasks' in result_packet
        assert 'ner' in result_packet['tasks']
        print("Test passed!")

if __name__ == "__main__":
    test_executor()
