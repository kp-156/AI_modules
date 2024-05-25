from typing import Dict, Optional
from src.modules.intent_ner_processor import IntentNERProcessor
from jina import Flow, requests, Executor
from docarray import BaseDoc, DocList

from benedict import benedict
from benedict.serializers import JSONSerializer
import json
# s = JSONSerializer()
from logger_ import execute_and_log, replace_keys_in_nested_dict

class Pack(BaseDoc):
    item : bytes = JSONSerializer().encode(benedict(keyattr_dynamic=True))

class IntentExec(Executor):
    
    @requests
    async def process(self, docs: DocList[Pack], **kwargs) -> DocList[Pack]:
        for doc in docs:
            packet = benedict(JSONSerializer().decode(doc.item))

            if 'intent' in packet['tasks']:
                lang = packet['tasks','intent','input','lang']
                c_id = packet['tasks','intent','input','client_id']
                input_ = {
                    'text' : packet['tasks','intent','input','text'],
                    'threshold' :  0.7
                }
                print("Input -->",input_)
                intentdetector = IntentNERProcessor(text=input_['text'], language=lang, client_id=c_id)
                result = execute_and_log(intentdetector.process_intent, **input_)
                result = replace_keys_in_nested_dict(result, '.', '<period>')
                print(result)
                packet['tasks','intent'] = result['sentences'] 
                packet['tasks','intent','type'] = result['type'] 
                packet['tasks','intent','intents'] = result['intents']  
                # print(packet.to_json(indent= 2))
            doc.item = JSONSerializer().encode(packet)
        return docs

def main():
    f = Flow(port=11000)
    f = f.add(uses=IntentExec)

    # with f:
    #     f.block()

# """
    with f:
        text = "I am looking for a hotel in Paris from 4th July to 7th July"
        item = benedict(keyattr_dynamic=True)
        item['tasks', 'intent'] = True
        item['tasks', 'intent', 'input', 'lang'] = 'en'
        item['tasks', 'intent', 'input', 'client_id'] = '1'
        item['tasks', 'intent', 'input', 'text'] = text
        input_ = Pack(item=JSONSerializer().encode(item))
        resp = f.post(on='/', inputs=[input_], return_type=DocList[Pack])
        for res in resp:
            dict_b = benedict(JSONSerializer().decode(res.item))
            print(dict_b.to_json(indent=2))

# """

if __name__ == "_main_":
    main()