from langchain import PromptTemplate, HuggingFaceHub, LLMChain

class ResponseModifier:
    def __init__(self, model_name="google/pegasus-large"):
        self.model_name = model_name
        self.prompt = PromptTemplate(
            template="### Instruction: Modify the retrieved response based on the user's query, intent, and named entities.\nUser Intent: {}\nNER Dictionary: {}\nUser Query: {}\n\n### Response: {}\n",
            input_variables=["user_intent", "ner_dict", "user_query", "retrieved_response"]
        )
        self.llm = LLMChain(
            prompt=self.prompt,
            llm=HuggingFaceHub(repo_id=self.model_name, model_kwargs={"temperature": 0, "max_length": 512})
        )

    def modify_response(self, user_intent, ner_dict, user_query, retrieved_response):
        modified_response = self.llm.invoke(user_intent=user_intent, ner_dict=ner_dict, user_query=user_query, retrieved_response=retrieved_response)
        return modified_response





# Example usage
user_intent = "flight_search"
ner_dict = {"location": ["chennai", "mumbai"], "date": "may 12"}
user_query = "i want to book flight travel from chennai to Mumbai on may 12"
retrieved_response = "When booking a flight, compare prices across different platforms to find the best deal. Use flight comparison websites to save time and money. you can search flight for dubai to france"

response_modifier = ResponseModifier()
modified_response = response_modifier.modify_response(user_intent, ner_dict, user_query, retrieved_response)
print("Modified Response:", modified_response)
