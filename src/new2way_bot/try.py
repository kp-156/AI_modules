from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextGenerationPipeline
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

class Chatbot:
    def __init__(self):
        self.model_id = "gpt2"  # or specify any other GPT model
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to end-of-sequence token
        self.model = GPT2LMHeadModel.from_pretrained(self.model_id)

        # Create text generation pipeline
        self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer)

        self.template = """
        ###Instruction: You are a chatbot designed to provide assistance based on user queries.
        Conversation Flow:
        - Start with a greeting or introduction.
        - Respond to the user's queries based on their intent, named entities, and information from the database.
        - End the conversation after providing necessary information.

        ###User Query: {user_query}
        ###User Intent: {user_intent}
        ###NER Dictionary: {ner_dictionary}
        ###Response from DB: {response_from_db}
        ###Response: """.strip()

        # Create Conversation Chain
        self.conversation = ConversationChain(
            llm=self.pipeline,
            prompt=PromptTemplate(
                input_variables=['user_query', 'user_intent', 'ner_dictionary', 'response_from_db'],
                template=self.template,
                template_format='f-string',
                validate_template=False
            )
        )

    def generate_response(self, user_query: str, user_intent: str, ner_dictionary: dict, response_from_db: str):
        # Generate response
        response = self.conversation.invoke(
            user_query=user_query,
            user_intent=user_intent,
            ner_dictionary=ner_dictionary,
            response_from_db=response_from_db
        )

        return response

# Example usage
chatbot = Chatbot()

# Sample inputs
user_query = "What are the top attractions in Paris?"
user_intent = "get_attractions"
ner_dictionary = {"location": "Paris"}
response_from_db = "Here are some top attractions in Paris: Eiffel Tower, Louvre Museum, Notre-Dame Cathedral."

# Generate a response using the provided inputs
response = chatbot.generate_response(user_query, user_intent, ner_dictionary, response_from_db)

# Print the generated response
print("Generated Response:", response)
