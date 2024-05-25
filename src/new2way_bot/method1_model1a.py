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
        ### Instruction: You are a travel assistant chatbot designed to help customers with their travel-related queries.
        Conversation Flow:
        - Start with a greeting or introduction.
        - Ask for clarification or additional details if needed.
        - Respond to the user's queries or requests for assistance using information from the database response.
        - End the conversation after providing necessary information or suggesting further actions (e.g., visiting a website for booking).

        ### User Query: {user_query}
        ### Response from DB: {response_from_db}
        ### Response: """.strip()

        # Create Conversation Chain without memory
        self.conversation = ConversationChain(
            llm=self.pipeline,
            prompt=PromptTemplate(
                input_variables=["user_query", "response_from_db"],
                template=self.template,
                template_format="f-string",
                validate_template=False
            )
        )

    def generate_response(self, user_query: str, response_from_db: str):
        # Generate response using the conversation chain
        response = self.conversation.invoke(
            user_query=user_query,
            response_from_db=response_from_db
        )

        # Additional processing or refinement of the response can be done here
        # (e.g., summarizing, adding call to action)

        return response

# Example usage
chatbot = Chatbot()

# Sample user query
user_query = "I want to book a flight from New York to Paris for 2 people on May 15th. What are my options?"

# Sample response retrieved from the database (simulate actual DB interaction)
response_from_db = "Here are some flight options from our partners: \n - [Airline A] departing at [time], arriving at [time], starting from [price]. \n - [Airline B] departing at [time], arriving at [time], starting from [price]."

# Generate a response using the user query and response from the database
response = chatbot.generate_response(user_query, response_from_db)

# Print the generated response
print("Generated Response:", response)
