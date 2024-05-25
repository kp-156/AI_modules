from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextGenerationPipeline
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class Chatbot:
    def __init__(self):
        self.model_id = "gpt2"  # or specify any other GPT model
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token to end-of-sequence token
        self.model = GPT2LMHeadModel.from_pretrained(self.model_id)

        # Create text generation pipeline
        self.pipeline = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer)

        self.template = """
        ###Instruction: You are a travel assistant chatbot designed to help customers with their Easemytrip travel-related queries.
        In order to assist customers effectively, your chatbot should be able to handle a variety of travel-related inquiries, including travel and hotel information, flight details, Easemytrip service, and general assistance.
        Your task is to develop a function or script that takes two parameters, `input` and `history`, and generates appropriate responses based on the user's queries and the conversation history. Chatbot should maintain context throughout the conversation, provide accurate and relevant information, and ensure customer satisfaction.
        Remember to strictly adhere to the conversation flow provided and end the conversation after gathering all the necessary input and chat history.

        Conversation Flow:
        - Start the conversation with a greeting or introduction.
        - Respond to the user's queries or requests for assistance.
        - Gather necessary information from the user.
        - Provide confirmation or additional details as needed.
        - End the conversation after receiving all required input.
        Make the output short simple and concise
        Strictly give responses of 70 words and no more

        Current conversation:
        {chat_history}
        ###User: [INST]{input}[/INST]\n.
        ###Response: """.strip()

        self.memory = ConversationBufferMemory(
                                        memory_key='chat_history',
                                        return_messages=False
                                        )
        
        # Create Conversation Chain without memory
        self.conversation = ConversationChain(
            llm=self.pipeline,
            memory= self.memory,
            prompt=PromptTemplate(
                                
                                input_variables=["chat_history", "user_input"],
                                template=self.template,
                                template_format="f-string",
                                validate_template=False
            )
        )

    def generate_response(self, user_query: str):
        # Generate response using the conversation chain
        response = self.conversation.predict(
            user_query=user_query
        )

        # Additional processing or refinement of the response can be done here
        # (e.g., summarizing, adding call to action)

        return response


chatbot = Chatbot()
chatbot.conversation.memory.clear()
# user_query = "I want to book a flight from New York to Paris for 2 people on May 15th. What are my options?"
user_input="i want to book ticket for paris"
# history= ""

response_from_db = "Here are some flight options from our partners: \n - [Airline A] departing at [time], arriving at [time], starting from [price]. \n - [Airline B] departing at [time], arriving at [time], starting from [price]."

response = chatbot.generate_response(user_input)

print("Generated Response:", response)
