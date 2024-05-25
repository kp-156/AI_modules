import dotenv
import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import gradio as gr

dotenv.load_dotenv('/.env')
HF_ACCESS_TOKEN = os.getenv('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')


model_id = 'TheBloke/Nous-Hermes-13B-GPTQ'
# Configure for 4-bit quantization (optimizes model deployment)
# bnb_config = BitsAndBytesConfig(
#     # bnb_4bit_compute_dtype = 'float16',
#     # bnb_4bit_quant_type='nf4',
#     load_in_4bit=True,
# )

# Load model configuration
model_config = AutoConfig.from_pretrained(
    model_id,
    # use_auth_token=HF_ACCESS_TOKEN
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # config=model_config,
    device_map='auto',
    # quantization_config=bnb_config,
    # use_auth_token=HF_ACCESS_TOKEN
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    # use_auth_token=HF_ACCESS_TOKEN
)

# Set model into evaluation mode (optimizes inference)
model.eval()
# Set up the text-generation pipeline
pipe = pipeline(
                    "text-generation",
                    model= model,
                    tokenizer= tokenizer,
                    max_length=5096,
                    temperature=0.75,
                    top_p=0.95,
                    repetition_penalty=1.15,
                    batch_size=12,
                    do_sample=True
)


llm = HuggingFacePipeline(pipeline=pipe)

# Template using jinja2 syntax
template = """
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

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template,
    template_format="jinja2"
)

# Initialize the conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    prompt=prompt,
    verbose=False
)
# Start the conversation
def predict(message: str, *history: str):
    
    response = conversation.predict(input=message)

    return response

message = input("query: ")
history = []
print(predict(message, history))