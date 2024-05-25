from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig
from langchain_community.llms import HuggingFacePipeline
#from transformers.pipelines import TextStreamer
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Model2A class for general conversation processing with Llama 2 7B
class Model1a:
    def __init__(self):
        self.model_id = "meta-llama/Llama-2-7b-hf"  # Use Llama 2 7B or Llama 2 13B
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            load_in_8bit=True  # Use 8-bit to reduce memory consumption
        )

        # Configure generation settings for Llama 2 7B
        self.generation_config = GenerationConfig(
            max_length=512,
            temperature=0.75,
            top_p=0.95,
            repetition_penalty=1.15
        )

        # Text streamer for efficient output processing
        #self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Pipeline for text generation with Llama 2 7B
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            generation_config=self.generation_config,
            #treamer=self.streamer,
            batch_size=12,
            do_sample=True
        )

        # HuggingFacePipeline for conversational processing
        self.llm = HuggingFacePipeline(pipeline=self.pipe)

        # Conversation template for a generalized response
        self.template = """
        ###Instruction: You are a general chatbot designed to assist with various queries.
        Please respond to user queries in a helpful and concise manner.

        Current conversation:
        {chat_history}

        ###User: {input}
        ###Response: """.strip()

        # Conversation buffer memory for context management
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False
        )

        # Create a conversation chain with the defined prompt and memory
        self.conversation = ConversationChain(
            llm=self.llm,
            verbose=False,
            memory=self.memory,
            prompt=PromptTemplate(
                input_variables=["chat_history", "input"],
                template=self.template
            )
        )
