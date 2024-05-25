from transformers import AutoTokenizer
model_id = "TheBloke/Nous-Hermes-13B-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_id)
max_input_length = tokenizer.model_max_length

print(f"Maximum input length for {model_id}: {max_input_length}")
