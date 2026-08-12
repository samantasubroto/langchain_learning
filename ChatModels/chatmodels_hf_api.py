from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

##Downloading the modal in local system and then running

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

messages = [
    {"role": "user", "content": "Who is the longest continues serving Prime Minister of India?"}
]

prompt = pipe.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

result = pipe(prompt, max_new_tokens=50)

print(result[0]["generated_text"])