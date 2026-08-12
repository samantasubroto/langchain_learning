from huggingface_hub import InferenceClient

HF_TOKEN = ""

client = InferenceClient(token=HF_TOKEN)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages = [
        {"role": "user", "content": "Who is the Prime Minister of India?"},
        {"role": "assistant", "content": "Narendra Modi."},
        {"role": "user", "content": "What did I just ask?"}
    ]
)

print(response.choices[0].message.content)