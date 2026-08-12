from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Delhi is capital of India",
    "Kolkata is capital of WestBengal",
    "Paris is capital of France"
]

vector = embeddings.embed_documents(documents);
print(vector);