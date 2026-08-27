from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

text = """
LangChain is used to build LLM applications.
It provides tools for prompts and chains.
These chains can connect multiple components.

Python is a popular language for AI development.
It has many libraries for machine learning.
"""

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

splitter = SemanticChunker(embeddings)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)