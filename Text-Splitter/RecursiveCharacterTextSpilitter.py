from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
LangChain is a framework for building applications with large language models.
It provides tools for prompts, models, output parsers, chains, retrievers, and more.

Text splitting is important when working with large documents.
We split large documents into smaller chunks before sending them to an LLM.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)