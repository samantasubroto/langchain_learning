from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
LangChain is a framework for building applications with large language models.
It provides tools for prompts, models, output parsers, chains, retrievers, and more.
Text splitting is important when working with large documents.
We split large documents into smaller chunks before sending them to an LLM.
"""

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=50,
    chunk_overlap=10,
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)