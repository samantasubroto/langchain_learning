from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_chroma import Chroma


load_dotenv()


# =========================================================
# 1. DOCUMENT
# =========================================================

text = """
Virat Kohli is one of India's greatest batsmen. He is known for
his consistency, aggressive batting style and ability to chase
large targets.

Rohit Sharma is an explosive opening batsman. He is famous for
his elegant batting style and exceptional six-hitting ability.

Jasprit Bumrah is one of India's best fast bowlers. He is known
for his deadly yorkers, accuracy and ability to perform under
pressure.

MS Dhoni is a legendary Indian captain and wicketkeeper. He is
known for his calm leadership, finishing ability and excellent
decision making.

Sachin Tendulkar is one of the greatest batsmen in cricket
history. He represented India for many years and holds numerous
records.
"""


document = Document(page_content=text)


# =========================================================
# 2. CHUNKING
# =========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents([document])

print(f"Number of chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content)


# =========================================================
# 3. EMBEDDING MODEL
# =========================================================

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    dimensions=300
)


# =========================================================
# 4. STORE CHUNKS IN CHROMA
# =========================================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="cricket_rag"
)

print("\nDocuments stored in Chroma!")


# =========================================================
# 5. CREATE RETRIEVER
# =========================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)


# =========================================================
# 6. USER QUERY
# =========================================================

query = "Who is India's best fast bowler?"


# =========================================================
# 7. RETRIEVE RELEVANT DOCUMENTS
# =========================================================

retrieved_docs = retriever.invoke(query)

print("\n==============================")
print("QUERY")
print("==============================")

print(query)


print("\n==============================")
print("RETRIEVED DOCUMENTS")
print("==============================")

for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Document {i + 1} ---")
    print(doc.page_content)


# =========================================================
# 8. GEMINI LLM
# =========================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)


# =========================================================
# 9. CREATE CONTEXT
# =========================================================

context = "\n\n".join(
    doc.page_content
    for doc in retrieved_docs
)


# =========================================================
# 10. PROMPT
# =========================================================

prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context,
say "I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


# =========================================================
# 11. RAG CHAIN
# =========================================================

rag_chain = (
    prompt
    | model
    | StrOutputParser()
)


# =========================================================
# 12. GENERATE ANSWER
# =========================================================

answer = rag_chain.invoke({
    "context": context,
    "question": query
})


print("\n==============================")
print("FINAL ANSWER")
print("==============================")

print(answer)