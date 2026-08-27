from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# =========================================================
# 1. LOAD PDF
# =========================================================

loader = PyPDFLoader("sidd-encr.pdf")

pages = loader.load()

print(f"Number of pages: {len(pages)}")


# =========================================================
# 2. CHUNKING
# =========================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)

chunks = splitter.split_documents(pages)

print(f"Number of chunks: {len(chunks)}")


# =========================================================
# 3. EMBEDDING MODEL
# =========================================================

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    encode_kwargs={
        "normalize_embeddings": True
    }
)


# =========================================================
# 4. CREATE VECTOR STORE
# =========================================================

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embedding
)

print("Documents stored in FAISS!")


# =========================================================
# 5. CREATE RETRIEVER
# =========================================================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# =========================================================
# 6. GEMINI
# =========================================================

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)

parser = StrOutputParser()


# =========================================================
# 7. RAG PROMPT
# =========================================================

prompt = PromptTemplate(
    template="""
You are answering questions about a book.

Use ONLY the provided context to answer the question.

If the answer cannot be found in the context, say:
"I couldn't find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)


# =========================================================
# 8. RAG LOOP
# =========================================================

while True:

    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    # -----------------------------------------------------
    # Retrieve relevant chunks
    # -----------------------------------------------------

    retrieved_docs = retriever.invoke(query)

    # Combine retrieved chunks
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # -----------------------------------------------------
    # Send context + question to Gemini
    # -----------------------------------------------------

    chain = prompt | model | parser

    answer = chain.invoke({
        "context": context,
        "question": query
    })

    print("\nAnswer:")
    print(answer)