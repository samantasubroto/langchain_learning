from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

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
    encode_kwargs={"normalize_embeddings": True}
)


# =========================================================
# 4. CREATE VECTOR STORE
# =========================================================

vector_store = FAISS.from_documents(documents=chunks, embedding=embedding)
print("Documents stored in FAISS!")


# =========================================================
# 5. CREATE RETRIEVER
# =========================================================

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 15}
)


# =========================================================
# 6. GEMINI
# =========================================================

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
parser = StrOutputParser()


# =========================================================
# 7. QUERY REFORMULATION (so follow-ups like "why?" retrieve correctly)
# =========================================================

reformulate_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and a follow-up question, rewrite the follow-up "
               "as a standalone question that includes any needed context from the history. "
               "If it's already standalone, return it unchanged. Return ONLY the rewritten question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

reformulate_chain = reformulate_prompt | model | parser


# =========================================================
# 8. RAG PROMPT (loosened — allows reasoning/interpretation)
# =========================================================

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a thoughtful literary assistant answering questions about the book 
"Siddhartha" by Hermann Hesse.

Use the provided context as your primary source of truth. You may use your own understanding 
of the passage's meaning, themes, and implications to explain and interpret it — you don't 
need to just restate the context verbatim. Feel free to reason about motivations, symbolism, 
and connections, as long as it stays grounded in what the context supports.

If the context genuinely doesn't contain enough information to answer, say:
"I couldn't find enough information in the document to answer that."

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

rag_chain = rag_prompt | model | parser


# =========================================================
# 9. RAG LOOP
# =========================================================

chat_history = []  # list of HumanMessage / AIMessage

while True:

    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    # -----------------------------------------------------
    # Reformulate query using history (handles "why?" etc.)
    # -----------------------------------------------------

    standalone_query = reformulate_chain.invoke({
        "chat_history": chat_history,
        "question": query
    }) if chat_history else query

    # -----------------------------------------------------
    # Retrieve relevant chunks using the standalone query
    # -----------------------------------------------------

    retrieved_docs = retriever.invoke(standalone_query)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # -----------------------------------------------------
    # Send context + history + question to Gemini
    # -----------------------------------------------------

    answer = rag_chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": query
    })

    # -----------------------------------------------------
    # Update history
    # -----------------------------------------------------

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))

    # Keep history from growing unbounded
    if len(chat_history) > 12:
        chat_history = chat_history[-12:]

    print("\nAnswer:")
    print(answer)