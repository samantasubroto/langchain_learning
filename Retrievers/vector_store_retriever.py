from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


# -----------------------
# 1. Documents
# -----------------------

documents = [
    Document(
        page_content="Virat Kohli is a legendary Indian batsman known for consistency and aggression."
    ),
    Document(
        page_content="Rohit Sharma is an explosive Indian opener famous for his six-hitting ability."
    ),
    Document(
        page_content="Jasprit Bumrah is a world-class fast bowler known for deadly yorkers and accuracy."
    ),
    Document(
        page_content="MS Dhoni is a legendary Indian captain and wicketkeeper known for calm leadership."
    ),
    Document(
        page_content="Sachin Tendulkar is one of the greatest batsmen in cricket history."
    )
]


# -----------------------
# 2. Embedding model
# -----------------------

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    dimensions=300
)


# -----------------------
# 3. Create Vector Store
# -----------------------

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    collection_name="cricketers"
)


# -----------------------
# 4. Create Retriever
# -----------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 1}
)


# -----------------------
# 5. Query
# -----------------------

query = "Who is a fast bowler?"

results = retriever.invoke(query)


# -----------------------
# 6. Print results
# -----------------------

for result in results:
    print("\n--- Result ---")
    print(result.page_content)