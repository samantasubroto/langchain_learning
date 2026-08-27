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
        page_content="Jasprit Bumrah is an Indian fast bowler known for his deadly yorkers and exceptional accuracy."
    ),

    Document(
        page_content="Jasprit Bumrah is a world-class pace bowler who can bowl accurate yorkers in pressure situations."
    ),

    Document(
        page_content="Jasprit Bumrah is one of India's best fast bowlers and is famous for his accuracy and bowling variations."
    ),

    Document(
        page_content="Mohammed Shami is an Indian fast bowler known for his seam movement and ability to take wickets."
    ),

    Document(
        page_content="Virat Kohli is an Indian batsman known for his consistency and aggressive batting."
    ),

    Document(
        page_content="Rohit Sharma is an Indian opener famous for his explosive batting and six-hitting ability."
    ), 
    
    Document(
        page_content="Jadeja a good all-rounder"
    )
]


# -----------------------
# 2. Embeddings
# -----------------------

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    dimensions=300
)


# -----------------------
# 3. Vector Store
# -----------------------

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    collection_name="cricket_mmr"
)


# -----------------------
# 4. MMR Retriever
# -----------------------

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 6
    }
)


# -----------------------
# 5. Query
# -----------------------

query = "Tell me about Indian fast bowlers"

print("\nQUERY:")
print(query)


# -----------------------
# 6. Retrieve
# -----------------------

results = retriever.invoke(query)


# -----------------------
# 7. Display results
# -----------------------

print("\nRETRIEVED DOCUMENTS:")

for i, result in enumerate(results):
    print(f"\n--- Document {i + 1} ---")
    print(result.page_content)