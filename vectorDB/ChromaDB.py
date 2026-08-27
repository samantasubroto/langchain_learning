from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv


load_dotenv()


# -----------------------
# 1. Some text
# -----------------------

text = """
Virat Kohli: Legendary Indian batsman known for consistency, aggression, and chasing big targets.

Rohit Sharma: Explosive opener famous for his elegant batting and incredible six-hitting ability.

Jasprit Bumrah: World-class fast bowler known for deadly yorkers and exceptional accuracy.

MS Dhoni: Legendary captain and wicketkeeper famous for his calm leadership and finishing ability.

Sachin Tendulkar: Cricket icon and one of the greatest batsmen in the history of the game.

Ravindra Jadeja: World class all-rounder, knows for this quick fielding and best reflexes.

Zaheer Khan: Best Bowler of Indian Team that we ever witnessed, a quick left-arm break bowler.
"""


# -----------------------
# 2. Split by paragraphs
# -----------------------

paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

chunks = [Document(page_content=p) for p in paragraphs]


# -----------------------
# 3. Create embeddings
# -----------------------

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    dimensions=300
)


# -----------------------
# 4. Store in Chroma
# -----------------------

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name="cricketers"
)

print("\nDocuments stored in Chroma!")


# -----------------------
# 5. Query
# -----------------------

query = "Who is the best bowler from this?"

results = vector_store.similarity_search(
    query,
    k=1
)

print("\n--- Result ---")
for item in results:
    print(item)