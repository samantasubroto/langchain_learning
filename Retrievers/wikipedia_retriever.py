from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=2,
    doc_content_chars_max=3000
)

results = retriever.invoke("Virat Kohli")

for doc in results:
    print(doc.page_content)