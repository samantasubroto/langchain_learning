from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Virat Kohli is one of the greatest batsmen in modern cricket. He has represented India across all formats and captained the national team for several years. Kohli is known for his aggressive batting style and exceptional fitness. He has scored more than 80 international centuries during his career. His consistency has made him one of the most successful cricketers in the world.",
    "Sachin Tendulkar is widely regarded as the God of Cricket. He made his international debut at the age of sixteen. Tendulkar is the highest run-scorer in international cricket history. He became the first player to score a double century in One Day Internationals. In 2014, he was honored with the Bharat Ratna, India's highest civilian award.",
    "MS Dhoni is one of the most successful captains in Indian cricket history. He led India to victory in the 2007 T20 World Cup, the 2011 ODI World Cup, and the 2013 Champions Trophy. Dhoni is famous for his calm leadership and finishing abilities. He is also considered one of the best wicketkeepers in the game. His helicopter shot became one of the most iconic strokes in cricket.",
    "Rohit Sharma is known for his elegant batting and record-breaking performances in One Day Internationals. He is the only player to score three double centuries in ODI cricket. Rohit has captained the Indian national team in multiple formats. He has also led the Mumbai Indians franchise to several IPL titles. His ability to play long innings makes him one of the best opening batsmen in the world.",
    "Jasprit Bumrah is India's premier fast bowler across all formats. He is known for his unique bowling action and deadly yorkers. Bumrah has played a crucial role in India's victories in overseas conditions. He has consistently been ranked among the top bowlers in the ICC rankings. His accuracy and ability to bowl under pressure make him one of the finest fast bowlers in modern cricket."
]

query = 'tell me about virat kohli'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_documents(query)

print(cosine_similarity([query_embedding], doc_embeddings))