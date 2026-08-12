from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class Review(BaseModel):
    summary: str = Field(description="A short summary of the review")
    sentiment: str = Field(description="One of: Very Positive, Excellent, Happy, Bad, Mixed")
    rating: int = Field(description="rating starting from one to five")
    improvement: str = Field(description="what can be improved")
    category: str = Field(description="which category this review is about")
    author: Optional[str] = Field(description="name of the author")


structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """I dont have a problem with the content. It did fulfill the purpose.
But you need to understand how human cognition works.
Creating slides where the content disappears and not depicting the entire process is not an art of teaching. my name is nitesh
When you want to explain a process, the slide to depict the same end-to-end and things should not disappear half way through the process."""
)
  
print('summary ', result.summary)
print(result.sentiment)
print(result.rating)
print(result.improvement)
print(result.category)
print(result.author)