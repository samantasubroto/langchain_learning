from huggingface_hub import InferenceClient
import json
from pydantic import BaseModel

HF_TOKEN = ""

client = InferenceClient(token=HF_TOKEN)

class Review(BaseModel):
    summary: str
    sentiment: str
    rating: int

review = """
The course content was excellent but the slides were confusing.
"""

prompt = f"""
Extract the following review.

Return ONLY valid JSON.

Schema:

{{
    "summary": "",
    "sentiment": "",
    "rating": 0
}}

Review:
{review}
"""

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

text = response.choices[0].message.content

print(text)

data = json.loads(text)

review = Review.model_validate(data)

print(review)