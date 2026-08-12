from dotenv import load_dotenv

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Pydantic schema
class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    occupation: str = Field(description="Occupation")
    country: str = Field(description="Country")
    skills: list[str] = Field(description="List of skills")

# Create parser
parser = PydanticOutputParser(pydantic_object=Person)

# Prompt
prompt = PromptTemplate(
    template="""
Extract the following information.

{format_instructions}

Text:
{text}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# Create chain
chain = prompt | llm | parser

# Input
result = chain.invoke({
    "text": """
    John is a 30-year-old software engineer from the United States.
    He is skilled in Python, Java, and Machine Learning.
    """
})

print(result)
print(type(result))

print(result.name)
print(result.age)
print(result.occupation)
print(result.country)
print(result.skills)