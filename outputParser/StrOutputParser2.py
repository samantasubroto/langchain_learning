from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

template1 = PromptTemplate(
    template="""
Write a detailed report on {topic}.

Return only the report.
""",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="""
Write a concise 5-line summary of the following text.

{text}
""",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = (
    template1
    | llm
    | parser
    | template2
    | llm
    | parser
)

result = chain.invoke({
    "topic": "Black Hole"
})

print(result)