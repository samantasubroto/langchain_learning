from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

response_schemas = [
    ResponseSchema(
        name="name",
        description="Name of the cricketer"
    ),
    ResponseSchema(
        name="country",
        description="Country of the cricketer"
    ),
    ResponseSchema(
        name="profession",
        description="Profession"
    ),
    ResponseSchema(
        name="achievements",
        description="List of major achievements"
    ),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

prompt = PromptTemplate(
    template="""
Answer the following question.

{format_instructions}

Question:
{question}
""",
    input_variables=["question"],
    partial_variables={
        "format_instructions": format_instructions
    }
)

chain = prompt | llm | parser

result = chain.invoke({
    "question": "Tell me about Virat Kohli."
})

print(result)
print(type(result))