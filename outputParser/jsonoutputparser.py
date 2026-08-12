from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

# Create JSON parser
parser = JsonOutputParser()

# Prompt
prompt = PromptTemplate(
    template="""
You are an expert information extractor.

Answer the user's query in JSON format.

{format_instructions}

Question:
{query}
""",
    input_variables=["query"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# Create chain
chain = prompt | llm | parser

# Invoke
result = chain.invoke({
    "query": "Tell me about Virat Kohli."
})

print(result)
print(type(result))