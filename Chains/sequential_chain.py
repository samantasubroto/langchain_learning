from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
parser = StrOutputParser()

prompt = PromptTemplate(
    template="Generate a detailed report on this {topic}",
    input_variables=["topic"]
)

refined = PromptTemplate(
    template="From this detailed report, extract a 5-line summary:\n{report}",
    input_variables=["report"]
)

chain = (
    prompt
    | model
    | parser
    | (lambda report: {"report": report})
    | refined
    | model
    | parser
)

result = chain.invoke({"topic": "OSHO"})

print(result)

chain.get_graph().print_ascii()