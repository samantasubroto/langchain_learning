from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash')
parser = StrOutputParser()

prompt = PromptTemplate(
    template='Generate a detailed report on this {topic}',
    input_variables=['topic']
)

chain = prompt | model | parser

result = chain.invoke({'topic':'exercise'})

print(result) 

chain.get_graph().print_ascii()