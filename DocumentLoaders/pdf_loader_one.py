from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Load PDF
loader = PyPDFLoader(
    "Enlightenment, the Only Revolution (The Mahageeta, Vol 1) (Osho) (Z-Library).pdf"
)

pages = loader.load()

print(f"Total pages: {len(pages)}")


# Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)

parser = StrOutputParser()


# Prompt
page_prompt = PromptTemplate(
    template="""
Summarize the following page of a PDF.

Keep the important information, facts and concepts.
Do not add information that isn't present in the page.

PAGE:
{page}
""",
    input_variables=["page"]
)


# Chain
page_summary_chain = page_prompt | model | parser


# Page 150
page_150 = pages[150].page_content


# Generate summary
summary = page_summary_chain.invoke({
    "page": page_150
})

print(summary)