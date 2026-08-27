from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

loader = PyPDFLoader("Enlightenment, the Only Revolution (The Mahageeta, Vol 1) (Osho) (Z-Library).pdf")

pages = loader.load()

print(f"Total pages: {len(pages)}")


# -----------------------
# Gemini
# -----------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)

parser = StrOutputParser()


# -----------------------
# Page summary chain
# -----------------------

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

page_summary_chain = page_prompt | model | parser


# -----------------------
# Run all pages in parallel
# -----------------------

parallel_chain = RunnableParallel(
    {
        f"page_{i}": page_summary_chain
        for i in range(len(pages))
    }
)

page_inputs = [
    {"page": page.page_content}
    for page in pages
]

summaries = page_summary_chain.batch(page_inputs)


# -----------------------
# Combine summaries
# -----------------------

combined_summaries = "\n\n".join(
    summaries.values()
)


# -----------------------
# Final summary
# -----------------------

final_prompt = PromptTemplate(
    template="""
You are given summaries of all pages of a PDF.

Create one comprehensive summary of approximately 300 lines.

Combine related information and remove repetition.
Preserve important facts, concepts and conclusions.
Do not introduce information that isn't present in the summaries.

PAGE SUMMARIES:

{summaries}
""",
    input_variables=["summaries"]
)

final_chain = final_prompt | model | parser

final_summary = final_chain.invoke({
    "summaries": combined_summaries
})

print(final_summary)