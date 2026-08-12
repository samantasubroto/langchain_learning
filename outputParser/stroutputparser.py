from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

##Downloading the modal in local system and then running

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it", 
)

llm = HuggingFacePipeline(pipeline=pipe)

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Write a 5 line summary on given text. /n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | llm | parser | template2 | llm | parser

report = chain.invoke({'topic':'Black Hole'})

print(report)