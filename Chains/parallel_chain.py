from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel

load_dotenv()

model_one = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
model_two = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text\n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question and answers from the follwoing text\n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model_one | parser,
    'quiz': prompt2 | model_two | parser
}) | prompt3 | model_one | parser

text = """Begin by setting a clear accuracy goal for your use case, where you’re clear on the accuracy that would be “good enough” for this use case to go to production. You can accomplish this through:

Setting a clear accuracy target: Identify what your target accuracy statistic is going to be.
For example, 90 of customer service calls need to be triaged correctly at the first interaction.
Developing an evaluation dataset: Create a dataset that allows you to measure the model’s performance against these goals.
To extend the example above, capture 100 interaction examples where we have what the user asked for, what the LLM triaged them to, what the correct triage should be, and whether this was correct or not.
Using the most powerful model to optimize: Start with the most capable model available to achieve your accuracy targets. Log all responses so we can use them for distillation of a smaller model.
Use retrieval-augmented generation to optimize for accuracy
Use fine-tuning to optimize for consistency and behavior
During this process, collect prompt and completion pairs for use in evaluations, few-shot learning, or fine-tuning. This practice, known as prompt baking, helps you produce high-quality examples for future use."""

result = parallel_chain.invoke({'text': text})

print(result)

parallel_chain.get_graph().print_ascii()