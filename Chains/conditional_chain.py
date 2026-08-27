from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from enum import Enum
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  # see note below
str_parser = StrOutputParser()


# -------------------------
# Sentiment classifier
# -------------------------

class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"


class SentimentResult(BaseModel):
    sentiment: Sentiment


pydantic_parser = PydanticOutputParser(pydantic_object=SentimentResult)

sentiment_prompt = PromptTemplate(
    template="""
Analyze the sentiment of the following customer statement.

{format_instructions}

Customer statement:
{feedback}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instructions": pydantic_parser.get_format_instructions()
    }   
)

sentiment_chain = sentiment_prompt | model | pydantic_parser


# Positive response

positive_prompt = PromptTemplate(
    template="""
    The customer gave positive feedback.

    Thank them warmly for their feedback.
    Keep the response concise.

    Customer feedback:
    {feedback}
    """,
    input_variables=["feedback"]
)

positive_chain = positive_prompt | model | str_parser



negative_prompt = PromptTemplate(
    template="""
    The customer gave negative feedback.

    Apologize politely and tell them to contact
    customer care for further assistance.

    Customer feedback:
    {feedback}
    """,
    input_variables=["feedback"]
)

negative_chain = negative_prompt | model | str_parser


classification_chain = RunnableParallel(
    feedback=lambda x: x["feedback"],
    sentiment=sentiment_chain
)


feedback_chain = (
    classification_chain
    | RunnableBranch(
        (
            lambda x: x["sentiment"].sentiment == "positive",
            positive_chain
        ),
        (
            lambda x: x["sentiment"].sentiment == "negative",
            negative_chain
        ),
        negative_chain
    )
)



result = feedback_chain.invoke({
    "feedback": "The product itself is honestly amazing and works exactly as advertised, "
                "but the delivery took three weeks longer than promised, the packaging was "
                "damaged, and your support team never responded to my emails. I'll probably "
                "buy from you again, but this whole experience left a bad taste in my mouth."
})

print(result)