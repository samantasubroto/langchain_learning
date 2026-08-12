import os
import streamlit as st
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="Dynamic Prompt Generator", page_icon="🤖")

## testing the prompt need to refactor later todo
## mathna nahi.

domain = st.selectbox(
    "Knowledge Source",
    [
        "Research Papers",
        "Programming",
        "Mathematics",
        "General Knowledge",
        "Medical",
        "Finance"
    ]
)

answer_type = st.selectbox(
    "Response Style",
    [
        "Basic Explanation",
        "Code Intensive",
        "Mathematical",
        "Step-by-Step",
        "Professional"
    ]
)

length = st.selectbox(
    "Response Length",
    [
        "Short",
        "Medium",
        "Large"
    ]
)

question = st.text_area(
    "Ask your question",
    height=150
)

# ------------------------------
# Prompt Template
# ------------------------------

PROMPT_TEMPLATE = """
You are an expert AI assistant.

### Instructions

Knowledge Domain:
{domain}

Response Style:
{answer_type}

Response Length:
{length}

User Question:
{question}

Rules:

1. Answer ONLY using verified knowledge.
2. Never fabricate facts.
3. Never hallucinate.
4. If the answer is uncertain or information is insufficient, reply ONLY with:

Not enough information.

5. Do not guess.
6. Keep the answer focused.
7. Match the requested response style.
8. Match the requested response length.
9. Do not include unnecessary introductions or conclusions.
"""

# ------------------------------
# Button
# ------------------------------

if st.button("Generate"):

    if question.strip() == "":
        st.warning("Please enter a question.")
    else:

        final_prompt = PROMPT_TEMPLATE.format(
            domain=domain,
            answer_type=answer_type,
            length=length,
            question=question
        )

        with st.spinner("Generating..."):

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=final_prompt
            )

        st.subheader("Answer")

        st.write(response.text)

        with st.expander("Generated Prompt"):
            st.code(final_prompt)