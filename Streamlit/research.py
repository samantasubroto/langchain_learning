import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

st.set_page_config(page_title="Subro_chat", page_icon="🤖")

st.title("Panda Chat")
st.write("Ask me anything!")

user_input = st.text_input("Enter your prompt:")

if st.button("Generate Response"):

    if user_input.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating response..."):
            response = model.generate_content(user_input)

        st.subheader("Response")
        st.write(response.text)