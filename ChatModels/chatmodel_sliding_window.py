from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

chat_history = []
WINDOW_SIZE = 3  # Number of conversation pairs to remember

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))

    # Keep only the last 3 conversation pairs (6 messages)
    if len(chat_history) > WINDOW_SIZE * 2:
        chat_history = chat_history[-WINDOW_SIZE * 2:]

    print("\nCurrent Context:")
    for msg in chat_history:
        print(f"{msg.type}: {msg.content}")

    result = model.invoke(chat_history)

    print("\nAI:", result.content)

    chat_history.append(AIMessage(content=result.content))

    # Again trim after adding the AI response
    if len(chat_history) > WINDOW_SIZE * 2:
        chat_history = chat_history[-WINDOW_SIZE * 2:]