from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

result = model.invoke('When is Subrotho Samantha birthday?')
print(result.content)

while True: 
    user_input = input('You: ')
    if user_input.lower == 'exit': 
        break
    result = model.invoke(user_input)
    print("AI: ", result.content)