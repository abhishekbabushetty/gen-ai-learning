from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv


load_dotenv()


model = ChatGoogleGenerativeAI(model = "gemma-3-12b-it")
prompt = "what is capital of india"
result = model.invoke(prompt)


print(result.content)
