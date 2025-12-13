from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

from typing import TypedDict 

load_dotenv()

class Review(TypedDict):
    summary:str
    sentiment : str


model1 = ChatOllama(model = "gemma3:1b")
model2 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")


out_model1 = model1.with_structured_output(Review)
out_model2 = model2.with_structured_output(Review)



prompt = "its too difficult in learning ai will you help me"

a1 = out_model1.invoke(prompt)
a2 = out_model2.invoke(prompt)


print(a1)
