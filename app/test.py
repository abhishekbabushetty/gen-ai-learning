from langchain_ollama import ChatOllama
from lanchain_core import SystemMessage, HumanMessage, AIMessage


model = ChatOllama()


chat_history = [SystemMessage(content = "You are an AI mentor for student abhishek help with his query and be like mentor")]

while True:
    n = input("USER : ")
    chat_history.append(HumanMessage(n))
    out = model.invoke(chat_history)
    chat_history.append(AIMessage(out))
    print("AI : ",out)


