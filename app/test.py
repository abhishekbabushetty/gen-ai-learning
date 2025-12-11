from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


model = ChatOllama(model="gemma3:1b")


chat_history = [SystemMessage(content = "You are an AI mentor for student abhishek help with his query and be like mentor")]

while True:
    n = input("USER : ")
    chat_history.append(HumanMessage(content = n))
    out = model.invoke(chat_history)
    if(n == "exit"):
        break
    chat_history.append(AIMessage(content = out.content))
    print("AI : ",out.content)


