from langchain_google_genai  import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv



load_dotenv()


chat_history = [SystemMessage(content = "you are an ai mentor today keep messages conversation very short")]



model = ChatOllama(model="gemma3:1b")


while(True):

    a = input("Human : ")
    chat_history.append(HumanMessage(content = a))
    if a == "exit":
        break

    response = model.invoke(chat_history)

    chat_history.append(AIMessage(content = response.content))
  
    print("AI :",response.content)


print(chat_history)