from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import getpass
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage , SystemMessage , AIMessage
from langchain_core.tools import Tool
from langchain import hub
from langchain.agents import create_react_agent , AgentExecutor , create_structured_chat_agent
from langchain.memory import ConversationBufferMemory

load_dotenv()

llm = ChatGroq(model="deepseek-r1-distill-llama-70b")
# llm = ChatGroq(model="llama-3.3-70b-versatile")


def get_current_time(*args, **kwargs):
    """
    This function returns the current time in the format "YYYY-MM-DD HH:MM:SS".
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def search_wikipedia(query):
    """
    This function searches Wikipedia for the given query.
    """
    from wikipedia import summary

    try:
       return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that"
   

tools = [
    Tool(
      name="Time",
      description = "Get the current Dtae and time",
      func = get_current_time
    ),
    Tool(
        name="Wikipedia",
        description="Get the current Wikipedia",
        func=search_wikipedia
    )
]

prompt = hub.pull("hwchase17/structured-chat-agent")

memory = ConversationBufferMemory(memory_key="chat_history" ,return_messages=True)


agent = create_structured_chat_agent(
   llm=llm,
   tools=tools,
   prompt=prompt,)

agent_executor = AgentExecutor.from_agent_and_tools(
   agent=agent,
   tools=tools,
   verbose=True,
   memory=memory,
   handle_parsing_errors=True
)

initial_message= "You are Genz AI assitant that can provide helpfull answer using available tools and Genz and cool language.\nif you are unble to answer just say I dont know i am a chill guy."
memory.chat_memory.add_messages({'type':'system','content':initial_message})

# Initialize FastAPI app
app = FastAPI()

# Define request model
class ChatRequest(BaseModel):
    input: str

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    user_input = chat_request.input
    if not user_input:
        raise HTTPException(status_code=400, detail="No input provided")
    
    memory.chat_memory.add_messages({'type': 'user', 'content': user_input})    

    response = agent_executor.invoke({"input": user_input})
    # print(response)
    print("AI: ", response["output"])

    memory.chat_memory.add_messages({'type': 'ai', 'content': response["output"]})
    return {"response": response["output"]}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)