from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langgraph.prebuilt import tools_condition
from typing import TypedDict
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import START, StateGraph,END
from langgraph.prebuilt import ToolNode
from fastapi import FastAPI
import matplotlib.pyplot as plt
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles





builder: StateGraph = StateGraph(MessagesState)
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0
)


def multiply(a: int, b: int) -> int:
    """Multiply a and b .

    Args:
        a: first int
        b: second int
    """
    return a * b

tools=[multiply]

llm_with_tools = llm.bind_tools(tools)


def assistant1(state: MessagesState) -> MessagesState:
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


def assistant2(state: MessagesState) -> MessagesState:
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


builder.add_node("assistant", assistant1)
builder.add_node("multiply",multiply)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")

builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition
)
builder.add_edge("tools",'assistant')

print(llm.invoke('hi').content)

graph: CompiledStateGraph = builder.compile()



# Generate the mermaid graph using graph.get_graph().draw_mermaid_png()
graph_data = graph.get_graph().draw_mermaid_png()  # Replace with appropriate method if necessary

with open("mermaid_graph.png", "wb") as f:
    f.write(graph_data)

print("Mermaid graph saved as mermaid_graph.png")

# Assuming graph_data is a byte string representing the image data:
# plt.imshow(plt.imread(io.BytesIO(graph_data)))
# plt.show()



# messages = [HumanMessage(content="Multiply 3 by 2.")]
# response=graph.invoke({'messages':messages})


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/')
def home():
    return {"message": "Welcome to the LangGraph!"}

@app.post("/chat2")
def chat2(input_message: str):
    messages = [HumanMessage(content=input_message)]
    response = llm.invoke( messages)
    return {response.content}


@app.post("/chat")
def chat1(input_message: str):
    messages = [HumanMessage(content=input_message)]
    response = graph.invoke({'messages': messages})
    return {"response": response["messages"][-1].content}

@app.post("/chat2")
def chat2(input_message: str):
    messages = [HumanMessage(content=input_message)]
    response = llm.invoke( messages)
    return {response.content}
