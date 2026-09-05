import os
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tools import tools, get_document

load_dotenv()


# -------------------------
# 1. Agent State
# -------------------------


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# -------------------------
# 2. LLM
# -------------------------

model = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
).bind_tools(tools)


# -------------------------
# 3. Agent Node
# -------------------------


def our_agent(state: AgentState) -> AgentState:

    system_prompt = SystemMessage(
        content=f"""
You are Drafter, a helpful writing assistant.

You help the user create, update, and save documents.

Rules:
- If the user wants to update or modify content,
  use the 'update' tool with the complete updated content.
- If the user wants to save and finish,
  use the 'save' tool.
- Always show the current document state after modifications.

Current document content:
{get_document()}
"""
    )

    # First interaction
    if not state["messages"]:
        user_input = (
            "I'm ready to help you update a document. What would you like to create?"
        )

        user_message = HumanMessage(content=user_input)

    # Later interactions
    else:
        user_input = input("\nWhat would you like to do with the document? ")

        print(f"\n👤 USER: {user_input}")

        user_message = HumanMessage(content=user_input)

    # Give the LLM:
    # system prompt
    # previous conversation
    # new user message

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": (list(state["messages"]) + [user_message, response])}


# -------------------------
# 4. Decide What Happens Next
# -------------------------


def should_continue(state: AgentState) -> str:
    """Determine if the graph should continue or end."""

    messages = state["messages"]

    if not messages:
        return "continue"

    for message in reversed(messages):
        if (
            isinstance(message, ToolMessage)
            and "saved" in message.content.lower()
            and "document" in message.content.lower()
        ):
            return "end"

    return "continue"


# -------------------------
# 5. Build Graph
# -------------------------

graph = StateGraph(AgentState)


graph.add_node("agent", our_agent)

graph.add_node("tools", ToolNode(tools))


graph.set_entry_point("agent")


graph.add_edge("agent", "tools")


graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)


# Compile graph
app = graph.compile()


# -------------------------
# 6. Run Agent
# -------------------------


def run_document_agent():

    print("\n===== DRAFTER =====")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n===== DRAFTER FINISHED =====")


# -------------------------
# 7. Print Tool Messages
# -------------------------


def print_messages(messages):

    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")
