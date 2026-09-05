# rag_agent.py

import os
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from rag_tools import tools

# -------------------------
# 1. Environment
# -------------------------

load_dotenv()


# -------------------------
# 2. LLM
# -------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


# Give the LLM access to our tools
llm = llm.bind_tools(tools)


# -------------------------
# 3. Agent State
# -------------------------


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# -------------------------
# 4. System Prompt
# -------------------------

system_prompt = """
You are an intelligent AI assistant who answers
questions about Stock Market Performance in 2024
based on the PDF document loaded into your
knowledge base.

Use the retriever tool available to answer
questions about the stock market performance data.

You can make multiple calls if needed.

If you need to look up some information before
asking a follow-up question, you are allowed
to do that.

Please always cite the specific parts of the
documents you use in your answers.
"""


# -------------------------
# 5. Tool Dictionary
# -------------------------

tools_dict = {tool_instance.name: tool_instance for tool_instance in tools}


# -------------------------
# 6. LLM Node
# -------------------------


def call_llm(state: AgentState) -> AgentState:
    """
    Send the current conversation to the LLM.
    """

    messages = list(state["messages"])

    # Add system prompt before conversation
    messages = [SystemMessage(content=system_prompt)] + messages

    # Ask the LLM
    message = llm.invoke(messages)

    # Add the AI response to state
    return {"messages": [message]}


# -------------------------
# 7. Decide Whether to Use Tool
# -------------------------


def should_continue(state: AgentState):
    """
    Check whether the LLM requested a tool call.
    """

    result = state["messages"][-1]

    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


# -------------------------
# 8. Retriever Agent Node
# -------------------------


def take_action(state: AgentState) -> AgentState:
    """
    Execute the tools requested by the LLM.
    """

    tool_calls = state["messages"][-1].tool_calls

    results = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]

        query = tool_call["args"].get("query", "")

        print(f"\nCalling Tool: {tool_name}")

        print(f"Query: {query}")

        # Check whether the requested tool exists
        if tool_name not in tools_dict:
            print(f"\nTool: {tool_name} does not exist.")

            result = (
                "Incorrect Tool Name. "
                "Please retry and select a tool "
                "from the available tools."
            )

        else:
            # Execute the tool
            result = tools_dict[tool_name].invoke(query)

            print(f"Result length: {len(str(result))}")

        # Convert tool result into ToolMessage
        results.append(
            ToolMessage(
                tool_call_id=tool_call["id"], name=tool_name, content=str(result)
            )
        )

    print("Tools Execution Complete. Back to the model!")

    return {"messages": results}


# -------------------------
# 9. Build LangGraph
# -------------------------

graph = StateGraph(AgentState)


# Add nodes
graph.add_node("llm", call_llm)

graph.add_node("retriever_agent", take_action)


# LLM decides whether to use a tool
graph.add_conditional_edges(
    "llm", should_continue, {True: "retriever_agent", False: END}
)


# After retrieving information,
# go back to the LLM
graph.add_edge("retriever_agent", "llm")


# Start with LLM
graph.set_entry_point("llm")


# Compile graph
rag_agent = graph.compile()
