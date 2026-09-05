from langchain_core.messages import HumanMessage
from rag_agent import rag_agent


def running_agent():

    print("\n=== RAG AGENT ===")

    while True:
        user_input = input("\nWhat is your question: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        # Convert user input into a HumanMessage
        messages = [HumanMessage(content=user_input)]

        # Run the LangGraph agent
        result = rag_agent.invoke({"messages": messages})

        # Print final answer
        print("\n=== ANSWER ===")

        print(result["messages"][-1].content)


if __name__ == "__main__":
    running_agent()
