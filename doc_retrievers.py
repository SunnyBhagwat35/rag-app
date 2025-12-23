from doc_loader import vector_store, model
from langchain.tools import tool
from langchain.agents import create_agent


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

if __name__ == "__main__":
    tools = [retrieve_context]
    prompt = (
        "You have access to a tool that retrieves context from a blog post. "
        "Use the tool to help answer user queries."
    )
    print("Creating Agent")
    agent = create_agent(model, tools, system_prompt=prompt)
    print("Agent Created.")
    # query = (
    #     "What is the standard method for Task Decomposition?\n\n"
    #     "Once you get the answer, look up common extensions of that method."
    # )
    
    query = ("What is task decomposition")

    print("Now agent streaming starts..")
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


