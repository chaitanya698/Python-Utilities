import click
from chatbot_graph import build_graph
from chatbot_state import ChatState
from chatbot_nodes import greeting_run

graph = build_graph()


@click.command()
def chat():
    
    state: ChatState = {"messages": []}

    
    state.update(greeting_run(state))
    print(f"Bot: {state['assistant_response']}")

    while not state.get("done"):
        user = input("You: ")
        state["user_input"] = user
        state = graph.invoke(state)       
        print(f"Bot: {state['assistant_response']}")


if __name__ == "__main__":
    chat()
