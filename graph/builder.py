from langgraph.graph import StateGraph, END
from graph.state import GraphState
from graph.nodes import (intent_node,general_chat_node,document_qa_node,)


def build_graph():
    graph = StateGraph(GraphState)

    # Nodes
    graph.add_node("intent", intent_node)
    graph.add_node("general_chat", general_chat_node)
    graph.add_node("document_qa", document_qa_node)

    # Entry point
    graph.set_entry_point("intent")

    # Routing logic
    def route_by_intent(state: GraphState):
        if state["intent"] == "DOCUMENT_QA":        
            return "document_qa"
        return "general_chat"

    graph.add_conditional_edges(
        "intent",
        route_by_intent,
        {
            "general_chat": "general_chat",
            "document_qa": "document_qa",
        }
    )

    # End states
    graph.add_edge("general_chat", END)
    graph.add_edge("document_qa", END)

    return graph.compile()
