from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.synthesizer import SynthesizerAgent

# 1. Define State
class State(TypedDict):
    query: str
    query_type: str
    plan_description: str
    retrieved_chunks: List[dict]
    final_answer: str

# 2. Initialize Agents
planner = PlannerAgent()
retriever = RetrieverAgent()
synthesizer = SynthesizerAgent()

# 3. Define Nodes
def planner_node(state: State):
    return planner.plan(state)

def retriever_node(state: State):
    return retriever.retrieve(state)

def synthesizer_node(state: State):
    return synthesizer.synthesize(state)

# 4. Build Graph
workflow = StateGraph(State)

# Add Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("synthesizer", synthesizer_node)

# Add Edges (Linear)
workflow.set_entry_point("planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "synthesizer")
workflow.add_edge("synthesizer", END)

# 5. Compile
app = workflow.compile()
