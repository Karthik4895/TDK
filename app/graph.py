import re
import logging
from langgraph.graph import StateGraph
from langchain_anthropic import ChatAnthropic
from app.models import GraphState
from app.rag import retrieve
from app.memory import get_memory
from app.config import MAX_ITERATIONS, SCORE_THRESHOLD, CLAUDE_MODEL

logger = logging.getLogger("tdk.graph")

llm = ChatAnthropic(model=CLAUDE_MODEL, temperature=0.7, max_tokens=1024)


def load_memory_node(state: GraphState) -> GraphState:
    state["memory"] = get_memory(state["user_id"])
    logger.debug("Loaded %d memory entries for %s", len(state["memory"]), state["user_id"])
    return state


def refine_query_node(state: GraphState) -> GraphState:
    prompt = (
        "You are an expert at understanding customer queries for a mehendi (henna) service. "
        "Rewrite the query below to be more specific and searchable while preserving intent. "
        "Return only the refined query, nothing else.\n\n"
        f"Query: {state['query']}"
    )
    result = llm.invoke(prompt)
    state["refined_query"] = result.content.strip()
    logger.debug("Refined: %s", state["refined_query"])
    return state


def retrieve_node(state: GraphState) -> GraphState:
    state["context"] = retrieve(state["refined_query"])
    return state


def write_node(state: GraphState) -> GraphState:
    memory_text = "\n".join(state["memory"]) if state["memory"] else "No prior preferences on record."
    prompt = f"""You are a warm, knowledgeable assistant for TDK Mehendi, a professional henna service in Chennai.

Customer history:
{memory_text}

Relevant information:
{state['context']}

Answer the following customer query in a helpful, friendly, and professional manner. Include specific details such as pricing, timing, or design tips where relevant.

Query: {state['query']}"""
    result = llm.invoke(prompt)
    state["draft"] = result.content
    return state


def review_node(state: GraphState) -> GraphState:
    prompt = f"""You are a quality reviewer for a customer service response at a mehendi business.

Evaluate the response below on:
1. Accuracy and helpfulness
2. Warmth and professional tone
3. Completeness — does it fully answer the query?

Response:
{state['draft']}

Give a score from 0–10 (10 = perfect) and brief improvement notes.
Format your reply as: Score: X/10 — [feedback]"""
    result = llm.invoke(prompt)
    state["review"] = result.content
    match = re.search(r"\b(10|[0-9])\b", result.content)
    state["score"] = int(match.group()) if match else 5
    logger.debug("Review score: %d", state["score"])
    return state


def improve_node(state: GraphState) -> GraphState:
    prompt = f"""Improve the customer service response below based on the reviewer feedback. Keep it warm and professional.

Reviewer feedback:
{state['review']}

Current response:
{state['draft']}

Return only the improved response."""
    result = llm.invoke(prompt)
    state["draft"] = result.content
    state["iteration"] += 1
    logger.debug("Improvement iteration %d", state["iteration"])
    return state


def should_continue(state: GraphState) -> str:
    if state["score"] >= SCORE_THRESHOLD or state["iteration"] >= MAX_ITERATIONS:
        return "end"
    return "improve"


builder = StateGraph(GraphState)
builder.add_node("memory", load_memory_node)
builder.add_node("refine", refine_query_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("write", write_node)
builder.add_node("review", review_node)
builder.add_node("improve", improve_node)

builder.set_entry_point("memory")
builder.add_edge("memory", "refine")
builder.add_edge("refine", "retrieve")
builder.add_edge("retrieve", "write")
builder.add_edge("write", "review")
builder.add_conditional_edges("review", should_continue, {"end": "__end__", "improve": "improve"})
builder.add_edge("improve", "review")

graph = builder.compile()
