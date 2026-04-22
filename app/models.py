from typing import TypedDict, List
from pydantic import BaseModel


class GraphState(TypedDict):
    user_id: str
    query: str
    refined_query: str
    context: str
    draft: str
    review: str
    score: int
    iteration: int
    memory: List[str]


class QueryRequest(BaseModel):
    user_id: str
    query: str


class QueryResponse(BaseModel):
    answer: str
    score: int
    iterations: int


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
