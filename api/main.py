"""
Multi-Agent Research System — API
===================================
POST /research → 4 Agent 协作产出调研报告
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.llm import LLMService
from core.graph import ResearchGraph


app = FastAPI(
    title="Multi-Agent Research System",
    description="Planner → Search → Analyze → Write → Review 协作调研",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

llm = LLMService()


class ResearchRequest(BaseModel):
    topic: str
    depth: str = Field(default="medium", pattern="^(basic|medium|deep)$")


class ResearchResponse(BaseModel):
    topic: str
    report: str
    score: float
    plan: list[str]
    strengths: list[str]
    weaknesses: list[str]
    mode: str


@app.get("/")
def root():
    return {"service": "Multi-Agent Research System", "version": "1.0.0", "mode": "llm" if llm.available else "fallback"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    graph = ResearchGraph(llm)
    state = graph.run(req.topic, req.depth)

    if state.get("error"):
        raise HTTPException(500, state["error"])

    review = state.get("review", {})
    return ResearchResponse(
        topic=req.topic,
        report=state.get("report", ""),
        score=review.get("score", 0),
        plan=state.get("plan", []),
        strengths=review.get("strengths", []),
        weaknesses=review.get("weaknesses", []),
        mode="llm" if llm.available else "fallback",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
