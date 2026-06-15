"""
LangGraph 状态机 — Multi-Agent Research 编排
=============================================
5 节点 DAG: plan -> search -> analyze -> write -> review -> END

使用 LangGraph 编译后的 graph.invoke() 自动调度。
"""

from typing import TypedDict, List, Dict


class ResearchState(TypedDict):
    topic: str
    depth: str
    plan: List[str]
    findings: List[str]
    analysis: str
    report: str
    review: Dict
    phase: str
    current_index: int
    error: str


def make_initial_state(topic: str, depth: str = "medium") -> ResearchState:
    return ResearchState(
        topic=topic, depth=depth, plan=[], findings=[], analysis="",
        report="", review={}, phase="init", current_index=0, error="",
    )


class ResearchGraph:
    """Multi-Agent Research 编排器"""

    def __init__(self, llm):
        self.llm = llm
        self._compiled = None

    @property
    def graph(self):
        if self._compiled is None:
            self._compiled = self._build()
        return self._compiled

    def _build(self):
        from langgraph.graph import StateGraph, END
        workflow = StateGraph(ResearchState)
        workflow.add_node("plan", self._plan)
        workflow.add_node("search", self._search)
        workflow.add_node("analyze", self._analyze)
        workflow.add_node("write", self._write)
        workflow.add_node("review", self._review)
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "search")
        workflow.add_conditional_edges(
            "search", self._route_after_search,
            {"search": "search", "analyze": "analyze"},
        )
        workflow.add_edge("analyze", "write")
        workflow.add_edge("write", "review")
        workflow.add_edge("review", END)
        return workflow.compile()

    def _plan(self, state: ResearchState) -> ResearchState:
        try:
            state["phase"] = "planning"
            state["plan"] = self.llm.plan(state["topic"], state.get("depth", "medium"))
            state["current_index"] = 0
        except Exception as e:
            state["error"] = f"Plan node failed: {e}"
        return state

    def _search(self, state: ResearchState) -> ResearchState:
        if state.get("error"):
            return state
        try:
            state["phase"] = "searching"
            plan = state.get("plan", [])
            idx = state.get("current_index", 0)
            if idx < len(plan):
                question = plan[idx]
                result = self.llm.search(question)
                state["findings"].append(f"Q: {question}\nA: {result}")
                state["current_index"] = idx + 1
        except Exception as e:
            state["error"] = f"Search node failed: {e}"
        return state

    def _analyze(self, state: ResearchState) -> ResearchState:
        if state.get("error"):
            return state
        try:
            state["phase"] = "analyzing"
            state["analysis"] = self.llm.analyze(state["topic"], state.get("findings", []))
        except Exception as e:
            state["error"] = f"Analyze node failed: {e}"
        return state

    def _write(self, state: ResearchState) -> ResearchState:
        if state.get("error"):
            return state
        try:
            state["phase"] = "writing"
            state["report"] = self.llm.write(state["topic"], state["analysis"])
        except Exception as e:
            state["error"] = f"Write node failed: {e}"
        return state

    def _review(self, state: ResearchState) -> ResearchState:
        if state.get("error"):
            return state
        try:
            state["phase"] = "reviewing"
            state["review"] = self.llm.review(state["report"])
            state["phase"] = "done"
        except Exception as e:
            state["error"] = f"Review node failed: {e}"
        return state

    def _route_after_search(self, state: ResearchState) -> str:
        if state.get("error"):
            return "analyze"
        idx = state.get("current_index", 0)
        plan = state.get("plan", [])
        if idx < len(plan):
            return "search"
        return "analyze"

    def run(self, topic: str, depth: str = "medium") -> ResearchState:
        state = make_initial_state(topic, depth)
        return self.graph.invoke(state)
