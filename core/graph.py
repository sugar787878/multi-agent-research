"""
LangGraph 状态机 — Multi-Agent Research 编排
=============================================
5 节点 DAG: plan → search → analyze → write → review → END
"""

from typing import TypedDict, List, Dict, Optional


class ResearchState(TypedDict):
    topic: str
    depth: str
    plan: List[str]              # 子问题列表
    findings: List[str]          # 每个子问题的搜索结果
    analysis: str                # 综合分析
    report: str                  # 最终报告
    review: Dict                 # 审核结果 {score, strengths, weaknesses, suggestions}
    phase: str                   # init/planning/searching/analyzing/writing/reviewing/done
    current_index: int           # 当前处理的子问题索引
    error: str                   # 错误信息


def make_initial_state(topic: str, depth: str = "medium") -> ResearchState:
    return ResearchState(
        topic=topic,
        depth=depth,
        plan=[],
        findings=[],
        analysis="",
        report="",
        review={},
        phase="init",
        current_index=0,
        error="",
    )


class ResearchGraph:
    """Multi-Agent Research 编排器"""

    def __init__(self, llm):
        self.llm = llm

    # ---- 节点 ----

    def _plan(self, state: ResearchState) -> ResearchState:
        state["phase"] = "planning"
        state["plan"] = self.llm.plan(state["topic"], state.get("depth", "medium"))
        state["current_index"] = 0
        return state

    def _search(self, state: ResearchState) -> ResearchState:
        state["phase"] = "searching"
        plan = state.get("plan", [])
        idx = state.get("current_index", 0)

        if idx < len(plan):
            question = plan[idx]
            result = self.llm.search(question)
            state["findings"].append(f"Q: {question}\nA: {result}")
            state["current_index"] = idx + 1

        return state

    def _analyze(self, state: ResearchState) -> ResearchState:
        state["phase"] = "analyzing"
        state["analysis"] = self.llm.analyze(
            state["topic"], state.get("findings", [])
        )
        return state

    def _write(self, state: ResearchState) -> ResearchState:
        state["phase"] = "writing"
        state["report"] = self.llm.write(state["topic"], state["analysis"])
        return state

    def _review(self, state: ResearchState) -> ResearchState:
        state["phase"] = "reviewing"
        state["review"] = self.llm.review(state["report"])
        state["phase"] = "done"
        return state

    # ---- 条件路由 ----

    def _route_after_search(self, state: ResearchState) -> str:
        idx = state.get("current_index", 0)
        plan = state.get("plan", [])
        if idx < len(plan):
            return "search"       # 继续下一个子问题
        return "analyze"          # 全部搜索完

    # ---- 构建图 ----

    def build(self):
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
            {"search": "search", "analyze": "analyze"}
        )
        workflow.add_edge("analyze", "write")
        workflow.add_edge("write", "review")
        workflow.add_edge("review", END)

        return workflow.compile()

    # ---- 公开接口 ----

    def run(self, topic: str, depth: str = "medium") -> ResearchState:
        graph = self.build()
        state = make_initial_state(topic, depth)
        # 逐步执行直到结束
        while state["phase"] != "done":
            if state["phase"] == "init":
                state = self._plan(state)
            elif state["phase"] == "planning":
                state = self._search(state)
            elif state["phase"] == "searching":
                route = self._route_after_search(state)
                if route == "search":
                    state = self._search(state)
                else:
                    state = self._analyze(state)
            elif state["phase"] == "analyzing":
                state = self._write(state)
            elif state["phase"] == "writing":
                state = self._review(state)
            elif state["phase"] == "reviewing":
                state["phase"] = "done"
        return state
