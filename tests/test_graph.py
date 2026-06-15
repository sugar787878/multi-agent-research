"""
Multi-Agent Research — TDD 测试
=================================
覆盖: LLM降级 · Graph节点/路由 · 错误传播 · 全流程 · API · 搜索工具
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_llm():
    from core.llm import LLMService
    llm = Mock(spec=LLMService)
    llm.available = False
    llm.plan.return_value = ["核心技术原理", "主流方案对比", "工程实践要点"]
    llm.search.return_value = "这是一个快速发展的技术领域，涉及多个关键组件。"
    llm.analyze.return_value = "综合分析：该技术方向核心创新集中在架构优化和工程落地。"
    llm.write.return_value = (
        "# 技术调研报告\n\n## 概述\n\n这是一个重要的技术方向。\n\n"
        "## 核心技术原理\n\n基于坚实的理论基础。\n\n"
        "## 主流方案对比\n\n开源方案灵活，商业方案稳定。"
    )
    llm.review.return_value = {
        "score": 3.5,
        "strengths": ["结构清晰"],
        "weaknesses": ["缺少数据"],
        "suggestions": "增加量化分析",
    }
    return llm


@pytest.fixture
def graph(mock_llm):
    from core.graph import ResearchGraph
    return ResearchGraph(llm=mock_llm)


class TestLLMService:
    def test_init_without_key_sets_unavailable(self):
        from core.llm import LLMService
        with patch.dict("os.environ", {}, clear=True):
            llm = LLMService(api_key=None)
            assert llm.available is False

    def test_fallback_plan_returns_list(self):
        from core.llm import LLMService
        llm = LLMService(api_key=None)
        result = llm.plan("RAG系统设计")
        assert isinstance(result, list)
        assert len(result) >= 3

    def test_fallback_search_returns_string(self):
        from core.llm import LLMService
        llm = LLMService(api_key=None)
        result = llm.search("什么是RAG")
        assert isinstance(result, str)
        assert len(result) > 10

    def test_fallback_review_returns_dict(self):
        from core.llm import LLMService
        llm = LLMService(api_key=None)
        result = llm.review("# Test Report")
        assert isinstance(result, dict)
        assert "score" in result

    def test_fallback_uses_explicit_mode(self):
        from core.llm import LLMService
        llm = LLMService(api_key=None)
        llm.search("test query")
        assert llm._mode == "search"
        llm.plan("test topic")
        assert llm._mode == "plan"


class TestResearchGraphNodes:
    def test_plan_node(self, graph):
        from core.graph import make_initial_state
        state = make_initial_state("RAG系统")
        state = graph._plan(state)
        assert len(state["plan"]) == 3
        assert state["phase"] == "planning"

    def test_search_node(self, graph):
        from core.graph import make_initial_state
        state = make_initial_state("RAG系统")
        state["plan"] = ["Q1", "Q2"]
        state["phase"] = "planning"
        state = graph._search(state)
        assert len(state["findings"]) == 1
        assert state["current_index"] == 1

    def test_search_loop(self, graph):
        from core.graph import make_initial_state
        state = make_initial_state("RAG系统")
        state["plan"] = ["Q1", "Q2"]
        state["phase"] = "planning"
        state = graph._search(state)
        state = graph._search(state)
        assert len(state["findings"]) == 2
        assert state["current_index"] == 2

    def test_analyze_node(self, graph):
        from core.graph import make_initial_state
        state = make_initial_state("RAG系统")
        state["findings"] = ["finding1", "finding2"]
        state = graph._analyze(state)
        assert len(state["analysis"]) > 0
        assert state["phase"] == "analyzing"

    def test_write_node(self, graph):
        from core.graph import make_initial_state
        state = make_initial_state("RAG系统")
        state["analysis"] = "test analysis"
        state = graph._write(state)
        assert len(state["report"]) > 0
        assert state["phase"] == "writing"

    def test_review_node(self, graph):
        from core.graph import make_initial_state
        state = make_initial_state("RAG系统")
        state["report"] = "# Test Report"
        state = graph._review(state)
        assert state["phase"] == "done"
        assert state["review"]["score"] == 3.5


class TestResearchGraphRouting:
    @pytest.fixture
    def graph(self, mock_llm):
        from core.graph import ResearchGraph
        return ResearchGraph(llm=mock_llm)

    def test_route_continue_search(self, graph):
        from core.graph import make_initial_state
        state = make_initial_state("RAG系统")
        state["plan"] = ["Q1", "Q2"]
        state["current_index"] = 1
        assert graph._route_after_search(state) == "search"

    def test_route_to_analyze(self, graph):
        from core.graph import make_initial_state
        state = make_initial_state("RAG系统")
        state["plan"] = ["Q1", "Q2"]
        state["current_index"] = 2
        assert graph._route_after_search(state) == "analyze"

    def test_route_empty_plan(self, graph):
        from core.graph import make_initial_state
        state = make_initial_state("RAG系统")
        state["plan"] = []
        state["current_index"] = 0
        assert graph._route_after_search(state) == "analyze"


class TestErrorPropagation:
    def test_plan_error_sets_error_field(self):
        from core.graph import ResearchGraph, make_initial_state
        from core.llm import LLMService
        llm = Mock(spec=LLMService)
        llm.plan.side_effect = RuntimeError("Boom in planner")
        g = ResearchGraph(llm=llm)
        state = make_initial_state("test")
        state = g._plan(state)
        assert "Boom in planner" in state["error"]

    def test_search_skipped_when_error_present(self):
        from core.graph import ResearchGraph, make_initial_state
        from core.llm import LLMService
        llm = Mock(spec=LLMService)
        g = ResearchGraph(llm=llm)
        state = make_initial_state("test")
        state["error"] = "prior error"
        state["plan"] = ["Q1"]
        state = g._search(state)
        assert len(state["findings"]) == 0
        assert state["error"] == "prior error"

    def test_error_routes_to_analyze(self, graph):
        from core.graph import make_initial_state
        state = make_initial_state("test")
        state["error"] = "something went wrong"
        assert graph._route_after_search(state) == "analyze"


class TestFullPipeline:
    def test_full_run_via_invoke(self, graph):
        state = graph.run("RAG系统设计")
        assert state["phase"] == "done"
        assert state["error"] == ""
        assert len(state["plan"]) == 3
        assert len(state["findings"]) == 3
        assert len(state["report"]) > 10
        assert state["review"]["score"] > 0

    def test_graph_is_compiled(self, graph):
        compiled = graph.graph
        assert compiled is not None
        assert hasattr(compiled, "invoke")

    def test_graph_compiled_once(self, graph):
        g1 = graph.graph
        g2 = graph.graph
        assert g1 is g2


class TestSearchTool:
    def test_search_and_summarize_returns_string(self):
        from core.search_tool import search_and_summarize
        result = search_and_summarize("Python programming language")
        assert isinstance(result, str)
        assert len(result) > 20

    def test_web_search_returns_list(self):
        from core.search_tool import web_search
        result = web_search("machine learning basics")
        assert isinstance(result, list)

    def test_fallback_templates(self):
        from core.search_tool import search_and_summarize
        with patch("core.search_tool.web_search", return_value=[]):
            r1 = search_and_summarize("深度学习")
            r2 = search_and_summarize("前端开发")
            assert len(r1) > 20
            assert len(r2) > 20


class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["service"] == "Multi-Agent Research System"

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_research(self, client):
        r = client.post("/research", json={
            "topic": "Transformer架构", "depth": "medium",
        })
        assert r.status_code == 200
        data = r.json()
        assert len(data["report"]) > 10
        assert len(data["plan"]) >= 3
        assert "score" in data
        assert data["mode"] in ("llm", "fallback")
