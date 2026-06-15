"""
Multi-Agent Research — TDD 测试
=================================
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_llm():
    from core.llm import LLMService
    llm = Mock(spec=LLMService)
    llm.available = False  # 降级模式
    llm.plan.return_value = ["核心技术原理", "主流方案对比", "工程实践要点"]
    llm.search.return_value = "这是一个快速发展的技术领域，涉及多个关键组件。"
    llm.analyze.return_value = "综合分析：该技术方向核心创新集中在架构优化和工程落地。"
    llm.write.return_value = (
        "# 技术调研报告\n\n## 概述\n\n这是一个重要的技术方向，涉及多个核心组件。\n\n"
        "## 核心技术原理\n\n基于坚实的理论基础，系统采用模块化设计。\n\n"
        "## 主流方案对比\n\n开源方案灵活，商业方案稳定。\n\n## 工程实践\n\n需要关注性能测试和监控。"
    )
    llm.review.return_value = {"score": 3.5, "strengths": ["结构清晰"], "weaknesses": ["缺少数据"], "suggestions": "增加量化分析"}
    return llm


@pytest.fixture
def graph(mock_llm):
    from core.graph import ResearchGraph
    return ResearchGraph(llm=mock_llm)


class TestLLMService:
    def test_fallback_plan_returns_list(self):
        from core.llm import LLMService
        llm = LLMService(api_key=None)
        assert llm.available == False
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


class TestResearchGraph:
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

    def test_full_run(self, graph):
        state = graph.run("RAG系统设计")
        assert state["phase"] == "done"
        assert len(state["plan"]) == 3
        assert len(state["findings"]) == 3
        assert len(state["report"]) > 100
        assert state["review"]["score"] > 0


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
        r = client.post("/research", json={"topic": "Transformer架构", "depth": "medium"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["report"]) > 50
        assert len(data["plan"]) >= 3
        assert "score" in data
