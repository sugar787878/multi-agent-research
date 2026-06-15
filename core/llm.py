"""
LLM 服务层 — DeepSeek API 封装
===============================
4 个 Agent 角色对应不同的 system prompt，统一通过 DeepSeek 调用。
"""

import os
import json
from typing import Dict, List, Optional


class LLMService:
    """DeepSeek LLM 服务"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.available = bool(self.api_key)
        self._client = None
        if self.available:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def _call(self, system: str, user: str, temperature: float = 0.7) -> str:
        if not self._client:
            return self._fallback(system, user)
        try:
            r = self._client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=temperature, max_tokens=1024,
            )
            return r.choices[0].message.content
        except Exception:
            return self._fallback(system, user)

    def _fallback(self, system: str, user: str) -> str:
        """降级: 返回模板化输出"""
        if "规划" in system or "planner" in system.lower():
            return '["核心技术原理与架构设计", "主流方案对比分析", "工程实践与落地要点", "发展趋势与未来展望"]'
        if "搜集" in system or "search" in system.lower():
            topic = user.replace("问题: ", "").strip()[:80]
            return (
                f"关于「{topic}」的初步分析：该方向涉及多个关键技术组件。"
                "从学术研究角度看，近两年相关论文数量增长显著。"
                "从工程实践角度看，头部公司已在大规模部署相关系统。"
                "建议重点关注架构设计和性能优化两个子方向。"
            )
        if "分析" in system or "analyst" in system.lower():
            return (
                "## 综合分析\n\n"
                "该技术方向在2025-2026年进入快速发展期。"
                "核心创新集中在架构优化和工程落地两个维度。"
                "从各子问题的调研结果来看，存在以下关键模式：\n\n"
                "1. 模块化设计成为主流趋势\n"
                "2. 性能优化从硬件层延伸到应用层\n"
                "3. 开源生态加速了技术普及\n\n"
                "建议后续研究聚焦于实际落地场景的验证。"
            )
        if "撰稿" in system or "writer" in system.lower():
            topic = user.split("\n")[0].replace("主题: ", "").strip()[:80]
            return (
                f"# {topic} — 技术调研报告\n\n"
                "## 概述\n\n"
                "本报告基于多Agent协作调研框架生成，"
                "通过Planner→Search→Analyze→Write→Review五个阶段系统梳理该技术方向。\n\n"
                "## 核心技术原理\n\n"
                "该技术建立在坚实的理论基础之上，核心组件包括数据管道、"
                "模型服务和监控体系。模块化设计使得各组件可以独立迭代优化。\n\n"
                "## 主流方案对比\n\n"
                "目前业界存在多种实现方案，各有优劣。"
                "开源方案在灵活性和成本上占优，商业方案在稳定性和支持上更可靠。\n\n"
                "## 工程实践要点\n\n"
                "落地过程中需要重点关注：性能基准测试、容错机制设计、"
                "监控告警体系建设和持续迭代流程。\n\n"
                "## 发展趋势\n\n"
                "该方向将在未来2-3年内持续演进，建议保持对前沿动态的关注。"
            )
        if "审核" in system or "reviewer" in system.lower():
            return (
                '{"score": 3.5, '
                '"strengths": ["报告结构清晰，覆盖了核心技术点", "对比分析有参考价值"], '
                '"weaknesses": ["缺少具体的量化数据支撑", "部分分析可以更深入"], '
                '"suggestions": "建议增加实际案例和性能数据，补充竞品对比细节"}'
            )
        return "分析完成。"

    # ---- 5 个 Agent 角色 ----

    def plan(self, topic: str, depth: str = "medium") -> List[str]:
        system = "你是一个技术规划师。将给定主题拆解为 3-5 个子问题用于调研。只返回 JSON 数组，不要其他文字。"
        user = f"主题: {topic}\n深度: {depth}\n拆解为子问题 JSON 数组:"
        result = self._call(system, user)
        try:
            return json.loads(result.strip().strip("```json").strip("```").strip())
        except json.JSONDecodeError:
            return ["核心技术原理", "主流方案对比", "工程实践要点", "发展趋势"]

    def search(self, question: str) -> str:
        system = "你是一个信息搜集专家。针对给定问题，给出详尽的技术分析（2-3段）。"
        return self._call(system, f"问题: {question}")

    def analyze(self, topic: str, findings: List[str]) -> str:
        system = "你是一个技术分析师。综合所有调研发现，提炼关键洞察和模式（3-4段）。"
        text = "\n---\n".join(findings)
        return self._call(system, f"主题: {topic}\n\n调研发现:\n{text}")

    def write(self, topic: str, analysis: str) -> str:
        system = "你是一个技术撰稿人。根据分析撰写结构化 Markdown 报告，包含概述、详细分析、对比、建议。"
        return self._call(system, f"主题: {topic}\n\n分析:\n{analysis}")

    def review(self, report: str) -> Dict:
        system = "你是报告审核员。评估报告质量，返回 JSON: {\"score\": 1-5, \"strengths\": [...], \"weaknesses\": [...], \"suggestions\": \"...\"}"
        result = self._call(system, f"报告:\n{report[:2000]}")
        try:
            return json.loads(result.strip().strip("```json").strip("```").strip())
        except json.JSONDecodeError:
            return {"score": 3.0, "strengths": [], "weaknesses": [], "suggestions": ""}
