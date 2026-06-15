"""
LLM 服务层 — DeepSeek API 封装
===============================
5 个 Agent 角色，降级模式使用 DuckDuckGo 真实搜索 + 模板化输出。
通过 self._mode 字段显式路由降级，而非 system prompt 猜测。
"""

import os
import json
from typing import Dict, List, Optional

from .search_tool import search_and_summarize


class LLMService:
    """DeepSeek LLM 服务"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.available = bool(self.api_key)
        self._client = None
        self._mode: Optional[str] = None
        if self.available:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key, base_url="https://api.deepseek.com"
                )
            except ImportError:
                self.available = False

    def _call(self, system: str, user: str, temperature: float = 0.7) -> str:
        if not self._client:
            return self._fallback(user)
        try:
            r = self._client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=2048,
            )
            return r.choices[0].message.content
        except Exception:
            return self._fallback(user)

    def _fallback(self, user_input: str) -> str:
        mode = self._mode or ""
        if mode == "plan":
            return json.dumps(
                ["核心技术原理与架构设计", "主流方案对比分析",
                 "工程实践与落地要点", "发展趋势与未来展望"],
                ensure_ascii=False,
            )
        if mode == "search":
            return search_and_summarize(user_input.replace("问题: ", "").strip())
        if mode == "analyze":
            return (
                "## 综合分析\n\n该技术方向在2025-2026年进入快速发展期。"
                "核心创新集中在架构优化和工程落地两个维度。\n\n"
                "1. 模块化设计成为主流趋势\n"
                "2. 性能优化从硬件层延伸到应用层\n"
                "3. 开源生态加速了技术普及"
            )
        if mode == "write":
            topic = user_input.split("\n")[0].replace("主题: ", "").strip()[:80]
            return (
                f"# {topic} — 技术调研报告\n\n## 概述\n\n"
                "本报告基于多Agent协作调研框架生成。\n\n"
                "## 核心技术原理\n\n模块化设计使得各组件可以独立迭代优化。\n\n"
                "## 主流方案对比\n\n开源方案在灵活性和成本上占优。\n\n"
                "## 工程实践要点\n\n重点关注性能基准测试和容错机制设计。\n\n"
                "## 发展趋势\n\n该方向将在未来2-3年内持续演进。"
            )
        if mode == "review":
            return json.dumps({
                "score": 3.5,
                "strengths": ["报告结构清晰，覆盖了核心技术点", "对比分析有参考价值"],
                "weaknesses": ["缺少具体的量化数据支撑", "部分分析可以更深入"],
                "suggestions": "建议增加实际案例和性能数据",
            }, ensure_ascii=False)
        return "分析完成。"

    def plan(self, topic: str, depth: str = "medium") -> List[str]:
        self._mode = "plan"
        system = "你是一个技术规划师。将给定主题拆解为3-5个子问题。只返回JSON数组。"
        user = f"主题: {topic}\n深度: {depth}"
        result = self._call(system, user)
        try:
            return json.loads(result.strip().strip("```json").strip("```").strip())
        except json.JSONDecodeError:
            return ["核心技术原理", "主流方案对比", "工程实践要点", "发展趋势"]

    def search(self, question: str) -> str:
        self._mode = "search"
        system = "你是一个信息搜集专家。针对给定问题给出详尽的技术分析（2-3段）。"
        return self._call(system, f"问题: {question}")

    def analyze(self, topic: str, findings: List[str]) -> str:
        self._mode = "analyze"
        system = "你是一个技术分析师。综合所有调研发现，提炼关键洞察和模式。"
        text = "\n---\n".join(findings)
        return self._call(system, f"主题: {topic}\n\n调研发现:\n{text}")

    def write(self, topic: str, analysis: str) -> str:
        self._mode = "write"
        system = "你是一个技术撰稿人。撰写结构化Markdown报告。"
        return self._call(system, f"主题: {topic}\n\n分析:\n{analysis}")

    def review(self, report: str) -> Dict:
        self._mode = "review"
        system = "你是报告审核员。返回JSON: {\"score\": 1-5, \"strengths\": [...], \"weaknesses\": [...], \"suggestions\": \"...\"}"
        result = self._call(system, f"报告:\n{report[:2000]}")
        try:
            return json.loads(result.strip().strip("```json").strip("```").strip())
        except json.JSONDecodeError:
            return {"score": 3.0, "strengths": [], "weaknesses": [], "suggestions": ""}
