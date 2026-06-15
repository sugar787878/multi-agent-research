# Multi-Agent Research System

基于 **LangGraph** 的多 Agent 协作技术调研系统。

## 流程

```
POST /research {"topic": "RAG系统设计"}
  -> Planner Agent: 拆解为子问题
  -> Search Agent: 逐个搜索分析 (DuckDuckGo 真实搜索)
  -> Analysis Agent: 综合提炼洞察
  -> Writer Agent: 撰写结构化报告
  -> Reviewer Agent: 评分+改进建议
  -> 返回: 报告 + 质量评分 + 强项/弱项
```

## v1.1.0 更新 (2026-06-15)

- **LangGraph 真正驱动**: `graph.invoke()` 替代手动 while 循环
- **真实搜索**: DuckDuckGo 免费 API，降级模式下也能返回有意义的搜索结果
- **错误传播**: 各节点 try/except，错误写入 state.error 并短路
- **显式降级路由**: `self._mode` 字段替代 system prompt 字符串猜测
- **LLM 依赖注入**: 每个请求独立创建 LLMService，支持 API key 热加载

## 快速启动

```bash
# Docker
docker compose up -d   # -> http://localhost:8001

# 本地
pip install -r requirements.txt
python -m api.main
```

## API

```bash
curl -X POST http://localhost:8001/research \
  -H 'Content-Type: application/json' \
  -d '{"topic":"RAG系统设计","depth":"medium"}'
```

## 环境变量

| 变量 | 说明 | 默认 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API key | 未设置->降级模式(DuckDuckGo+模板) |

## 技术栈

LangGraph · DeepSeek API · DuckDuckGo Search · FastAPI · Docker · TDD (26 tests)
