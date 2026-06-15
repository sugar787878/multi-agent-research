# Multi-Agent Research System

基于 **LangGraph** 的多 Agent 协作技术调研系统。

## 流程

```
POST /research {"topic": "RAG系统设计"}
  → Planner: 拆解为子问题
  → Search Agent: 逐个搜索分析
  → Analysis Agent: 综合提炼洞察
  → Writer Agent: 撰写结构化报告
  → Reviewer Agent: 评分+改进建议
  → 返回: 报告 + 质量评分
```

## 快速启动

```bash
docker compose up -d
# → http://localhost:8001
```

```bash
curl -X POST http://localhost:8001/research \
  -H 'Content-Type: application/json' \
  -d '{"topic":"RAG系统设计","depth":"medium"}'
```

## 技术栈

LangGraph · DeepSeek · FastAPI · Docker · TDD
