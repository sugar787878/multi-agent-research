"""
真实搜索工具
============
DuckDuckGo 免费搜索 + 结果摘要提取。无需 API key。
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

_FALLBACK_TEMPLATES = [
    "该技术方向近年来发展迅速。从学术角度，NeurIPS/ICML/ACL等顶会均有大量投稿。从工程角度，头部公司已在大规模部署相关系统。",
    "核心技术栈包括深度学习框架(PyTorch/TensorFlow)、分布式训练(DeepSpeed/FSDP)、模型服务和MLOps工具链。Docker+Kubernetes是部署标准。",
    "性能优化方面，量化(INT8/INT4)、蒸馏、Flash Attention等技术可将推理延迟降低50-90%。KV-Cache和Continuous Batching是提升吞吐量的关键。",
    "2025-2026年前沿方向：多模态融合、Agent自主决策、长上下文处理(100K+ tokens)、端侧部署。开源模型能力快速缩小与闭源的差距。",
]


def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """DuckDuckGo 网页搜索，返回 [{title, url, snippet}]。"""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
            return results
    except ImportError:
        logger.info("duckduckgo-search not installed, using fallback")
        return []
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []


def search_and_summarize(query: str, max_results: int = 3) -> str:
    """搜索并拼接为文本摘要。有网络时返回真实搜索结果，否则用模板。"""
    results = web_search(query, max_results=max_results)
    if results:
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[{i}] {r['title']}\n    URL: {r['url']}\n    {r['snippet']}")
        return "\n\n".join(parts)
    idx = hash(query) % len(_FALLBACK_TEMPLATES)
    return f"[离线模式]\n\n{_FALLBACK_TEMPLATES[idx]}"
