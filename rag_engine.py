#!/usr/bin/env python3
"""
天津日报 RAG 引擎 —— DeepSeek API 驱动的检索增强生成
- 基于 TF-IDF 检索相关文章
- 调用 DeepSeek API 生成自然语言回答
- 支持多轮对话
"""

import json
import os
import pickle
import re
import time
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from tjrb_tokenizer import jieba_tokenizer
from time_parser import parse_time_range

# ---------------------------------------------------------------------------
# 路径 & 默认值
# ---------------------------------------------------------------------------
KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tjrb_kb")

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")

# 检索 & 生成参数（针对 V4 Pro 1M 上下文优化）
DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "100"))
DEFAULT_MAX_CONTENT_LEN = int(os.environ.get("RAG_MAX_CONTENT_LEN", "0"))
DEFAULT_TEMPERATURE = float(os.environ.get("RAG_TEMPERATURE", "0.3"))
DEFAULT_MAX_TOKENS = int(os.environ.get("RAG_MAX_TOKENS", "8192"))

# ---------------------------------------------------------------------------
# 知识库加载（单例）
# ---------------------------------------------------------------------------
_kb_loaded = False
_vectorizer = None
_tfidf_matrix = None
_metadata: list = []


def load_kb() -> bool:
    """加载 TF-IDF 索引和元数据。返回 True 表示加载成功。"""
    global _kb_loaded, _vectorizer, _tfidf_matrix, _metadata

    if _kb_loaded:
        return True

    v_path = os.path.join(KB_DIR, "vectorizer.pkl")
    m_path = os.path.join(KB_DIR, "tfidf_matrix.npz")
    d_path = os.path.join(KB_DIR, "metadata.json")

    if not all(os.path.exists(p) for p in [v_path, m_path, d_path]):
        return False

    with open(v_path, "rb") as f:
        _vectorizer = pickle.load(f)
    _tfidf_matrix = load_npz(m_path)
    with open(d_path, "r", encoding="utf-8") as f:
        _metadata = json.load(f)

    _kb_loaded = True
    return True


def is_ready() -> bool:
    return _kb_loaded


def kb_stats() -> dict:
    """知识库统计信息"""
    if not _kb_loaded:
        return {"loaded": False}
    return {
        "loaded": True,
        "total_articles": len(_metadata),
        "date_range": f"{_metadata[0]['date']} ~ {_metadata[-1]['date']}",
    }


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------
def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    section_filter: Optional[str] = None,
) -> list[dict]:
    """
    TF-IDF 检索，返回 top_k 篇最相关文章。
    每篇文章附带相似度得分 score。
    """
    if not _kb_loaded:
        return []

    query_vec = _vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, _tfidf_matrix)[0]
    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked:
        if score < 0.01:
            continue
        art = dict(_metadata[idx])
        # 日期过滤
        if date_from and art["date"] < date_from:
            continue
        if date_to and art["date"] > date_to:
            continue
        # 版面过滤
        if section_filter and section_filter not in art["section"]:
            continue
        art["score"] = float(score)
        results.append(art)
        if len(results) >= top_k:
            break
    return results


# ---------------------------------------------------------------------------
# 上下文构建
# ---------------------------------------------------------------------------
def build_context(
    articles: list[dict],
    max_content_len: int = DEFAULT_MAX_CONTENT_LEN,
) -> str:
    """
    将检索到的文章拼接为 LLM 可用的上下文字符串。
    每篇文章包含：日期、版面、标题、正文摘要。
    """
    if not articles:
        return "（未找到相关文章）"

    parts = []
    for i, art in enumerate(articles):
        content = art.get("content", "")
        # 截断过长正文（max_content_len=0 表示不截断）
        if max_content_len > 0 and len(content) > max_content_len:
            content = content[:max_content_len] + "……"

        parts.append(
            f"【文章{i + 1}】\n"
            f"日期: {art['date']}\n"
            f"版面: {art['section']}\n"
            f"标题: {art['title']}\n"
            f"正文摘要:\n{content}\n"
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# DeepSeek 客户端
# ---------------------------------------------------------------------------
def _get_client() -> OpenAI:
    """创建 DeepSeek 客户端，强制绕过系统代理"""
    import httpx
    import os

    # 创建不读取环境变量代理设置的 transport
    transport = httpx.HTTPTransport(
        verify=True,
        retries=1,
    )
    # 用 mount 覆盖所有 URL 类型，避免触发环境变量代理解析
    http_client = httpx.Client(transport=transport)

    # 保险起见，直接操作环境变量移除 socks:// 这类不兼容的代理
    saved = {}
    for k in ("http_proxy", "https_proxy", "all_proxy",
              "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
              "no_proxy", "NO_PROXY"):
        v = os.environ.pop(k, None)
        if v is not None:
            saved[k] = v

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        http_client=http_client,
    )

    # 恢复代理环境变量
    for k, v in saved.items():
        os.environ[k] = v

    return client


SYSTEM_PROMPT = """你是一个专业的天津日报信息助手。你的任务是根据提供的天津日报文章内容，
回答用户的问题。请严格遵循以下规则：

1. 只依据提供的文章内容作答，不要编造文章中没有的信息。
2. 如果提供的文章不足以回答问题，请诚实告知用户，并建议调整搜索关键词或扩大日期范围。
3. 引用文章时，请注明日期和标题（格式：「YYYY-MM-DD《标题》」）。
4. 回答要客观、准确、有条理，使用中文。
5. 如果是撰写材料类问题（如梳理、提炼、总结），请组织成结构清晰的文稿形式。"""


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
def ask(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    section_filter: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> dict:
    """
    RAG 问答：检索 + DeepSeek 生成。

    参数:
        question: 用户自然语言问题
        top_k: 检索文章数
        temperature: 生成温度
        max_tokens: 最大输出 token
        date_from / date_to: 日期过滤
        section_filter: 版面过滤
        history: 多轮对话历史 [{"role":"user","content":"..."}, ...]

    返回:
        {
            "answer": "生成的回答",
            "articles": [...],   # 检索到的文章列表
            "model": "deepseek-chat",
            "usage": {"prompt_tokens": ..., "completion_tokens": ...}
        }
    """
    if not DEEPSEEK_API_KEY:
        return {
            "answer": "❌ 未配置 DEEPSEEK_API_KEY 环境变量，无法调用 DeepSeek API。\n"
                      "请在终端执行: export DEEPSEEK_API_KEY='your-api-key'",
            "articles": [],
            "model": DEEPSEEK_MODEL,
            "usage": {},
        }

    # Step 0: 自动解析提问中的时间范围
    auto_from, auto_to = parse_time_range(question)
    if not date_from:
        date_from = auto_from
    if not date_to:
        date_to = auto_to

    # Step 1: 检索
    articles = retrieve(question, top_k=top_k, date_from=date_from,
                        date_to=date_to, section_filter=section_filter)

    if not articles:
        time_hint = ""
        if date_from or date_to:
            time_hint = f"（时间范围: {date_from or '不限'} ~ {date_to or '不限'}）"
        return {
            "answer": f"未找到与您问题相关的天津日报文章{time_hint}，请尝试调整问题关键词或放宽筛选条件。",
            "articles": [],
            "model": DEEPSEEK_MODEL,
            "usage": {},
        }

    # Step 2: 构建上下文
    context = build_context(articles)

    # Step 3: 构建消息
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)

    user_message = (
        f"以下是根据你的问题检索到的天津日报相关文章：\n\n"
        f"{context}\n\n"
        f"用户问题：{question}\n\n"
        f"请根据以上文章内容回答问题。"
    )
    messages.append({"role": "user", "content": user_message})

    # Step 4: 调用 DeepSeek
    client = _get_client()
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "articles": articles,
        "model": DEEPSEEK_MODEL,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        },
    }


def ask_stream(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    section_filter: Optional[str] = None,
    history: Optional[list[dict]] = None,
):
    """
    流式版 RAG 问答 — 返回一个生成器，逐 chunk yield 文本。
    最后一个 chunk 为完整的 metadata dict。
    """
    if not DEEPSEEK_API_KEY:
        yield "❌ 未配置 DEEPSEEK_API_KEY 环境变量。请执行: export DEEPSEEK_API_KEY='your-api-key'"
        return

    # 自动解析提问中的时间范围
    auto_from, auto_to = parse_time_range(question)
    if not date_from:
        date_from = auto_from
    if not date_to:
        date_to = auto_to

    articles = retrieve(question, top_k=top_k, date_from=date_from,
                        date_to=date_to, section_filter=section_filter)

    if not articles:
        time_hint = ""
        if date_from or date_to:
            time_hint = f"（时间范围: {date_from or '不限'} ~ {date_to or '不限'}）"
        yield f"未找到相关文章{time_hint}，请调整关键词。"
        return

    context = build_context(articles)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": (
            f"以下是检索到的天津日报相关文章：\n\n{context}\n\n"
            f"用户问题：{question}\n\n请根据以上文章内容回答问题。"
        ),
    })

    client = _get_client()
    stream = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

    # 最后 yield 元数据
    yield {
        "articles": articles,
        "model": DEEPSEEK_MODEL,
    }


# ---------------------------------------------------------------------------
# CLI 测试入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if not DEEPSEEK_API_KEY:
        print("❌ 请先设置环境变量: export DEEPSEEK_API_KEY='your-api-key'")
        sys.exit(1)

    print("=" * 60)
    print("  天津日报 RAG 引擎 — DeepSeek API")
    print(f"  模型: {DEEPSEEK_MODEL}")
    print("=" * 60)

    if not load_kb():
        print("\n⚠ 知识库尚未构建！请先运行: python3 build_kb.py --build")
        sys.exit(1)

    stats = kb_stats()
    print(f"\n✅ 知识库已加载: {stats['total_articles']} 篇文章")
    print(f"📅 日期范围: {stats['date_range']}")
    print("\n输入问题开始对话，输入 quit 退出\n")

    history = []
    while True:
        try:
            question = input("\n🧑 您: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        print("\n🤖 AI: ", end="", flush=True)
        answer = ""
        for chunk in ask_stream(question, history=history):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
                answer += chunk
            else:
                # 最后一个 chunk 是 metadata
                print(f"\n\n--- 引用 {len(chunk.get('articles', []))} 篇文章 ---")
        print()

        # 保存历史
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        # 只保留最近 10 轮
        if len(history) > 20:
            history = history[-20:]
