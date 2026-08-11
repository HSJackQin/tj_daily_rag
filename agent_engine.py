#!/usr/bin/env python3
"""
天津日报 V0.2 Agent 引擎
- 模型自主决策搜索策略（关键词、角度、轮次）
- 支持 Tool Calling (search_tjrb)
- 混合检索: BGE-M3 语义 + 关键词
- 流式 SSE 事件输出
"""

import json
import os
import time
from datetime import datetime
from typing import Optional

from openai import OpenAI

from rag_engine import (
    DEEPSEEK_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    build_context,
)
from time_parser import parse_time_range

# 最大 Agent 搜索轮次
MAX_ROUNDS = int(os.environ.get("AGENT_MAX_ROUNDS", "5"))
# 每篇文章传给模型的摘要长度（节省 token）
ARTICLE_SUMMARY_LEN = 300

# ---------------------------------------------------------------------------
# Tool 定义 (OpenAI function-calling 格式)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_tjrb",
            "description": (
                "搜索天津日报知识库，返回与关键词相关的新闻文章。"
                "可以多次调用，用不同的关键词从不同角度搜索。"
                "例如：先搜'天津港 发展'，如果结果不够，再搜'天津港 智慧港口 吞吐量'。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词（支持多词组合），必填",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "起始日期 YYYY-MM-DD，不填则不限制",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "结束日期 YYYY-MM-DD，不填则不限制",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回数量，默认15，最大30",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

AGENT_SYSTEM_PROMPT = """你是一个专业的天津日报信息助手，帮助用户从近一年的天津日报中检索并梳理信息。

## 工作方式
你可以使用 search_tjrb 工具来搜索天津日报知识库。请按以下步骤工作：

1. **深度分析**：先仔细思考用户问题的真正意图、涉及的关键维度、需要覆盖的时间范围
2. **多角度搜索**：基于你的分析，从不同关键词角度搜索，确保信息全面
3. **综合回答**：基于搜索到的文章内容，给出准确、有条理的回答

## 搜索策略
- 根据你的深度分析，制定搜索计划
- 从不同角度、不同关键词多次搜索
- 对于梳理/总结类问题，至少搜索 2-3 次不同角度
- 时间范围明确的（如"最近一周"、"本月"），搜索时带上日期过滤

## 回答规范
- 只依据搜索到的文章内容作答，不要编造
- 引用文章时注明日期和标题（格式：「YYYY-MM-DD《标题》」）
- 材料整理类问题要结构清晰、有层次
- 如果信息不足，诚实告知并建议调整搜索条件

## 工具调用方式
当你需要搜索时，在回复中插入以下格式的工具调用：
<tool_call>
{"name": "search_tjrb", "arguments": {"query": "搜索关键词", "date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD", "top_k": 15}}
</tool_call>
你可以在一轮中插入多个 tool_call。"""

THINKING_PROMPT = """请对以下问题进行深度分析和拆解，然后制定搜索计划。

在分析中，请思考：
1. 这个问题的核心是什么？用户真正想了解什么？
2. 需要从哪些维度/角度来回答？（列出具体的方面）
3. 每个维度应该用什么关键词搜索？
4. 需要关注什么时间范围？

分析完毕后，使用 <tool_call> 标签按计划开始搜索。"""


def _parse_xml_tool_calls(content: str) -> tuple[list[dict], str]:
    """
    从 DeepSeek 的 XML 格式 content 中解析工具调用。
    返回: (tool_calls列表, 移除tool_call后的纯文本)
    """
    import re

    tool_calls = []
    # 匹配 <tool_call>...</tool_call>
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)

    for i, json_str in enumerate(matches):
        try:
            tc = json.loads(json_str)
            tool_calls.append({
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": tc.get("name", "search_tjrb"),
                    "arguments": json.dumps(tc.get("arguments", {}))
                }
            })
        except json.JSONDecodeError:
            continue

    # 移除 tool_call 标签，保留纯文本
    clean_text = re.sub(r'<tool_call>\s*\{.*?\}\s*</tool_call>', '', content, flags=re.DOTALL)
    clean_text = clean_text.strip()

    return tool_calls, clean_text


def _get_client() -> OpenAI:
    """创建 DeepSeek 客户端（绕过代理）"""
    import httpx

    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL",
                                        "https://api.deepseek.com")

    transport = httpx.HTTPTransport(verify=True, retries=1)
    http_client = httpx.Client(transport=transport)

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

    for k, v in saved.items():
        os.environ[k] = v

    return client


def _do_search(query: str, date_from=None, date_to=None, top_k=15):
    """TF-IDF 关键词检索"""
    from rag_engine import retrieve
    articles = retrieve(
        query=query,
        top_k=min(top_k, 30),
        date_from=date_from if date_from else None,
        date_to=date_to if date_to else None,
    )
    return articles, "tfidf"


def _format_search_results(articles: list, max_per_article: int = 300) -> str:
    """将搜索结果格式化为给模型的摘要文本"""
    if not articles:
        return "（未找到相关文章）"

    parts = []
    for i, art in enumerate(articles[:20]):  # 最多给模型看 20 篇
        content = art.get("content", "")
        if len(content) > max_per_article:
            content = content[:max_per_article] + "……"

        parts.append(
            f"【文章{i + 1}】\n"
            f"日期: {art['date']}\n"
            f"版面: {art['section']}\n"
            f"标题: {art['title']}\n"
            f"概要: {content}\n"
        )

    header = f"找到 {len(articles)} 篇文章，以下是最相关的 {min(len(articles), 20)} 篇：\n\n"
    return header + "\n".join(parts)


def agent_ask(
    question: str,
    history: list[dict] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    section_filter: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    deep_think: bool = True,
):
    """
    Agent 问答 — 生成器，逐条 yield SSE 事件 dict。

    事件类型:
        {"type": "status", "text": "..."}
        {"type": "tool_call", "search_id": N, "query": "...", ...}
        {"type": "tool_result", "search_id": N, "count": N, "method": "hybrid|tfidf"}
        {"type": "chunk", "text": "..."}
        {"type": "meta", "articles": [...], "model": "...", "total_searches": N}
        {"type": "error", "message": "..."}
    """
    if not os.environ.get("DEEPSEEK_API_KEY"):
        yield {
            "type": "error",
            "message": "未配置 DEEPSEEK_API_KEY 环境变量",
        }
        return

    client = _get_client()

    # ── 自动解析时间范围 ──
    auto_from, auto_to = parse_time_range(question)
    if not date_from:
        date_from = auto_from
    if not date_to:
        date_to = auto_to

    # ── 构建消息 ──
    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-20:])  # 最多保留 10 轮对话

    # 用户消息：附上时间范围提示
    user_prompt = question
    time_hint = ""
    if date_from or date_to:
        time_hint = f"\n（提示：用户可能关注的时间范围为 {date_from or '不限'} ~ {date_to or '不限'}，搜索时请酌情使用 date_from/date_to 参数。）"
    messages.append({"role": "user", "content": user_prompt + time_hint})

    # ── 统计 ──
    all_articles = {}  # key=(title,date) 去重
    search_count = 0
    search_summary = []  # 记录每次搜索的摘要

    # 深度思考关闭时限制轮次
    max_rounds = MAX_ROUNDS if deep_think else 3

    # ── 阶段1: 深度思考（可选）───
    pre_tool_calls = []
    if deep_think:
        yield {"type": "status", "text": "正在深度思考，分析问题..."}

        thinking_messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
        if history:
            thinking_messages.extend(history[-20:])
        thinking_messages.append({"role": "user", "content": THINKING_PROMPT + "\n\n用户问题：" + user_prompt + time_hint})

        thinking_response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=thinking_messages,
            tools=TOOLS,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        thinking_msg = thinking_response.choices[0].message
        thinking_content = thinking_msg.content or ""

        pre_tool_calls = list(thinking_msg.tool_calls) if thinking_msg.tool_calls else []
        if not pre_tool_calls and thinking_content and '<tool_call>' in thinking_content:
            pre_tool_calls, thinking_content = _parse_xml_tool_calls(thinking_content)

        if thinking_content:
            yield {"type": "thinking", "text": thinking_content}
            yield {"type": "status", "text": "深度分析完成，开始多轮检索..."}

        messages.append({"role": "assistant", "content": thinking_content})
    else:
        yield {"type": "status", "text": "正在检索相关文章（快速模式）..."}

    # ── 阶段2: 执行搜索（处理思考阶段可能产生的 tool calls）───
    if pre_tool_calls:
        messages[-1]["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type if hasattr(tc, 'type') else "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in pre_tool_calls
        ]

        for tc in pre_tool_calls:
            if tc.function.name != "search_tjrb":
                continue
            search_count += 1
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"query": tc.function.arguments}

            query = args.get("query", "")
            k = min(args.get("top_k", 15), 30)
            df = args.get("date_from") or date_from
            dt = args.get("date_to") or date_to

            yield {
                "type": "tool_call",
                "search_id": search_count,
                "query": query,
                "date_from": df,
                "date_to": dt,
            }

            articles, method = _do_search(query, date_from=df, date_to=dt, top_k=k)
            new_count = 0
            for art in articles:
                key = (art["title"], art["date"])
                if key not in all_articles:
                    all_articles[key] = art
                    new_count += 1

            yield {
                "type": "tool_result",
                "search_id": search_count,
                "query": query,
                "count": len(articles),
                "new_count": new_count,
                "total_count": len(all_articles),
                "method": method,
            }

            search_summary.append({
                "search_id": search_count,
                "query": query,
                "found": len(articles),
                "method": method,
            })

            result_text = _format_search_results(articles, ARTICLE_SUMMARY_LEN)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

    # ── 阶段3: Agent 多轮搜索循环 ──
    for round_num in range(max_rounds):
        # 非流式调用（带 tools）
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        msg = response.choices[0].message

        # ── 处理 tool calls（标准格式或 DeepSeek XML 格式）───
        tool_calls = list(msg.tool_calls) if msg.tool_calls else []

        # 如果标准 tool_calls 为空，尝试从 content 中解析 XML 格式
        if not tool_calls and msg.content and '<tool_call>' in msg.content:
            tool_calls, clean_content = _parse_xml_tool_calls(msg.content)
            if tool_calls:
                # 用清理后的文本替换原始 content
                msg.content = clean_content

        if tool_calls:
            # 先附加 assistant 消息（含 tool_calls）
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            # 再附加 tool 结果（作为对 tool_calls 的响应）
            for tc in tool_calls:
                if tc.function.name != "search_tjrb":
                    continue

                search_count += 1
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"query": tc.function.arguments}

                query = args.get("query", "")
                k = min(args.get("top_k", 15), 30)
                df = args.get("date_from") or date_from
                dt = args.get("date_to") or date_to

                # 通知前端：开始搜索
                yield {
                    "type": "tool_call",
                    "search_id": search_count,
                    "query": query,
                    "date_from": df,
                    "date_to": dt,
                }

                # 执行搜索
                articles, method = _do_search(query, date_from=df, date_to=dt, top_k=k)

                # 去重合并
                new_count = 0
                for art in articles:
                    key = (art["title"], art["date"])
                    if key not in all_articles:
                        all_articles[key] = art
                        new_count += 1

                # 通知前端：搜索完成
                yield {
                    "type": "tool_result",
                    "search_id": search_count,
                    "query": query,
                    "count": len(articles),
                    "new_count": new_count,
                    "total_count": len(all_articles),
                    "method": method,
                }

                search_summary.append({
                    "search_id": search_count,
                    "query": query,
                    "found": len(articles),
                    "method": method,
                })

                # 将 tool 结果附加到对话
                result_text = _format_search_results(articles, ARTICLE_SUMMARY_LEN)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })

            continue  # 继续下一轮

        # ── 模型直接回答（无 tool calls） ──
        # 此时切换到流式输出最终回答
        yield {"type": "status", "text": "正在生成回答..."}

        # 重新发起流式请求（带上完整对话）
        stream = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        full_answer = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                full_answer += text
                yield {"type": "chunk", "text": text}

        # ── 最终元数据 ──
        # 按日期排序参考文章
        sorted_articles = sorted(
            all_articles.values(),
            key=lambda x: x["date"],
            reverse=True,
        )

        yield {
            "type": "meta",
            "articles": [
                {
                    "title": a["title"],
                    "date": a["date"],
                    "section": a["section"],
                    "score": a.get("score", 0),
                    "source_url": a.get("source_url", ""),
                }
                for a in sorted_articles[:30]  # 最多返回 30 篇引用
            ],
            "model": DEEPSEEK_MODEL,
            "answer_length": len(full_answer),
            "total_searches": search_count,
            "total_unique_articles": len(all_articles),
            "search_summary": search_summary,
        }

        return  # Agent 完成

    # ── 达到最大轮次，强制生成回答 ──
    yield {"type": "status",
           "text": f"已完成 {MAX_ROUNDS} 轮搜索，正在生成回答..."}

    stream = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    full_answer = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full_answer += text
            yield {"type": "chunk", "text": text}

    sorted_articles = sorted(
        all_articles.values(),
        key=lambda x: x["date"],
        reverse=True,
    )
    yield {
        "type": "meta",
        "articles": [
            {
                "title": a["title"],
                "date": a["date"],
                "section": a["section"],
                "score": a.get("score", 0),
                "source_url": a.get("source_url", ""),
            }
            for a in sorted_articles[:30]
        ],
        "model": DEEPSEEK_MODEL,
        "answer_length": len(full_answer),
        "total_searches": search_count,
        "total_unique_articles": len(all_articles),
        "search_summary": search_summary,
    }


# ---------------------------------------------------------------------------
# CLI 测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("❌ 请先设置 DEEPSEEK_API_KEY")
        sys.exit(1)

    from rag_engine import load_kb

    print("=" * 60)
    print("  天津日报 V0.2 Agent 引擎")
    print(f"  模型: {DEEPSEEK_MODEL}")
    print(f"  最大搜索轮次: {MAX_ROUNDS}")
    print("=" * 60)

    kb_ok = load_kb()
    print(f"  TF-IDF 知识库: {'✅' if kb_ok else '❌'}")

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

        print()
        answer_parts = []
        for event in agent_ask(question, history=history):
            etype = event.get("type")

            if etype == "status":
                print(f"  ℹ️  {event['text']}")

            elif etype == "tool_call":
                sid = event["search_id"]
                q = event["query"]
                df = event.get("date_from", "")
                dt = event.get("date_to", "")
                time_str = f" [{df}~{dt}]" if df or dt else ""
                print(f"  🔍 [{sid}] 搜索: \"{q}\"{time_str}")

            elif etype == "tool_result":
                sid = event["search_id"]
                method = event.get("method", "?")
                n = event["count"]
                new = event.get("new_count", n)
                total = event.get("total_count", n)
                print(f"  📄 [{sid}] 找到 {n} 篇 (新增 {new}, 累计 {total}) "
                      f"[{method}]")

            elif etype == "chunk":
                if not answer_parts:
                    print("  🤖 ", end="", flush=True)
                print(event["text"], end="", flush=True)
                answer_parts.append(event["text"])

            elif etype == "meta":
                if answer_parts:
                    print()
                answer = "".join(answer_parts)
                n_searches = event.get("total_searches", 0)
                n_articles = event.get("total_unique_articles", 0)
                print(f"\n  --- 共搜索 {n_searches} 次，"
                      f"引用 {n_articles} 篇文章 ---")

                # 保存历史
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": answer})
                if len(history) > 20:
                    history = history[-20:]

            elif etype == "error":
                print(f"  ❌ {event['message']}")
