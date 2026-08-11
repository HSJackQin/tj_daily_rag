#!/usr/bin/env python3
"""
天津日报知识库 Web 服务 —— 搜索 + AI 智能问答
- 关键词全文搜索 (TF-IDF)
- AI 智能问答 (DeepSeek RAG)
- 启动: python3 kb_server.py
- 访问: http://localhost:8699
"""

import json
import os
import pickle
import re
import sys
import urllib.parse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from tjrb_tokenizer import jieba_tokenizer

# 导入 RAG 引擎
from rag_engine import (
    load_kb as rag_load_kb,
    is_ready as rag_ready,
    kb_stats,
    retrieve,
    ask,
    ask_stream,
    DEEPSEEK_MODEL,
)

# 导入 Agent 引擎 (V0.2)
from agent_engine import agent_ask

KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tjrb_kb")
HOST = "0.0.0.0"
PORT = 8699
ACCESS_PASSWORD = os.environ.get("ACCESS_PASSWORD", "")  # 访问密钥，为空则不启用

# ---------------------------------------------------------------------------
# TF-IDF 检索（复用原有逻辑，保持搜索功能独立）
# ---------------------------------------------------------------------------
vectorizer = None
tfidf_matrix = None
metadata = None


def load_tfidf_kb():
    """加载 TF-IDF 索引（搜索用）"""
    global vectorizer, tfidf_matrix, metadata
    v_path = os.path.join(KB_DIR, "vectorizer.pkl")
    m_path = os.path.join(KB_DIR, "tfidf_matrix.npz")
    d_path = os.path.join(KB_DIR, "metadata.json")
    if not all(os.path.exists(p) for p in [v_path, m_path, d_path]):
        return False
    with open(v_path, "rb") as f:
        vectorizer = pickle.load(f)
    tfidf_matrix = load_npz(m_path)
    with open(d_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return True


# ---------------------------------------------------------------------------
# 查询扩展 (V0.2)
# ---------------------------------------------------------------------------
EXPAND_PROMPT = """你是一个搜索词扩展助手。将用户输入的关键词拆分为 2-4 个不同角度的搜索短语，用于在新闻库中检索。

规则：
1. 保留原意，从不同表述角度拆分
2. 如果原词包含多个概念，分别组合
3. 最多 4 个短语，每个 2-8 个字
4. 严格返回 JSON 格式，不要其他文字

示例：
输入: "天津港发展"
输出: {"phrases": ["天津港发展", "天津港 经济 成效", "天津港口 增长"]}

输入: "人工智能产业政策"
输出: {"phrases": ["人工智能产业政策", "AI 产业发展", "人工智能 政策扶持", "智能科技 产业"]}"""


def expand_query(query: str) -> list[str]:
    """用 DeepSeek 扩展搜索关键词，返回多个搜索短语"""
    import json as _json
    try:
        from agent_engine import _get_client
        client = _get_client()
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": EXPAND_PROMPT},
                {"role": "user", "content": f'请扩展: "{query}"'},
            ],
            temperature=0.3,
            max_tokens=200,
            stream=False,
        )
        text = response.choices[0].message.content.strip()
        # 提取 JSON（可能包裹在 ```json ... ``` 中）
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = _json.loads(text)
        phrases = data.get("phrases", [query])
        # 确保原查询在第一位
        if query not in phrases:
            phrases.insert(0, query)
        return phrases[:5]
    except Exception:
        return [query]


def tfidf_search(query, top_k=20, date_from=None, date_to=None,
                 section_filter=None, sort_by="relevance", title_only=False,
                 expand=False):
    if vectorizer is None:
        return []

    # ── 查询扩展 ──
    if expand and query:
        phrases = expand_query(query)
    else:
        phrases = [query]

    # 提取查询关键词
    keywords = [w for w in query.strip().split() if len(w) >= 2]

    if title_only:
        # ── 仅标题模式：直接做关键词匹配，不走 TF-IDF ──
        results = []
        for idx, art in enumerate(metadata):
            if date_from and art["date"] < date_from:
                continue
            if date_to and art["date"] > date_to:
                continue
            if section_filter and section_filter not in art["section"]:
                continue
            title = art["title"]
            # 计算标题命中关键词的个数和权重
            matched = 0
            for kw in keywords:
                if kw.lower() in title.lower():
                    matched += 1
            if matched == 0:
                continue
            art = dict(art)
            # 得分：命中数 / 总关键词数，标题完全命中所有词=1.0
            art["score"] = matched / len(keywords)
            results.append(art)
        # 按得分降序
        results.sort(key=lambda x: x["score"], reverse=True)
        if sort_by == "date_desc":
            results.sort(key=lambda x: x["date"], reverse=True)
        return results[:top_k]

    # ── TF-IDF 搜索（支持多短语扩展）───
    seen = {}  # key=(title,date) → best_score
    for phrase in phrases:
        query_vec = vectorizer.transform([phrase])
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
        ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        for idx, score in ranked:
            if score < 0.01:
                continue
            art = metadata[idx]
            if date_from and art["date"] < date_from:
                continue
            if date_to and art["date"] > date_to:
                continue
            if section_filter and section_filter not in art["section"]:
                continue

            # 标题加权：标题中每命中一个关键词，得分提升 20%
            title = art["title"]
            title_match = 0
            for kw in keywords:
                if kw.lower() in title.lower():
                    title_match += 1
            if title_match > 0:
                score = float(score) * (1.0 + 0.2 * title_match)
            else:
                score = float(score)

            key = (art["title"], art["date"])
            # 扩展模式下取多短语中的最高分
            if key not in seen or score > seen[key][0]:
                seen[key] = (score, art)

    # 去重合并，按得分排序
    results = []
    for (title, date), (score, art) in seen.items():
        art = dict(art)
        art["score"] = score
        results.append(art)

    if sort_by == "relevance":
        results.sort(key=lambda x: x["score"], reverse=True)
    elif sort_by == "date_desc":
        results.sort(key=lambda x: x["date"], reverse=True)

    return results[:top_k]


def highlight(text, query):
    if not query or not text:
        return text
    keywords = [w for w in query.strip().split() if len(w) >= 2]
    for kw in keywords:
        text = re.sub(
            f"({re.escape(kw)})", r"<mark>\1</mark>", text, flags=re.IGNORECASE
        )
    return text


# ---------------------------------------------------------------------------
# HTML 页面模板
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>天津日报知识库</title>
<style>
:root {
  --bg: #FAFAFA;
  --surface: #FFFFFF;
  --ink: #1A1A1A;
  --muted: #6B7280;
  --rule: #E5E7EB;
  --accent: #DC2626;
  --accent-soft: #FEF2F2;
  --radius: 10px;
  --font: "PingFang SC","Noto Sans CJK SC","Microsoft YaHei",system-ui,sans-serif;
}
*{margin:0;padding:0;box-sizing:border-box}
body{
  font-family:var(--font);background:var(--bg);color:var(--ink);
  line-height:1.6;min-height:100vh;
}
header{
  background:var(--surface);border-bottom:1px solid var(--rule);
  padding:16px 0;position:sticky;top:0;z-index:100;
}
.header-inner{max-width:900px;margin:0 auto;padding:0 20px}
header h1{font-size:22px;font-weight:700;color:var(--accent);margin-bottom:2px}
header .subtitle{font-size:13px;color:var(--muted)}

/* Tabs */
.tabs{display:flex;gap:0;max-width:900px;margin:0 auto;padding:0 20px;
      border-bottom:2px solid var(--rule)}
.tab-btn{
  padding:12px 28px;font-size:15px;font-weight:600;border:none;
  background:none;color:var(--muted);cursor:pointer;
  font-family:var(--font);border-bottom:3px solid transparent;
  margin-bottom:-2px;transition:all 0.2s;
}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-btn:hover:not(.active){color:var(--ink)}
.tab-panel{display:none}
.tab-panel.active{display:block}

/* Search */
.search-box{max-width:900px;margin:0 auto;padding:24px 20px}
.search-form{display:flex;gap:10px;align-items:center}
.search-input{
  flex:1;padding:12px 18px;font-size:16px;
  border:2px solid var(--rule);border-radius:var(--radius);
  font-family:var(--font);outline:none;transition:border-color .2s;
}
.search-input:focus{border-color:var(--accent)}
.search-btn{
  padding:12px 28px;font-size:16px;font-weight:600;
  background:var(--accent);color:#FFF;border:none;
  border-radius:var(--radius);cursor:pointer;font-family:var(--font);
  transition:opacity .2s;
}
.search-btn:hover{opacity:.9}
.search-btn:disabled{opacity:.5;cursor:not-allowed}
.filters{display:flex;gap:12px;margin-top:12px;flex-wrap:wrap}
.filters input,.filters select{
  padding:8px 12px;font-size:13px;border:1px solid var(--rule);
  border-radius:6px;font-family:var(--font);background:var(--surface);
}

/* Results */
.results{max-width:900px;margin:0 auto;padding:0 20px 40px}
.result-card{
  background:var(--surface);border:1px solid var(--rule);
  border-radius:var(--radius);padding:20px;margin-bottom:14px;
  transition:box-shadow .2s;
}
.result-card:hover{box-shadow:0 2px 12px rgba(0,0,0,.06)}
.result-header{
  display:flex;justify-content:space-between;
  align-items:flex-start;gap:12px;margin-bottom:8px;
}
.result-title{font-size:17px;font-weight:600;color:var(--ink);text-decoration:none}
.result-title:hover{color:var(--accent);text-decoration:underline}
.result-score{
  font-size:12px;color:var(--accent);background:var(--accent-soft);
  padding:3px 8px;border-radius:12px;white-space:nowrap;font-weight:500;
}
.result-meta{
  font-size:13px;color:var(--muted);margin-bottom:10px;
  display:flex;gap:16px;flex-wrap:wrap;
}
.result-content{
  font-size:14px;color:var(--ink);line-height:1.7;
  max-height:200px;overflow:hidden;position:relative;
}
.result-content::after{
  content:'';position:absolute;bottom:0;left:0;right:0;
  height:40px;background:linear-gradient(transparent,var(--surface));
}
.result-full{max-height:none;overflow:visible}
.result-full::after{display:none}
.expand-link{
  display:inline-block;font-size:13px;color:var(--accent);
  text-decoration:none;cursor:pointer;margin-top:6px;font-weight:500;
}
.expand-link:hover{text-decoration:underline}
.result-link{
  font-size:12px;color:var(--accent);text-decoration:none;
  margin-top:6px;display:inline-block;margin-left:12px;
}
.result-link:hover{text-decoration:underline}
/* Login overlay */
.login-overlay{
  position:fixed;top:0;left:0;width:100%;height:100%;
  background:rgba(0,0,0,0.5);display:flex;align-items:center;
  justify-content:center;z-index:9999;
}
.login-box{
  background:var(--surface);padding:32px 36px;border-radius:var(--radius);
  box-shadow:0 8px 32px rgba(0,0,0,0.15);text-align:center;max-width:360px;width:90%;
}
.login-box h2{font-size:20px;margin-bottom:6px;color:var(--ink)}
.login-box p{font-size:13px;color:var(--muted);margin-bottom:20px}
.login-box input{
  width:100%;padding:12px 16px;font-size:15px;border:2px solid var(--rule);
  border-radius:var(--radius);font-family:var(--font);outline:none;
  text-align:center;margin-bottom:12px;
}
.login-box input:focus{border-color:var(--accent)}
.login-box .login-btn{
  width:100%;padding:10px;font-size:15px;font-weight:600;
  background:var(--accent);color:#FFF;border:none;border-radius:var(--radius);
  cursor:pointer;font-family:var(--font);
}
.login-box .login-btn:hover{opacity:.9}
.login-error{color:var(--accent);font-size:12px;margin-top:8px;display:none}

mark{background:#FDE68A;color:#92400E;padding:1px 3px;border-radius:2px}
.empty{text-align:center;padding:60px 20px;color:var(--muted)}
.empty h2{font-size:18px;margin-bottom:8px;color:var(--ink)}
.stats-bar{
  font-size:13px;color:var(--muted);margin-bottom:16px;
  padding:0 20px;max-width:900px;margin-left:auto;margin-right:auto;
}
footer{
  text-align:center;padding:20px;color:var(--muted);font-size:12px;
  border-top:1px solid var(--rule);margin-top:40px;
}

/* ====== AI Q&A Styles ====== */
.qa-container{max-width:900px;margin:0 auto;padding:24px 20px}
.qa-input-area{display:flex;gap:10px}
.qa-input{
  flex:1;padding:14px 18px;font-size:16px;
  border:2px solid var(--rule);border-radius:var(--radius);
  font-family:var(--font);resize:none;outline:none;
  transition:border-color .2s;min-height:80px;
}
.qa-input:focus{border-color:var(--accent)}
.qa-actions{display:flex;flex-direction:column;gap:10px}
.qa-btn{
  padding:10px 24px;font-size:14px;font-weight:600;
  background:var(--accent);color:#FFF;border:none;
  border-radius:var(--radius);cursor:pointer;font-family:var(--font);
  transition:opacity .2s;white-space:nowrap;
}
.qa-btn:hover{opacity:.9}
.qa-btn:disabled{opacity:.5;cursor:not-allowed}
.qa-btn-clear{
  padding:10px 24px;font-size:13px;font-weight:500;
  background:none;color:var(--muted);border:1px solid var(--rule);
  border-radius:var(--radius);cursor:pointer;font-family:var(--font);
}
.qa-examples{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.qa-example{
  font-size:12px;padding:6px 12px;background:var(--accent-soft);
  color:var(--accent);border-radius:16px;cursor:pointer;
  border:1px solid transparent;transition:all .2s;
}
.qa-example:hover{border-color:var(--accent)}

/* Chat bubble for Q&A */
.chat-area{margin-top:20px}
.chat-msg{margin-bottom:20px}
.chat-msg-user{text-align:right}
.chat-msg-user .chat-bubble{
  display:inline-block;background:var(--accent);color:#FFF;
  padding:12px 18px;border-radius:18px 18px 4px 18px;
  max-width:80%;text-align:left;font-size:15px;line-height:1.7;
  white-space:pre-wrap;word-break:break-word;
}
.chat-msg-ai .chat-bubble{
  display:inline-block;background:var(--surface);
  border:1px solid var(--rule);padding:16px 20px;
  border-radius:4px 18px 18px 18px;max-width:90%;
  font-size:15px;line-height:1.8;white-space:pre-wrap;
  word-break:break-word;
}
.chat-meta{font-size:11px;color:var(--muted);margin-top:4px}
.typing-indicator{
  display:inline-block;padding:12px 20px;background:var(--surface);
  border:1px solid var(--rule);border-radius:4px 18px 18px 18px;
  color:var(--muted);font-size:14px;
}
.typing-indicator span{animation:blink 1.4s infinite}
.typing-indicator span:nth-child(2){animation-delay:.2s}
.typing-indicator span:nth-child(3){animation-delay:.4s}
@keyframes blink{0%,60%,100%{opacity:.3}30%{opacity:1}}

/* Agent steps (V0.2) */
.agent-steps{margin-bottom:8px}
.agent-step{
  display:flex;align-items:flex-start;gap:8px;
  padding:8px 14px;margin-bottom:4px;border-radius:8px;
  font-size:13px;animation:fadeIn .3s ease;
}
@keyframes fadeIn{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:translateY(0)}}
.agent-step-status{
  background:#F0F9FF;color:#0369A1;border:1px solid #BAE6FD;
}
.agent-step-search{
  background:#FFF7ED;color:#C2410C;border:1px solid #FED7AA;
}
.agent-step-result{
  background:#F0FDF4;color:#166534;border:1px solid #BBF7D0;
}
.agent-step-icon{font-size:15px;flex-shrink:0;line-height:1.4}
.agent-step-content{flex:1;line-height:1.5}
.agent-step-query{font-weight:600}
.agent-step-summary{
  margin-top:8px;padding-top:6px;border-top:1px solid var(--rule);
  font-size:12px;color:var(--muted);
}

/* References in AI answer */
.ai-refs{
  margin-top:12px;padding-top:10px;border-top:1px solid var(--rule);
  font-size:12px;color:var(--muted);
}
.ai-refs summary{cursor:pointer;font-weight:500;color:var(--muted)}
.ai-ref-item{font-size:12px;color:var(--muted);padding:2px 0}
.ai-ref-link{color:var(--accent);text-decoration:none}
.ai-ref-link:hover{text-decoration:underline}

/* Copy button */
.copy-btn{
  display:inline-block;margin-top:8px;font-size:12px;
  color:var(--accent);cursor:pointer;background:none;border:none;
  font-family:var(--font);padding:2px 8px;
}
.copy-btn:hover{text-decoration:underline}

@media(max-width:640px){
  .search-form{flex-direction:column}
  .search-btn{width:100%}
  .qa-input-area{flex-direction:column}
  .qa-actions{flex-direction:row}
  .result-header{flex-direction:column}
  .tabs{overflow-x:auto}
}
</style>
</head>
<body>

<div id="loginOverlay" class="login-overlay" style="display:none">
  <div class="login-box">
    <h2>&#x1F512; 访问验证</h2>
    <p>请输入访问密钥</p>
    <input id="loginInput" type="password" placeholder="请输入密钥" onkeydown="if(event.key==='Enter')doLogin()">
    <button class="login-btn" onclick="doLogin()">确 认</button>
    <div class="login-error" id="loginError">密钥错误，请重试</div>
  </div>
</div>

<header>
  <div class="header-inner">
    <h1>&#x1F4F0; 天津日报知识库</h1>
    <div class="subtitle">
      近一年天津日报全文检索 &middot; AI 智能问答 &middot; 写稿参考资料
      <span style="margin-left:12px;font-size:11px;color:#aaa">__KB_INFO__</span>
    </div>
  </div>
</header>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('search')">&#x1F50D; 关键词搜索</button>
  <button class="tab-btn" onclick="switchTab('ask')">&#x1F916; AI 智能问答</button>
</div>

<!-- ===== Tab: &#x5173;&#x952E;&#x8BCD;&#x641C;&#x7D22; ===== -->
<div id="tab-search" class="tab-panel active">
  <div class="search-box">
    <form id="searchForm" method="GET" action="/search" accept-charset="UTF-8">
      <div class="search-form">
        <input class="search-input" type="text" name="q"
               placeholder="输入关键词搜索，如：民心工程、人工智能、防汛..."
               value="__QUERY_ESC__" autofocus>
        <button class="search-btn" type="submit">&#x1F50D; 搜索</button>
      </div>
      <div class="filters">
        <input type="date" name="date_from" value="__DATE_FROM__" placeholder="开始日期">
        <input type="date" name="date_to" value="__DATE_TO__" placeholder="结束日期">
        <select name="section">
          <option value="">全部版面</option>
          __SECTION_OPTIONS__
        </select>
        <select name="sort">
          <option value="relevance" __SORT_RELEVANCE_SEL__>按相关度</option>
          <option value="date_desc" __SORT_DATE_SEL__>按时间从晚到早</option>
        </select>
        <label style="display:flex;align-items:center;gap:4px;font-size:13px;color:var(--ink);cursor:pointer;user-select:none">
          <input type="checkbox" name="title_only" value="1" __TITLE_ONLY_CHECKED__ onchange="this.form.submit()">
          仅匹配标题
        </label>
        <label style="display:flex;align-items:center;gap:4px;font-size:13px;color:var(--ink);cursor:pointer;user-select:none">
          <input type="checkbox" name="expand" value="1" __EXPAND_CHECKED__ onchange="this.form.submit()">
          ✨ 智能扩展
        </label>
        <button class="search-btn" type="submit" style="padding:8px 16px;font-size:13px">筛选</button>
      </div>
    </form>
  </div>
  <div class="results">__RESULTS__</div>
</div>

<!-- ===== Tab: AI &#x667A;&#x80FD;&#x95EE;&#x7B54; ===== -->
<div id="tab-ask" class="tab-panel">
  <div class="qa-container">
    <div class="qa-input-area">
      <textarea id="qaInput" class="qa-input" rows="3"
        placeholder="用自然语言提问，AI 会从天津日报知识库中检索相关文章并生成回答。&#10;&#10;例如：请帮我梳理近一年天津日报上刊登的关于宣传天津港的内容，并提炼成一篇300字左右，反映天津港近一年亮点成效的文章。"></textarea>
      <div class="qa-actions">
        <button id="qaSubmit" class="qa-btn" onclick="askAI()">&#x1F916; 提问</button>
        <button class="qa-btn-clear" onclick="clearChat()">清空对话</button>
      </div>
    </div>
    <div class="qa-examples">
      <span style="font-size:12px;color:var(--muted);margin-right:4px">示例：</span>
      <span class="qa-example" onclick="useExample(this)">梳理天津港近一年亮点成效，300字文稿</span>
      <span class="qa-example" onclick="useExample(this)">近一年天津在人工智能领域有哪些重要举措？</span>
      <span class="qa-example" onclick="useExample(this)">总结近一年民心工程的进展和成果</span>
      <span class="qa-example" onclick="useExample(this)">天津防汛工作的相关报道有哪些？</span>
    </div>
    <div id="chatArea" class="chat-area"></div>
  </div>
</div>

<footer>
  天津日报知识库 &middot; 数据来源：epaper.tianjinwe.com &middot; AI 模型：__MODEL_NAME__ &middot; 仅供个人写稿参考
</footer>

<script>
// ===== Login =====
var ACCESS_PASSWORD = "__ACCESS_PASSWORD__";
var accessKey = sessionStorage.getItem('tjrb_key') || '';

function doLogin() {
  var input = document.getElementById('loginInput');
  var val = input.value.trim();
  if (val === ACCESS_PASSWORD) {
    accessKey = val;
    sessionStorage.setItem('tjrb_key', val);
    document.getElementById('loginOverlay').style.display = 'none';
    document.getElementById('loginError').style.display = 'none';
  } else {
    document.getElementById('loginError').style.display = 'block';
    input.value = '';
    input.focus();
  }
}

(function() {
  if (ACCESS_PASSWORD) {
    if (accessKey !== ACCESS_PASSWORD) {
      document.getElementById('loginOverlay').style.display = 'flex';
      document.getElementById('loginInput').focus();
    }
  }
})();

// ===== Tab Switching =====
function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  if (tab === 'search') {
    document.querySelector('.tab-btn').classList.add('active');
    document.getElementById('tab-search').classList.add('active');
  } else {
    document.querySelectorAll('.tab-btn')[1].classList.add('active');
    document.getElementById('tab-ask').classList.add('active');
  }
}

// ===== AI Q&A =====
let chatHistory = [];
let isAsking = false;

function useExample(el) {
  document.getElementById('qaInput').value = el.textContent;
  switchTab('ask');
}

function clearChat() {
  chatHistory = [];
  document.getElementById('chatArea').innerHTML = '';
}

function askAI() {
  var input = document.getElementById('qaInput');
  var question = input.value.trim();
  if (!question || isAsking) return;

  isAsking = true;
  var submitBtn = document.getElementById('qaSubmit');
  submitBtn.disabled = true;
  input.value = '';

  // 显示用户消息
  var chatArea = document.getElementById('chatArea');
  chatArea.innerHTML +=
    '<div class="chat-msg chat-msg-user">' +
    '<div class="chat-bubble">' + escapeHtml(question) + '</div>' +
    '</div>';

  // 显示输入指示器
  var typingId = 'typing-' + Date.now();
  chatArea.innerHTML +=
    '<div id="' + typingId + '" class="chat-msg chat-msg-ai">' +
    '<div class="typing-indicator">&#x1F914; AI 正在思考<span>.</span><span>.</span><span>.</span></div>' +
    '</div>';
  chatArea.scrollTop = chatArea.scrollHeight;

  // 获取筛选条件
  var dateFromEl = document.querySelector('input[name="date_from"]');
  var dateToEl = document.querySelector('input[name="date_to"]');
  var sectionEl = document.querySelector('#tab-search select[name="section"]');
  var dateFrom = dateFromEl ? dateFromEl.value : '';
  var dateTo = dateToEl ? dateToEl.value : '';
  var section = sectionEl ? sectionEl.value : '';

  // 调用流式 API
  fetch('/api/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json', 'X-Access-Key': accessKey},
    body: JSON.stringify({
      question: question,
      date_from: dateFrom || null,
      date_to: dateTo || null,
      section: section || null,
      history: chatHistory,
    }),
  }).then(function(response) {
    var typingEl = document.getElementById(typingId);
    if (!typingEl) return;

    // V0.2: Agent steps container + answer bubble
    typingEl.innerHTML =
      '<div class="agent-steps" id="' + typingId + '-steps"></div>' +
      '<div class="chat-bubble" id="' + typingId + '-content" style="display:none"></div>';
    var stepsEl = document.getElementById(typingId + '-steps');
    var contentEl = document.getElementById(typingId + '-content');
    var answerStarted = false;

    var reader = response.body.getReader();
    var decoder = new TextDecoder('utf-8');
    var fullAnswer = '';
    var metadata = null;
    var lineBuffer = '';
    var streamDone = false;
    var searchCount = 0;

    function addStep(type, html) {
      searchCount++;
      var div = document.createElement('div');
      div.className = 'agent-step agent-step-' + type;
      div.innerHTML = html;
      stepsEl.appendChild(div);
      chatArea.scrollTop = chatArea.scrollHeight;
    }

    function processLine(line) {
      if (line.slice(0, 6) !== 'data: ') return;
      var data = line.slice(6);
      if (data === '[DONE]') { streamDone = true; return; }
      try {
        var parsed = JSON.parse(data);
        if (parsed.type === 'thinking') {
          // 深度思考内容 — 折叠展示
          var thinkDiv = document.createElement('div');
          thinkDiv.className = 'agent-step agent-step-status';
          thinkDiv.style.display = 'block';
          thinkDiv.innerHTML =
            '<div style="font-size:12px;color:#0369A1;margin-bottom:4px;font-weight:600">' +
            '🧠 深度分析</div>' +
            '<div style="font-size:12px;color:#333;line-height:1.6;max-height:250px;overflow-y:auto;white-space:pre-wrap">' +
            escapeHtml(parsed.text) + '</div>';
          stepsEl.appendChild(thinkDiv);
          chatArea.scrollTop = chatArea.scrollHeight;
        } else if (parsed.type === 'status') {
          // 状态提示
          addStep('status',
            '<span class="agent-step-icon">ℹ️</span>' +
            '<span class="agent-step-content">' + escapeHtml(parsed.text) + '</span>');
        } else if (parsed.type === 'tool_call') {
          var q = escapeHtml(parsed.query);
          var timeInfo = '';
          if (parsed.date_from || parsed.date_to) {
            timeInfo = ' <span style="color:#9CA3AF">[' +
              (parsed.date_from || '...') + ' ~ ' + (parsed.date_to || '...') + ']</span>';
          }
          addStep('search',
            '<span class="agent-step-icon">🔍</span>' +
            '<span class="agent-step-content">正在检索: ' +
            '<span class="agent-step-query">' + q + '</span>' + timeInfo + '</span>');
        } else if (parsed.type === 'tool_result') {
          var sid = parsed.search_id;
          var n = parsed.count;
          var total = parsed.total_count;
          var method = parsed.method === 'hybrid' ? 'BGE-M3混合' : 'TF-IDF';
          addStep('result',
            '<span class="agent-step-icon">📄</span>' +
            '<span class="agent-step-content">找到 <b>' + n + '</b> 篇' +
            (parsed.new_count !== undefined && parsed.new_count < n ?
              '（新增 ' + parsed.new_count + ' 篇，累计 ' + total + ' 篇）' :
              '（累计 ' + total + ' 篇）') +
            ' <span style="color:#9CA3AF;font-size:11px">[' + method + ']</span></span>');
        } else if (parsed.type === 'chunk') {
          if (!answerStarted) {
            contentEl.style.display = '';
            answerStarted = true;
          }
          fullAnswer += parsed.text;
          contentEl.textContent = fullAnswer;
          chatArea.scrollTop = chatArea.scrollHeight;
        } else if (parsed.type === 'meta') {
          metadata = parsed;
        } else if (parsed.type === 'error') {
          fullAnswer = 'ERROR: ' + (parsed.message || 'unknown');
          contentEl.textContent = fullAnswer;
          contentEl.style.display = '';
          streamDone = true;
        }
      } catch(e) {}
    }

    function readStream() {
      reader.read().then(function(result) {
        if (result.done) {
          if (lineBuffer.trim()) processLine(lineBuffer.trim());
          finishAnswer(typingEl, fullAnswer, metadata);
          isAsking = false;
          submitBtn.disabled = false;
          chatHistory.push({role: 'user', content: question});
          chatHistory.push({role: 'assistant', content: fullAnswer});
          if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);
          return;
        }
        var chunk = decoder.decode(result.value, {stream: true});
        lineBuffer += chunk;
        var lines = lineBuffer.split('\n');
        lineBuffer = lines.pop();
        for (var i = 0; i < lines.length; i++) {
          processLine(lines[i]);
        }
        if (streamDone) {
          finishAnswer(typingEl, fullAnswer, metadata);
          isAsking = false;
          submitBtn.disabled = false;
          chatHistory.push({role: 'user', content: question});
          chatHistory.push({role: 'assistant', content: fullAnswer});
          if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);
          return;
        }
        chatArea.scrollTop = chatArea.scrollHeight;
        readStream();
      }).catch(function(err) {
        contentEl.textContent = 'Request failed: ' + err.message;
        isAsking = false;
        submitBtn.disabled = false;
      });
    }
    readStream();

  }).catch(function(err) {
    var typingEl = document.getElementById(typingId);
    if (typingEl) {
      typingEl.innerHTML = '<div class="chat-bubble">&#x274C; 请求失败: ' + escapeHtml(err.message) + '</div>';
    }
    isAsking = false;
    submitBtn.disabled = false;
  });
}

function finishAnswer(typingEl, fullAnswer, metadata) {
  var contentEl = typingEl.querySelector('.chat-bubble');
  if (!contentEl) return;

  // V0.2: Markdown 渲染
  contentEl.innerHTML = renderMarkdown(fullAnswer);

  // V0.2: Agent 搜索统计
  if (metadata && metadata.total_searches) {
    var summaryDiv = document.createElement('div');
    summaryDiv.className = 'agent-step-summary';
    summaryDiv.innerHTML = '📊 共搜索 <b>' + metadata.total_searches + '</b> 次，' +
      '引用 <b>' + metadata.total_unique_articles + '</b> 篇文章' +
      (metadata.search_summary ? ' | 关键词: ' +
        metadata.search_summary.map(function(s){return '"' + escapeHtml(s.query) + '"'}).join(', ') : '');
    typingEl.appendChild(summaryDiv);
  }

  // 添加参考文章
  if (metadata && metadata.articles && metadata.articles.length > 0) {
    var refsHtml = '';
    for (var i = 0; i < metadata.articles.length; i++) {
      var a = metadata.articles[i];
      refsHtml += '<div class="ai-ref-item">&#x1F4CC; [' + (i+1) + '] ' +
	        a.date + ' | ' + a.section + ' | ';
	      if (a.source_url) {
	        refsHtml += '<a href="' + a.source_url + '" target="_blank" rel="noopener" class="ai-ref-link">' +
	          '&#x300A;' + escapeHtml(a.title) + '&#x300B; &#x2197;</a>';
	      } else {
	        refsHtml += '&#x300A;' + escapeHtml(a.title) + '&#x300B;';
	      }
	      refsHtml += '</div>'
    }
    var refsDiv = document.createElement('div');
    refsDiv.className = 'ai-refs';
    refsDiv.innerHTML = '<details><summary>&#x1F4DA; 参考了 ' + metadata.articles.length + ' 篇文章</summary>' + refsHtml + '</details>';
    typingEl.appendChild(refsDiv);
  }

  // 添加复制按钮
  var copyBtn = document.createElement('button');
  copyBtn.className = 'copy-btn';
  copyBtn.textContent = '&#x1F4CB; 复制回答';
  copyBtn.onclick = function() {
    navigator.clipboard.writeText(fullAnswer).then(function() {
      copyBtn.textContent = '&#x2705; 已复制';
      setTimeout(function() { copyBtn.textContent = '&#x1F4CB; 复制回答'; }, 2000);
    });
  };
  typingEl.appendChild(copyBtn);
}

// V0.2: 轻量 Markdown → HTML 渲染
function renderMarkdown(text) {
  // 先转义 HTML
  var div = document.createElement('div');
  div.textContent = text;
  var html = div.innerHTML;

  var lines = html.split('\n');
  var result = '';
  var inList = false;
  var listType = '';  // 'ul' or 'ol'

  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];

    // 空行 → 关闭列表
    if (/^\s*$/.test(line)) {
      if (inList) { result += '</' + listType + '>'; inList = false; listType = ''; }
      continue;
    }

    // ### 三级标题
    var h3 = line.match(/^### (.+)/);
    if (h3) {
      if (inList) { result += '</' + listType + '>'; inList = false; listType = ''; }
      result += '<h3 style="font-size:16px;margin:12px 0 6px;font-weight:700">' + h3[1] + '</h3>';
      continue;
    }

    // ## 二级标题
    var h2 = line.match(/^## (.+)/);
    if (h2) {
      if (inList) { result += '</' + listType + '>'; inList = false; listType = ''; }
      result += '<h3 style="font-size:17px;margin:14px 0 8px;font-weight:700">' + h2[1] + '</h3>';
      continue;
    }

    // # 一级标题
    var h1 = line.match(/^# (.+)/);
    if (h1) {
      if (inList) { result += '</' + listType + '>'; inList = false; listType = ''; }
      result += '<h2 style="font-size:19px;margin:16px 0 10px;font-weight:700">' + h1[1] + '</h2>';
      continue;
    }

    // 有序列表
    var ol = line.match(/^(\d+)\.\s(.+)/);
    if (ol) {
      if (!inList || listType !== 'ol') {
        if (inList) result += '</' + listType + '>';
        result += '<ol style="margin:4px 0;padding-left:20px">';
        inList = true; listType = 'ol';
      }
      result += '<li>' + ol[2] + '</li>';
      continue;
    }

    // 无序列表
    var ul = line.match(/^[-*]\s(.+)/);
    if (ul) {
      if (!inList || listType !== 'ul') {
        if (inList) result += '</' + listType + '>';
        result += '<ul style="margin:4px 0;padding-left:20px">';
        inList = true; listType = 'ul';
      }
      result += '<li>' + ul[1] + '</li>';
      continue;
    }

    // 普通行 → 关闭列表
    if (inList) { result += '</' + listType + '>'; inList = false; listType = ''; }

    // **bold**
    line = line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    result += '<p style="margin:4px 0">' + line + '</p>';
  }

  // 关闭未闭合的列表
  if (inList) { result += '</' + listType + '>'; }

  return result;
}

function escapeHtml(text) {
  var div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Enter &#x53D1;&#x9001;&#xFF08;Shift+Enter &#x6362;&#x884C;&#xFF09;
document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('qaInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      askAI();
    }
  });
  // 恢复上次的 tab
  var savedTab = sessionStorage.getItem('tjrb_active_tab');
  if (savedTab) switchTab(savedTab);
});

// 保存 tab 状态
var origSwitchTab = switchTab;
switchTab = function(tab) {
  origSwitchTab(tab);
  sessionStorage.setItem('tjrb_active_tab', tab);
};
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
class SearchHandler(BaseHTTPRequestHandler):

    def _auth_ok(self):
        """检查访问密钥。未设置密钥时直接放行。"""
        if not ACCESS_PASSWORD:
            return True
        key = self.headers.get("X-Access-Key", "")
        return key == ACCESS_PASSWORD

    def _reject_auth(self):
        self.send_response(403)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps({"error": "invalid access key"}, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        raw_path = self.path.encode("latin-1").decode("utf-8")
        parsed = urllib.parse.urlparse(raw_path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/search":
            self._serve_search_page(params)
        elif path == "/api/search":
            if not self._auth_ok():
                return self._reject_auth()
            self._serve_search_api(params)
        elif path == "/stats":
            self._serve_stats()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        raw_path = self.path.encode("latin-1").decode("utf-8")
        if raw_path == "/api/ask":
            if not self._auth_ok():
                return self._reject_auth()
            self._serve_ask_api()
        elif raw_path == "/reload":
            if not self._auth_ok():
                return self._reject_auth()
            self._serve_reload()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    # ---------- 页面渲染 ----------
    def _serve_search_page(self, params):
        query = params.get("q", [""])[0].strip()
        date_from = params.get("date_from", [""])[0]
        date_to = params.get("date_to", [""])[0]
        section = params.get("section", [""])[0]
        sort_by = params.get("sort", ["relevance"])[0]
        title_only = params.get("title_only", [""])[0] == "1"

        # 搜索
        results_html = ""
        if query and vectorizer is not None:
            results = tfidf_search(
                query, top_k=20,
                date_from=date_from if date_from else None,
                date_to=date_to if date_to else None,
                section_filter=section if section else None,
                sort_by=sort_by,
                title_only=title_only,
                expand=expand,
            )
            if results:
                results_html = (
                    '<div class="stats-bar">找到 ' + str(len(results)) + ' 条结果</div>'
                )
                card_id = 0
                for r in results:
                    card_id += 1
                    cid = "card-" + str(card_id)
                    score_pct = str(int(r["score"] * 100)) + "%"
                    full_content = highlight(r["content"], query)
                    preview = highlight(r["content"][:300], query)
                    source_url = r.get("source_url", "")
                    is_truncated = len(r["content"]) > 300
                    results_html += (
                        '<div class="result-card">'
                        '<div class="result-header">'
                        '<a class="result-title" href="' + source_url + '" target="_blank" rel="noopener">' + r['title'] + '</a>'
                        '<span class="result-score">相关度 ' + score_pct + '</span>'
                        '</div>'
                        '<div class="result-meta">'
                        '<span>&#x1F4C5; ' + r['date'] + '</span>'
                        '<span>&#x1F4F0; ' + r['section'] + '</span>'
                        '</div>'
                        '<div class="result-content" id="' + cid + '-preview">' + preview + '</div>'
                        '<div class="result-content result-full" id="' + cid + '-full" style="display:none">' + full_content + '</div>'
                    )
                    if is_truncated:
                        results_html += (
                            '<a class="expand-link" id="' + cid + '-btn" '
                            'onclick="var p=document.getElementById(\'' + cid + '-preview\');'
                            'var f=document.getElementById(\'' + cid + '-full\');'
                            'var b=document.getElementById(\'' + cid + '-btn\');'
                            'if(f.style.display===\'none\'){p.style.display=\'none\';f.style.display=\'block\';b.textContent=\'\\u25B2 \\u6536\\u8D77\\u5168\\u6587\';}'
                            'else{f.style.display=\'none\';p.style.display=\'block\';b.textContent=\'\\u25BC \\u5C55\\u5F00\\u5168\\u6587 (' + str(len(r["content"])) + '\\u5B57)\';}"'
                            'href="javascript:void(0)">&#x25BC; 展开全文 (' + str(len(r["content"])) + '字)</a>'
                        )
                    if source_url:
                        results_html += (
                            '<a class="result-link" href="' + source_url + '" target="_blank" rel="noopener">'
                            '&#x1F517; 查看原文 &rarr;</a>'
                        )
                    results_html += '</div>'
            else:
                results_html = (
                    '<div class="empty">'
                    '<h2>未找到相关文章</h2>'
                    '<p>尝试更换关键词或放宽筛选条件</p></div>'
                )
        elif not query:
            results_html = (
                '<div class="empty">'
                '<h2>开始搜索</h2>'
                '<p>输入关键词，搜索近一年天津日报的全部文章</p></div>'
            )
        else:
            results_html = (
                '<div class="empty">'
                '<h2>知识库尚未构建</h2>'
                '<p>请先运行 python3 build_kb.py --build</p></div>'
            )

        # 版面选项
        sections_set = set()
        if metadata:
            for a in metadata:
                sections_set.add(a["section"])
        section_options = "".join(
            '<option value="' + s + '" ' + ('selected' if section == s else '') + '>' + s + '</option>'
            for s in sorted(sections_set)
        )

        # KB 信息
        kb_info = ""
        if metadata:
            kb_info = str(len(metadata)) + " 篇文章 | "
            kb_info += metadata[0]["date"] + " ~ " + metadata[-1]["date"]

        query_esc = query.replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")

        html = (HTML_PAGE
            .replace("__QUERY_ESC__", query_esc)
            .replace("__DATE_FROM__", date_from)
            .replace("__DATE_TO__", date_to)
            .replace("__SECTION_OPTIONS__", section_options)
            .replace("__SORT_RELEVANCE_SEL__", "selected" if sort_by == "relevance" else "")
            .replace("__SORT_DATE_SEL__", "selected" if sort_by == "date_desc" else "")
            .replace("__TITLE_ONLY_CHECKED__", "checked" if title_only else "")
            .replace("__EXPAND_CHECKED__", "checked" if expand else "")
            .replace("__RESULTS__", results_html)
            .replace("__KB_INFO__", kb_info)
            .replace("__MODEL_NAME__", DEEPSEEK_MODEL)
            .replace("__ACCESS_PASSWORD__", ACCESS_PASSWORD))

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    # ---------- 搜索 API ----------
    def _serve_search_api(self, params):
        query = params.get("q", [""])[0].strip()
        if not query:
            self._send_json({"results": [], "error": "No query"})
            return
        sort_by = params.get("sort", ["relevance"])[0]
        title_only = params.get("title_only", [""])[0] == "1"
        expand = params.get("expand", [""])[0] == "1"
        results = tfidf_search(query, top_k=10, sort_by=sort_by,
                               title_only=title_only, expand=expand)
        self._send_json({
            "results": [
                {
                    "title": r["title"],
                    "date": r["date"],
                    "section": r["section"],
                    "summary": r["summary"],
                    "score": r["score"],
                    "source_url": r.get("source_url", ""),
                }
                for r in results
            ],
            "total": len(results),
        })

    # ---------- AI 问答 API (Agent 流式 SSE, V0.2) ----------
    def _serve_ask_api(self):
        # 读取请求体
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"})
            return

        question = data.get("question", "").strip()
        date_from = data.get("date_from")
        date_to = data.get("date_to")
        section = data.get("section")
        history = data.get("history", [])

        if not question:
            self._send_json({"error": "No question"})
            return

        # 流式响应 (SSE)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            # V0.2: 使用 Agent 引擎（模型自主搜索策略）
            for event in agent_ask(
                question,
                history=history,
                date_from=date_from if date_from else None,
                date_to=date_to if date_to else None,
                section_filter=section if section else None,
            ):
                payload = json.dumps(event, ensure_ascii=False)
                self.wfile.write(("data: " + payload + "\n\n").encode("utf-8"))
                self.wfile.flush()

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception as e:
            error_payload = json.dumps(
                {"type": "error", "message": str(e)}, ensure_ascii=False
            )
            self.wfile.write(("data: " + error_payload + "\n\n").encode("utf-8"))
            self.wfile.flush()

    # ---------- 热加载 ----------
    def _serve_reload(self):
        """重新加载 TF-IDF 索引和 RAG 知识库，无需重启服务"""
        global vectorizer, tfidf_matrix, metadata
        ok_tfidf = load_tfidf_kb()
        ok_rag = rag_load_kb(force=True)
        if ok_tfidf and ok_rag:
            stats = kb_stats()
            self._send_json({
                "status": "ok",
                "total_articles": len(metadata),
                "date_range": metadata[0]["date"] + " ~ " + metadata[-1]["date"],
                "rag_ready": True,
            })
            sys.stderr.write(
                "[" + datetime.now().strftime('%H:%M:%S') + "] "
                "🔄 知识库已热加载: " + str(len(metadata)) + " 篇\n"
            )
        else:
            self._send_json({"status": "error", "message": "知识库加载失败"})

    # ---------- 统计 ----------
    def _serve_stats(self):
        if metadata is None:
            self._send_json({"error": "KB not loaded"})
            return
        from collections import Counter
        sections = Counter(a["section"] for a in metadata)
        self._send_json({
            "total_articles": len(metadata),
            "date_range": metadata[0]["date"] + " ~ " + metadata[-1]["date"],
            "sections": dict(sections.most_common()),
            "rag_ready": rag_ready(),
            "rag_model": DEEPSEEK_MODEL,
        })

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        sys.stderr.write("[" + datetime.now().strftime('%H:%M:%S') + "] " + str(args[0]) + "\n")


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 55)
    print("  天津日报知识库 —— 搜索 + AI 智能问答")
    print("=" * 55)

    # 加载 TF-IDF
    if not load_tfidf_kb():
        print("\n  ⚠ 知识库未找到！请先运行: python3 build_kb.py --build")
    else:
        print("  ✅ 知识库已加载: " + str(len(metadata)) + " 篇文章")

    # 加载 RAG
    if rag_load_kb():
        stats = kb_stats()
        print("  ✅ RAG 引擎就绪: " + str(stats["total_articles"]) + " 篇")

    print("\n  🌐 访问地址: http://localhost:" + str(PORT))
    print("  🔍 关键词搜索: http://localhost:" + str(PORT) + "/search")
    print("  🤖 AI 智能问答: http://localhost:" + str(PORT) + " (页面 Tab)")
    print("  📡 API 接口:")
    print("     GET  /api/search?q=关键词         — 关键词搜索")
    print("     POST /api/ask                      — AI 问答 (SSE流式)")
    print("     GET  /stats                        — 统计信息")
    print("  🧠 AI 模型: " + DEEPSEEK_MODEL)

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("\n  ⚠ 未设置 DEEPSEEK_API_KEY，AI 问答功能不可用")
        print("  请执行: export DEEPSEEK_API_KEY='sk-...'")

    print("\n  按 Ctrl+C 停止服务\n" + "=" * 55 + "\n")

    server = HTTPServer((HOST, PORT), SearchHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
        server.shutdown()


if __name__ == "__main__":
    main()
