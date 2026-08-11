#!/usr/bin/env python3
"""
BGE + TF-IDF 混合检索引擎 (V0.2)
- 稠密向量 (512维): bge-small-zh-v1.5 语义理解
- 稀疏向量: 现有 jieba TF-IDF 关键词匹配
- 混合检索: DENSE_WEIGHT * dense + SPARSE_WEIGHT * tfidf
"""

import json
import os
import pickle
import time

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tjrb_kb")

# 混合权重（可通过环境变量调整）
DENSE_WEIGHT = float(os.environ.get("HYBRID_DENSE_WEIGHT", "0.7"))
SPARSE_WEIGHT = float(os.environ.get("HYBRID_SPARSE_WEIGHT", "0.3"))

# bge-small 模型路径
MODEL_PATH = os.environ.get(
    "BGE_MODEL_PATH",
    "/home/qinjinqi/.cache/modelscope/models/BAAI--bge-small-zh-v1.5/snapshots/master"
)

# 全局状态
_model = None
_dense_embeddings = None   # (N, 512) float16
_metadata = None
_loaded = False

# TF-IDF（复用现有索引）
_tfidf_vectorizer = None
_tfidf_matrix = None


def _get_model():
    """懒加载 bge-small 模型"""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        model_path = MODEL_PATH if os.path.isdir(MODEL_PATH) else "BAAI/bge-small-zh-v1.5"
        print(f"  ⏳ 加载 BGE 模型: {model_path}")
        _model = SentenceTransformer(model_path)
        print("  ✅ BGE 模型就绪")
    return _model


def _load_tfidf():
    """加载 TF-IDF 索引作为稀疏匹配"""
    global _tfidf_vectorizer, _tfidf_matrix
    if _tfidf_vectorizer is not None:
        return True

    v_path = os.path.join(KB_DIR, "vectorizer.pkl")
    m_path = os.path.join(KB_DIR, "tfidf_matrix.npz")

    if not all(os.path.exists(p) for p in [v_path, m_path]):
        return False

    with open(v_path, "rb") as f:
        _tfidf_vectorizer = pickle.load(f)
    _tfidf_matrix = load_npz(m_path)
    return True


def load_embedding_index() -> bool:
    """加载预构建的嵌入索引和 TF-IDF。返回 True 表示加载成功。"""
    global _dense_embeddings, _metadata, _loaded
    if _loaded:
        return True

    dense_path = os.path.join(KB_DIR, "dense_embeddings.npy")
    meta_path = os.path.join(KB_DIR, "metadata.json")

    if not all(os.path.exists(p) for p in [dense_path, meta_path]):
        return False

    _dense_embeddings = np.load(dense_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        _metadata = json.load(f)

    # TF-IDF 可选（没有也能跑，但只有语义搜索）
    if not _load_tfidf():
        print("  ⚠ TF-IDF 索引未加载，将使用纯语义搜索")

    _loaded = True
    return True


def build_embedding_index(articles: list, kb_dir: str):
    """
    构建 bge-small 稠密向量索引并保存。

    参数:
        articles: 文章列表，每篇需含 search_text 字段
        kb_dir:  索引保存目录

    返回:
        dense_embeddings: (N, 512) float16 矩阵
    """
    model = _get_model()
    docs = [a["search_text"] for a in articles]
    n_docs = len(docs)

    print(f"\n🔨 构建 BGE 嵌入索引...")
    print(f"  模型: bge-small-zh-v1.5 (512维)")
    print(f"  文章数: {n_docs}")
    print(f"  混合权重: dense={DENSE_WEIGHT}, sparse={SPARSE_WEIGHT}")

    t_start = time.time()

    # 使用 sentence-transformers 批量编码
    dense_embeddings = model.encode(
        docs,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 归一化，直接用点积算相似度
    )

    # 转为 float16 节省存储
    dense_embeddings = dense_embeddings.astype(np.float16)

    elapsed = time.time() - t_start
    print(f"  稠密向量: shape={dense_embeddings.shape}, "
          f"dtype={dense_embeddings.dtype}, "
          f"内存={dense_embeddings.nbytes/1024/1024:.0f}MB")
    print(f"  编码速度: {n_docs/elapsed:.0f} 篇/秒 (总耗时 {elapsed:.0f}s)")

    # 保存
    os.makedirs(kb_dir, exist_ok=True)
    np.save(os.path.join(kb_dir, "dense_embeddings.npy"), dense_embeddings)
    print(f"  💾 嵌入索引已保存到 {kb_dir}")

    # 设置全局状态
    global _dense_embeddings, _metadata, _loaded
    _dense_embeddings = dense_embeddings
    _metadata = articles
    _loaded = True

    # 同时加载 TF-IDF
    if not _load_tfidf():
        print("  ⚠ TF-IDF 索引未找到，混合检索将回退为纯语义搜索")

    return dense_embeddings


def hybrid_search(
    query: str,
    top_k: int = 20,
    date_from: str | None = None,
    date_to: str | None = None,
    section_filter: str | None = None,
) -> list[dict]:
    """
    混合检索：BGE 语义 + TF-IDF 关键词。

    返回每篇文章附带:
        score:       混合得分
        dense_score: 语义相似度
        sparse_score:TF-IDF 关键词匹配度
    """
    if not _loaded:
        return []

    model = _get_model()
    n_docs = len(_metadata)

    # ── 1. 稠密语义相似度 ──
    query_vec = model.encode(
        [query],
        batch_size=1,
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0].astype(np.float32)

    # 已 L2 归一化，点积 = 余弦相似度
    dense_scores = np.dot(
        _dense_embeddings.astype(np.float32), query_vec
    )

    # ── 2. TF-IDF 关键词相似度 ──
    sparse_scores = np.zeros(n_docs, dtype=np.float32)
    if _tfidf_vectorizer is not None and _tfidf_matrix is not None:
        query_tfidf = _tfidf_vectorizer.transform([query])
        sparse_scores = cosine_similarity(query_tfidf, _tfidf_matrix)[0]

    # ── 3. 归一化 & 加权融合 ──
    if dense_scores.max() > 0:
        dense_scores = dense_scores / dense_scores.max()
    if sparse_scores.max() > 0:
        sparse_scores = sparse_scores / sparse_scores.max()

    has_tfidf = _tfidf_vectorizer is not None and sparse_scores.max() > 0
    if has_tfidf:
        hybrid_scores = DENSE_WEIGHT * dense_scores + SPARSE_WEIGHT * sparse_scores
    else:
        hybrid_scores = dense_scores  # 纯语义搜索

    # ── 4. 排序 & 过滤 ──
    ranked = sorted(enumerate(hybrid_scores), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked:
        if score < 0.01:
            continue
        art = dict(_metadata[idx])
        if date_from and art["date"] < date_from:
            continue
        if date_to and art["date"] > date_to:
            continue
        if section_filter and section_filter not in art["section"]:
            continue
        art["score"] = float(score)
        art["dense_score"] = float(dense_scores[idx])
        art["sparse_score"] = float(sparse_scores[idx])
        results.append(art)
        if len(results) >= top_k:
            break

    return results


def is_ready() -> bool:
    return _loaded


# ---------------------------------------------------------------------------
# CLI 测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if not load_embedding_index():
        print("❌ 嵌入索引未构建，请先运行 build_kb.py --build-embedding")
        sys.exit(1)

    query = sys.argv[1] if len(sys.argv) > 1 else "天津港智慧港口建设"
    print(f"\n🔍 混合检索: \"{query}\"\n")

    results = hybrid_search(query, top_k=5)
    for i, r in enumerate(results):
        print(f"[{i+1}] {r['title']}")
        print(f"    📅 {r['date']} | {r['section']}")
        print(f"    📊 hybrid={r['score']:.3f}  "
              f"dense={r['dense_score']:.3f}  "
              f"sparse={r['sparse_score']:.3f}")
        print(f"    📝 {r['summary'][:120]}")
        print()
