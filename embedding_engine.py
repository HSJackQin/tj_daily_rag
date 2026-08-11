#!/usr/bin/env python3
"""
BGE-M3 混合检索引擎
- 稠密向量 (1024维): 语义理解，"智慧港口" ↔ "智能码头"
- 稀疏/词法向量: 关键词精确匹配，类似 TF-IDF 但由模型学习
- 混合检索: DENSE_WEIGHT * dense + SPARSE_WEIGHT * sparse
"""

import json
import os
import time

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tjrb_kb")

# 混合权重（可通过环境变量调整）
DENSE_WEIGHT = float(os.environ.get("HYBRID_DENSE_WEIGHT", "0.7"))
SPARSE_WEIGHT = float(os.environ.get("HYBRID_SPARSE_WEIGHT", "0.3"))

# 全局状态
_model = None
_dense_embeddings = None   # (N, 1024) float16
_sparse_matrix = None       # CSR (N, vocab_size) float32
_metadata = None
_loaded = False


def _get_model():
    """懒加载 BGE-M3 模型（首次调用时下载/加载）"""
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        print("  ⏳ 加载 BGE-M3 模型 (BAAI/bge-m3)...")
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        print("  ✅ BGE-M3 模型就绪")
    return _model


def load_embedding_index() -> bool:
    """加载预构建的嵌入索引。返回 True 表示加载成功。"""
    global _dense_embeddings, _sparse_matrix, _metadata, _loaded
    if _loaded:
        return True

    dense_path = os.path.join(KB_DIR, "dense_embeddings.npy")
    sparse_path = os.path.join(KB_DIR, "sparse_matrix.npz")
    meta_path = os.path.join(KB_DIR, "metadata.json")

    if not all(os.path.exists(p) for p in [dense_path, sparse_path, meta_path]):
        return False

    _dense_embeddings = np.load(dense_path)
    _sparse_matrix = load_npz(sparse_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        _metadata = json.load(f)
    _loaded = True
    return True


def build_embedding_index(articles: list, kb_dir: str):
    """
    构建 BGE-M3 混合索引并保存到磁盘。

    参数:
        articles: 文章列表，每篇需含 search_text 字段
        kb_dir:  索引保存目录

    返回:
        (dense_embeddings, sparse_matrix)
    """
    model = _get_model()
    docs = [a["search_text"] for a in articles]
    n_docs = len(docs)
    batch_size = 8  # CPU 推理用较小的 batch

    print(f"\n🔨 构建 BGE-M3 嵌入索引...")
    print(f"  文章数: {n_docs}")
    print(f"  混合权重: dense={DENSE_WEIGHT}, sparse={SPARSE_WEIGHT}")

    all_dense = []
    all_sparse_weights = []
    t_start = time.time()

    for i in range(0, n_docs, batch_size):
        batch = docs[i : i + batch_size]
        output = model.encode(
            batch,
            return_dense=True,
            return_sparse=True,
            batch_size=batch_size,
            max_length=8192,
        )
        all_dense.append(output["dense_vecs"])
        all_sparse_weights.extend(output["lexical_weights"])

        done = min(i + batch_size, n_docs)
        if done % 200 == 0 or done == n_docs:
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_docs - done) / rate if rate > 0 else 0
            print(f"    已编码 {done}/{n_docs} 篇 "
                  f"({done/n_docs*100:.0f}%, {rate:.0f} 篇/秒, "
                  f"预计剩余 {eta:.0f}s)")

    # 拼接稠密向量
    dense_embeddings = np.concatenate(all_dense, axis=0).astype(np.float16)
    print(f"  稠密向量: shape={dense_embeddings.shape}, "
          f"dtype={dense_embeddings.dtype}, "
          f"内存={dense_embeddings.nbytes/1024/1024:.0f}MB")

    # 构建稀疏矩阵 (CSR)
    # BGE-M3 使用 XLM-Roberta tokenizer，vocab_size = 250002
    indices = []
    indptr = [0]
    data = []

    for weights in all_sparse_weights:
        if weights:
            sorted_items = sorted(weights.items())  # 按 token_id 排序
            for tid, val in sorted_items:
                indices.append(tid)
                data.append(val)
        indptr.append(len(indices))

    # 根据实际出现的最大 token_id 确定 matrix 宽度
    max_tid = max(indices) if indices else 250001
    vocab_size = max(max_tid + 1, 250002)

    sparse_matrix = csr_matrix(
        (data, indices, indptr),
        shape=(n_docs, vocab_size),
        dtype=np.float32,
    )
    print(f"  稀疏矩阵: shape={sparse_matrix.shape}, "
          f"nnz={sparse_matrix.nnz}, "
          f"密度={sparse_matrix.nnz/(n_docs*vocab_size)*100:.3f}%")

    # 保存
    os.makedirs(kb_dir, exist_ok=True)
    np.save(os.path.join(kb_dir, "dense_embeddings.npy"), dense_embeddings)
    save_npz(os.path.join(kb_dir, "sparse_matrix.npz"), sparse_matrix)

    elapsed = time.time() - t_start
    print(f"  💾 嵌入索引已保存到 {kb_dir} (耗时 {elapsed:.0f}s)")

    # 设置全局状态
    global _dense_embeddings, _sparse_matrix, _metadata, _loaded
    _dense_embeddings = dense_embeddings
    _sparse_matrix = sparse_matrix
    _metadata = articles
    _loaded = True

    return dense_embeddings, sparse_matrix


def hybrid_search(
    query: str,
    top_k: int = 20,
    date_from: str | None = None,
    date_to: str | None = None,
    section_filter: str | None = None,
) -> list[dict]:
    """
    混合检索：稠密语义 + 稀疏关键词。

    返回每篇文章附带:
        score:       混合得分
        dense_score: 语义相似度
        sparse_score:关键词匹配度
    """
    if not _loaded:
        return []

    model = _get_model()

    # 编码查询
    output = model.encode(
        [query],
        return_dense=True,
        return_sparse=True,
        max_length=8192,
    )
    query_dense = output["dense_vecs"][0].astype(np.float32)  # (1024,)
    query_sparse_weights = output["lexical_weights"][0]        # {token_id: weight}

    n_docs = len(_metadata)

    # ── 稠密相似度 ──
    if _dense_embeddings.dtype == np.float16:
        dense_scores = cosine_similarity(
            query_dense.reshape(1, -1), _dense_embeddings.astype(np.float32)
        )[0]
    else:
        dense_scores = cosine_similarity(
            query_dense.reshape(1, -1), _dense_embeddings
        )[0]

    # ── 稀疏相似度 ──
    sparse_scores = np.zeros(n_docs, dtype=np.float32)
    if query_sparse_weights:
        token_ids = np.array(list(query_sparse_weights.keys()), dtype=np.int32)
        token_weights = np.array(
            list(query_sparse_weights.values()), dtype=np.float32
        )
        # 构建查询稀疏向量 → 矩阵乘法获取所有文档得分
        query_sparse_vec = csr_matrix(
            (token_weights, (np.zeros_like(token_ids), token_ids)),
            shape=(1, _sparse_matrix.shape[1]),
            dtype=np.float32,
        )
        sparse_scores = query_sparse_vec.dot(_sparse_matrix.T).toarray()[0]

    # ── 归一化 & 加权融合 ──
    if dense_scores.max() > 0:
        dense_scores = dense_scores / dense_scores.max()
    if sparse_scores.max() > 0:
        sparse_scores = sparse_scores / sparse_scores.max()

    hybrid_scores = DENSE_WEIGHT * dense_scores + SPARSE_WEIGHT * sparse_scores

    # ── 排序 & 过滤 ──
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
        print("❌ 嵌入索引未构建，请先运行 build_kb.py --build")
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
