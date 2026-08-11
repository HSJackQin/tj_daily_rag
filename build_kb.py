#!/usr/bin/env python3
"""
天津日报知识库构建脚本
- 合并所有 JSON 数据
- 使用 jieba 分词 + TF-IDF 构建向量索引
- 支持中文语义搜索
"""

import json
import os
import glob
import pickle
import re
import sys
from datetime import datetime
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tjrb_tokenizer import jieba_tokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tjrb_data")
KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tjrb_kb")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_all_articles(data_dir):
    """加载所有 JSON 文件，合并为文章列表"""
    json_files = sorted(glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True))
    print(f"找到 {len(json_files)} 个日期文件")

    articles = []
    for f in json_files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                day_data = json.load(fp)
                for art in day_data:
                    # 构建用于检索的文本
                    search_text = f"{art['title']} {art['subtitle']} {art['content']}"
                    # 构建显示用的摘要
                    summary = art['content'][:200] + '...' if len(art['content']) > 200 else art['content']

                    articles.append({
                        'id': len(articles),
                        'date': art['date'],
                        'section': f"第{art['section_num']:02d}版：{art['section_name']}",
                        'title': art['title'],
                        'subtitle': art.get('subtitle', ''),
                        'content': art['content'],
                        'summary': summary,
                        'search_text': search_text,
                        'source_url': art.get('source_url', ''),
                    })
        except Exception as e:
            print(f"  ⚠ 读取失败: {f} - {e}")

    print(f"共加载 {len(articles)} 篇文章")
    return articles


def build_tfidf_index(articles, kb_dir):
    """构建 TF-IDF 向量索引"""
    print("\n🔨 构建 TF-IDF 索引...")

    # 准备文档
    docs = [a['search_text'] for a in articles]

    # 构建 TF-IDF 向量化器
    vectorizer = TfidfVectorizer(
        tokenizer=jieba_tokenizer,
        max_features=20000,
        max_df=0.9,
        min_df=2,
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(docs)
    print(f"  词汇量: {len(vectorizer.get_feature_names_out())}")
    print(f"  矩阵形状: {tfidf_matrix.shape}")

    # 保存向量化器和矩阵
    ensure_dir(kb_dir)
    with open(os.path.join(kb_dir, "vectorizer.pkl"), 'wb') as f:
        pickle.dump(vectorizer, f)

    # 稀疏矩阵保存
    from scipy.sparse import save_npz
    save_npz(os.path.join(kb_dir, "tfidf_matrix.npz"), tfidf_matrix)

    # 保存文章元数据
    metadata = []
    for a in articles:
        metadata.append({
            'id': a['id'],
            'date': a['date'],
            'section': a['section'],
            'title': a['title'],
            'subtitle': a['subtitle'],
            'content': a['content'],
            'summary': a['summary'],
            'source_url': a['source_url'],
        })

    with open(os.path.join(kb_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"  💾 索引已保存到: {kb_dir}")
    return vectorizer, tfidf_matrix, metadata


def search(query, vectorizer, tfidf_matrix, metadata, top_k=10,
           date_from=None, date_to=None, section_filter=None):
    """
    搜索文章
    - date_from/date_to: 日期范围过滤，格式 '2026-07-01'
    - section_filter: 版面过滤，如 '要闻'
    """
    # 向量化查询
    query_vec = vectorizer.transform([query])

    # 计算相似度
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]

    # 按相似度排序
    ranked = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    # 过滤并返回结果
    results = []
    for idx, score in ranked:
        if score < 0.01:  # 相似度太低，跳过
            continue

        art = metadata[idx]

        # 日期过滤
        if date_from and art['date'] < date_from:
            continue
        if date_to and art['date'] > date_to:
            continue

        # 版面过滤
        if section_filter and section_filter not in art['section']:
            continue

        results.append({
            **art,
            'score': float(score),
        })

        if len(results) >= top_k:
            break

    return results


def print_results(results):
    """格式化打印搜索结果"""
    if not results:
        print("  (无匹配结果)")
        return

    for i, r in enumerate(results):
        print(f"\n  {'─'*60}")
        print(f"  [{i+1}] {r['title']}  相似度: {r['score']:.3f}")
        print(f"  📅 {r['date']} | 📰 {r['section']}")
        print(f"  📝 {r['summary'][:150]}")
        if r['source_url']:
            print(f"  🔗 {r['source_url']}")


def stats(kb_dir):
    """显示知识库统计信息"""
    meta_file = os.path.join(kb_dir, "metadata.json")
    if not os.path.exists(meta_file):
        print("知识库尚未构建。请先运行: python3 build_kb.py --build")
        return

    with open(meta_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"\n{'='*50}")
    print(f"  天津日报知识库统计")
    print(f"{'='*50}")
    print(f"  总文章数: {len(metadata)} 篇")

    # 按日期统计
    dates = defaultdict(int)
    sections = defaultdict(int)
    for a in metadata:
        dates[a['date'][:7]] += 1  # 按月
        sections[a['section']] += 1

    print(f"  日期范围: {metadata[0]['date']} ~ {metadata[-1]['date']}")
    print(f"  月份数: {len(dates)}")
    print(f"\n  按版面统计:")
    for s, c in sorted(sections.items()):
        bar = '█' * (c // 50)
        print(f"    {s}: {c}篇 {bar}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法:")
        print("  python3 build_kb.py --build              # 构建知识库 (TF-IDF + BGE-M3)")
        print("  python3 build_kb.py --build-tfidf        # 仅构建 TF-IDF 索引")
        print("  python3 build_kb.py --build-embedding    # 仅构建 BGE-M3 嵌入索引")
        print("  python3 build_kb.py --search '关键词'     # TF-IDF 搜索")
        print("  python3 build_kb.py --search-hybrid '关键词'  # 混合检索测试")
        print("  python3 build_kb.py --stats              # 统计信息")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == '--build':
        articles = load_all_articles(DATA_DIR)
        if not articles:
            print("❌ 未找到任何数据，请先运行爬虫")
            sys.exit(1)
        build_tfidf_index(articles, KB_DIR)
        print(f"\n✅ TF-IDF 索引构建完成！")

        # 构建 BGE-M3 嵌入索引
        print("\n" + "=" * 50)
        try:
            from embedding_engine import build_embedding_index
            build_embedding_index(articles, KB_DIR)
            print(f"\n✅ BGE-M3 嵌入索引构建完成！")
        except ImportError:
            print("\n⚠ 未安装 FlagEmbedding，跳过 BGE-M3 索引构建")
            print("  安装: pip install FlagEmbedding")

        print(f"\n✅ 知识库构建完成！")

    elif cmd == '--build-tfidf':
        articles = load_all_articles(DATA_DIR)
        if not articles:
            print("❌ 未找到任何数据，请先运行爬虫")
            sys.exit(1)
        build_tfidf_index(articles, KB_DIR)
        print(f"\n✅ TF-IDF 索引构建完成！")

    elif cmd == '--build-embedding':
        articles = load_all_articles(DATA_DIR)
        if not articles:
            print("❌ 未找到任何数据，请先运行爬虫")
            sys.exit(1)
        from embedding_engine import build_embedding_index
        build_embedding_index(articles, KB_DIR)
        print(f"\n✅ BGE-M3 嵌入索引构建完成！")

    elif cmd == '--search':
        if len(sys.argv) < 3:
            print("请提供搜索关键词")
            sys.exit(1)
        query = sys.argv[2]

        # 加载索引
        with open(os.path.join(KB_DIR, "vectorizer.pkl"), 'rb') as f:
            vectorizer = pickle.load(f)
        from scipy.sparse import load_npz
        tfidf_matrix = load_npz(os.path.join(KB_DIR, "tfidf_matrix.npz"))
        with open(os.path.join(KB_DIR, "metadata.json"), 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        results = search(query, vectorizer, tfidf_matrix, metadata, top_k=10)
        print(f"\n🔍 TF-IDF 搜索: \"{query}\"")
        print_results(results)

    elif cmd == '--search-hybrid':
        if len(sys.argv) < 3:
            print("请提供搜索关键词")
            sys.exit(1)
        query = sys.argv[2]
        from embedding_engine import hybrid_search, load_embedding_index

        if not load_embedding_index():
            print("❌ BGE-M3 嵌入索引未构建，请先运行 --build-embedding")
            sys.exit(1)

        results = hybrid_search(query, top_k=10)
        print(f"\n🔍 混合检索 (BGE-M3): \"{query}\"")
        for i, r in enumerate(results):
            print(f"\n  {'─'*60}")
            print(f"  [{i+1}] {r['title']}")
            print(f"  📊 hybrid={r['score']:.3f}  "
                  f"dense={r['dense_score']:.3f}  "
                  f"sparse={r['sparse_score']:.3f}")
            print(f"  📅 {r['date']} | 📰 {r['section']}")
            print(f"  📝 {r['summary'][:150]}")

    elif cmd == '--stats':
        stats(KB_DIR)
        # 同时显示嵌入索引状态
        try:
            from embedding_engine import load_embedding_index
            if load_embedding_index():
                print("  BGE-M3 嵌入索引: ✅ 已就绪")
            else:
                print("  BGE-M3 嵌入索引: ❌ 未构建")
        except Exception:
            print("  BGE-M3 嵌入索引: ⚠ 模块不可用")

    else:
        print(f"未知命令: {cmd}")