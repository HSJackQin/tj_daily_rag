"""共享分词器模块，供 build_kb.py 和 kb_server.py 共同导入"""
import re
import jieba


def jieba_tokenizer(text):
    """jieba 分词器"""
    text = re.sub(r'\s+', ' ', text)
    words = jieba.lcut(text)
    return [w for w in words if len(w.strip()) > 1]