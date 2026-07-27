#!/usr/bin/env python3
"""
每日自动更新脚本
1. 爬取当天天津日报文章
2. 重建 TF-IDF 知识库
3. 通知 kb_server 热加载新索引

配合 cron 使用，例如每天 8:30 执行：
  30 8 * * * cd /home/qinjinqi/ai_project/tj_daily_rag && python3 daily_update.py >> logs/daily.log 2>&1
"""

import os
import sys
import subprocess
import json
import time
import urllib.request
from datetime import datetime

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(WORKSPACE, "logs")
SERVER_PORT = 8699


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    ensure_dir(LOG_DIR)
    log("=" * 50)
    log("每日更新开始")

    # Step 1: 爬取当天文章
    log("[1/3] 爬取当天文章...")
    result = subprocess.run(
        [sys.executable, os.path.join(WORKSPACE, "tjrb_crawler.py"), "--today"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        log(f"  ❌ 爬虫失败: {result.stderr[-500:]}")
    else:
        log(f"  ✅ 爬虫完成")

    # Step 2: 重建知识库
    log("[2/3] 重建知识库...")
    result = subprocess.run(
        [sys.executable, os.path.join(WORKSPACE, "build_kb.py"), "--build"],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        log(f"  ❌ 构建失败: {result.stderr[-500:]}")
    else:
        log(f"  ✅ 知识库重建完成")

    # Step 3: 通知 server 热加载
    log("[3/3] 通知服务端重新加载索引...")
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{SERVER_PORT}/reload",
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read())
        log(f"  ✅ 服务端响应: {data}")
    except Exception as e:
        log(f"  ⚠ 服务端通知失败 (server 可能未运行): {e}")

    log("每日更新完成")
    log("=" * 50)


if __name__ == "__main__":
    main()
