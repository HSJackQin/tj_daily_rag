#!/usr/bin/env python3
"""
一键启动脚本：自动检测爬虫完成后，构建知识库并启动搜索服务 (含 AI 问答)
"""
import os
import subprocess
import time
import sys
import json
import glob

WORKSPACE = os.path.dirname(os.path.abspath(__file__))


def count_progress():
    files = glob.glob(os.path.join(WORKSPACE, "tjrb_data/**/*.json"), recursive=True)
    return len(files)


def main():
    print("=" * 55)
    print("  天津日报知识库 — 一键部署 (搜索 + AI 问答)")
    print("=" * 55)

    # 检查 API Key
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("\n  ⚠ 未设置 DEEPSEEK_API_KEY 环境变量")
        print("  AI 智能问答功能将不可用，仅提供关键词搜索")
        print("  如需启用 AI 问答，请先执行:")
        print("    export DEEPSEEK_API_KEY='sk-...'")
        print()

    # Step 1: 检查爬虫状态
    print("\n[1/3] 检查爬虫状态...")
    result = subprocess.run(
        ["pgrep", "-f", "tjrb_crawler.py"],
        capture_output=True, text=True
    )
    crawler_running = bool(result.stdout.strip())

    current = count_progress()
    print(f"  当前已采集: {current} 天")

    if crawler_running:
        print(f"  爬虫运行中，等待完成...")
        last_count = current
        stall_count = 0
        while True:
            time.sleep(30)
            current = count_progress()
            result = subprocess.run(
                ["pgrep", "-f", "tjrb_crawler.py"],
                capture_output=True, text=True
            )

            if current != last_count:
                print(f"  已采集: {current} 天 (新增 {current - last_count} 天)")
                last_count = current
                stall_count = 0
            else:
                stall_count += 1

            if not result.stdout.strip():
                # 爬虫进程已退出
                print(f"  爬虫已完成！共采集 {current} 天")
                break

            if stall_count > 10:
                print(f"  爬虫可能已停止（{stall_count*30}秒无新数据），检查...")
                if not result.stdout.strip():
                    break
    else:
        print(f"  爬虫未在运行，使用已有数据 ({current} 天)")

    # Step 2: 构建知识库
    print(f"\n[2/3] 构建知识库...")
    result = subprocess.run(
        [sys.executable, os.path.join(WORKSPACE, "build_kb.py"), "--build"],
        cwd=WORKSPACE,
        capture_output=False
    )

    if result.returncode != 0:
        print("  ❌ 知识库构建失败，请检查错误信息")
        sys.exit(1)

    print("  ✅ 知识库构建完成")

    # Step 3: 启动搜索服务 (含 AI 问答)
    print(f"\n[3/3] 启动搜索服务 (含 AI 问答)...")
    print("=" * 55)
    os.execv(sys.executable, [
        sys.executable,
        os.path.join(WORKSPACE, "kb_server.py")
    ])


if __name__ == '__main__':
    main()
