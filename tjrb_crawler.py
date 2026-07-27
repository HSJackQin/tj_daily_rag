#!/usr/bin/env python3
"""
天津日报数字报爬虫
从 epaper.tianjinwe.com 采集文章标题、正文、版次、日期等信息

工作原理:
  1. node_1.htm → 获取所有版面 node_id
  2. node_XXXXXX.htm → 获取该版所有文章链接
  3. content_XXXXXX_YYYYYYY.htm → 提取文章标题和正文
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import time
import os
import sys
from datetime import datetime, timedelta

BASE_URL = "http://epaper.tianjinwe.com"
BASE_PATH = "/tjrb/html"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tjrb_data")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def fetch_page(url, retries=3, delay=1.0):
    """获取页面HTML，带重试"""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.encoding = 'utf-8'
            if resp.status_code == 200:
                return resp.text
            print(f"    [重试] HTTP {resp.status_code}")
        except Exception as e:
            print(f"    [重试] {e} ({attempt+1}/{retries})")
        time.sleep(delay * (attempt + 1))
    return None


def get_sections_from_date_page(date_str):
    """
    从 node_1.htm 获取所有版面的 node_id 和版名。
    返回: [{'num': 1, 'name': '要闻', 'node_id': '143078'}, ...]
    """
    url = f"{BASE_URL}{BASE_PATH}/{date_str}/node_1.htm"
    html = fetch_page(url)
    if not html:
        return []

    soup = BeautifulSoup(html, 'lxml')
    sections = []
    section_pattern = re.compile(r'第(\d+)版[：:](.+)')

    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text(strip=True)
        if not text:
            continue

        # 匹配版面链接: 文本如 "第01版：要闻", href 如 "node_143078.htm"
        if text.startswith('第') and '版' in text:
            match = section_pattern.match(text)
            if match:
                node_match = re.search(r'node_(\d+)', href)
                if node_match:
                    sections.append({
                        'num': int(match.group(1)),
                        'name': match.group(2).strip(),
                        'node_id': node_match.group(1)
                    })

    # 去重（保留第一次出现的）
    seen = set()
    unique = []
    for s in sections:
        key = s['node_id']
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return sorted(unique, key=lambda x: x['num'])


def get_articles_from_section_page(date_str, node_id):
    """
    从 node_XXXXXX.htm 获取该版所有文章链接。
    返回: [{'title': '...', 'url': '...'}, ...]
    """
    url = f"{BASE_URL}{BASE_PATH}/{date_str}/node_{node_id}.htm"
    html = fetch_page(url)
    if not html:
        return []

    soup = BeautifulSoup(html, 'lxml')
    articles = []
    seen = set()

    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text(strip=True)
        if not text:
            continue

        # 匹配文章链接
        content_match = re.search(r'content_(\d+)_(\d+)', href)
        if content_match and len(text) >= 2:
            art_key = f"{content_match.group(1)}_{content_match.group(2)}"
            if art_key not in seen:
                seen.add(art_key)
                # 处理多行标题（HTML中可能有换行）
                title = text.replace('\n', '').replace('\r', '').strip()
                # 构建完整URL
                full_url = f"{BASE_URL}{BASE_PATH}/{date_str}/{href}"
                articles.append({
                    'title': title,
                    'url': full_url,
                    'art_id': art_key
                })

    return articles


def parse_article_page(url):
    """
    解析单篇文章页面，提取标题和正文。
    返回: (title, subtitle, full_text)
    """
    html = fetch_page(url)
    if not html:
        return None, None, None

    soup = BeautifulSoup(html, 'lxml')

    # 方法1: 从页面title提取
    title = None
    subtitle = None

    page_title_tag = soup.find('title')
    if page_title_tag:
        page_title = page_title_tag.get_text(strip=True)
        # 格式: "天津日报数字报刊平台-文章标题"
        if '天津日报数字报刊平台-' in page_title:
            title = page_title.replace('天津日报数字报刊平台-', '').strip()

    # 移除脚本和样式
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()

    # 提取所有文本行
    text_all = soup.get_text(separator='\n', strip=True)
    lines = [l.strip() for l in text_all.split('\n') if l.strip()]

    # 如果方法1失败，从文本中提取
    if not title:
        skip_patterns = [
            '天津日报数字报刊平台', '天津日报多媒体数字版',
            '标题导航', '上一版', '下一版', '上一篇', '下一篇',
            '回到首页', '按日期查阅', '快速导航', '朗读', '放大', '缩小', '默认',
            '天津海河传媒中心', '法律事务部', '微信搜一搜',
            '上一期', '下一期', '敬告用户',
        ]
        for line in lines:
            if len(line) < 3 or len(line) > 80:
                continue
            if any(skip in line for skip in skip_patterns):
                continue
            if re.match(r'^\d{4}$', line):
                continue
            if re.match(r'^\d{4}年\d{2}月\d{2}日', line):
                continue
            if re.match(r'^第\d+版', line):
                continue
            title = line
            break

    # 提取正文 - 从"本报讯"或"新华社"开始
    content_lines = []
    in_content = False
    skip_markers = [
        '天津日报数字报刊平台', '标题导航', '上一版', '下一版', '上一篇', '下一篇',
        '回到首页', '朗读', '放大', '缩小', '默认',
        '天津海河传媒中心法律事务部', '微信搜一搜',
    ]

    for line in lines:
        # 跳过导航类
        if any(skip in line for skip in skip_markers):
            if in_content:
                if '天津海河传媒中心' in line:
                    break
            continue

        if len(line) < 3:
            continue

        # 正文开始标记
        if not in_content:
            if line.startswith('本报讯') or line.startswith('新华社'):
                in_content = True
                content_lines.append(line)
            elif line == title or line == subtitle:
                continue
            elif len(line) > 40 and any(c in line for c in '，。、'):
                # 可能是没有"本报讯"标记的正文
                in_content = True
                content_lines.append(line)
            continue

        if in_content:
            # 遇到版权信息停止
            if '天津海河传媒中心' in line or '天津日报' in line:
                if len(line) < 30:
                    break
            content_lines.append(line)

    full_text = '\n'.join(content_lines)

    # 清理正文
    if full_text:
        # 移除正文结尾的版权信息
        full_text = re.sub(r'\n*天津海河传媒中心法律事务部.*$', '', full_text)
        full_text = full_text.strip()

    return title, subtitle, full_text


def crawl_date(date_str, output_dir):
    """
    爬取指定日期的所有文章。
    date_str: "2026-07/21"
    """
    date_display = date_str.replace('/', '-')
    print(f"\n{'─'*50}")
    print(f"📅 {date_display}")

    # Step 1: 获取版面列表
    sections = get_sections_from_date_page(date_str)
    if not sections:
        print(f"  ⚠ 该日期无报纸或版面信息")
        return []

    print(f"  共 {len(sections)} 个版面")

    all_articles = []
    total_articles = 0

    # Step 2: 遍历每个版面
    for section in sections:
        sec_label = f"第{section['num']:02d}版：{section['name']}"
        articles = get_articles_from_section_page(date_str, section['node_id'])
        print(f"  📰 {sec_label} → {len(articles)} 篇文章")

        if not articles:
            continue

        # Step 3: 抓取每篇文章正文
        for i, art in enumerate(articles):
            title, subtitle, content = parse_article_page(art['url'])

            if title is None:
                title = art['title']

            article = {
                'date': date_display,
                'section_num': section['num'],
                'section_name': section['name'],
                'title': title,
                'subtitle': subtitle or '',
                'content': content or '',
                'source_url': art['url'],
            }
            all_articles.append(article)

            status = f"✓ {len(content)}字" if content else "⚠ 空"
            print(f"    [{i+1}/{len(articles)}] {title[:35]}... {status}")

            time.sleep(0.3)

        total_articles += len(articles)

    # 保存
    if all_articles:
        date_dir = os.path.join(output_dir, date_display.replace('-', os.sep))
        ensure_dir(date_dir)
        output_file = os.path.join(date_dir, f"{date_display}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_articles, f, ensure_ascii=False, indent=2)
        print(f"  💾 保存: {output_file} ({len(all_articles)} 篇)")

    return all_articles


def crawl_date_range(start_date, end_date, output_dir):
    """按日期范围爬取"""
    print(f"\n{'='*60}")
    print(f"  天津日报数字报爬虫")
    print(f"  范围: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print(f"  输出: {output_dir}")
    print(f"{'='*60}")

    ensure_dir(output_dir)

    total = 0
    success = 0
    fail = 0
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y-%m/%d")
        try:
            articles = crawl_date(date_str, output_dir)
            if articles:
                total += len(articles)
                success += 1
            else:
                fail += 1
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            fail += 1

        current += timedelta(days=1)

    print(f"\n{'='*60}")
    print(f"  完成! 成功: {success}天 | 无报/失败: {fail}天 | 文章: {total}篇")
    print(f"{'='*60}")
    return total


if __name__ == '__main__':
    today = datetime.now()
    start = today
    end = today

    if len(sys.argv) >= 2:
        if sys.argv[1] == '--today':
            pass
        elif sys.argv[1] == '--year':
            start = today - timedelta(days=365)
            end = today
        elif sys.argv[1] == '--range' and len(sys.argv) >= 4:
            start = datetime.strptime(sys.argv[2], '%Y-%m-%d')
            end = datetime.strptime(sys.argv[3], '%Y-%m-%d')
        elif sys.argv[1] == '--date':
            start = datetime.strptime(sys.argv[2], '%Y-%m-%d')
            end = start
        else:
            print("用法:")
            print("  python tjrb_crawler.py --today          # 爬取今天")
            print("  python tjrb_crawler.py --date 2026-07-21 # 爬取指定日期")
            print("  python tjrb_crawler.py --range 2026-07-15 2026-07-21")
            print("  python tjrb_crawler.py --year           # 爬取近一年(365天)")
            sys.exit(1)

    crawl_date_range(start, end, OUTPUT_DIR)