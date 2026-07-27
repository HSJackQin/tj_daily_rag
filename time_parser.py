#!/usr/bin/env python3
"""
自然语言时间解析器
从用户的自然语言提问中提取时间范围，转换为 date_from / date_to。
支持的表述:
  今天/今日/昨天/前天
  最近一周/近一周/本周/这周
  最近一个月/近一个月/本月
  最近三个月/近三个月/本季度
  最近半年/近半年
  最近一年/近一年/今年
  上个月/上月/去年
  X月/XX月/X月份
  2026年/2026年X月
  X月X日 到 X月X日
"""

import re
from datetime import datetime, timedelta


def parse_time_range(question: str) -> tuple:
    """
    从自然语言中提取时间范围。
    返回: (date_from, date_to) 格式 'YYYY-MM-DD'，未识别返回 (None, None)
    """
    today = datetime.now()

    # ── 绝对日期 ──
    # 今天、今日
    if re.search(r'今天|今日', question):
        d = today.strftime('%Y-%m-%d')
        return (d, d)

    # 昨天、昨日
    if re.search(r'昨天|昨日', question):
        d = (today - timedelta(days=1)).strftime('%Y-%m-%d')
        return (d, d)

    # 前天
    if re.search(r'前天', question):
        d = (today - timedelta(days=2)).strftime('%Y-%m-%d')
        return (d, d)

    # ── 相对时间范围 ──
    # 最近一周 / 近一周 / 本周 / 这周 / 这一周
    if re.search(r'最近[一1]周|近[一1]周|本周|这[一]?周', question):
        return (
            (today - timedelta(days=7)).strftime('%Y-%m-%d'),
            today.strftime('%Y-%m-%d'),
        )

    # 最近两周 / 近两周 / 这两周
    if re.search(r'最近[两2]周|近[两2]周|这[两2]周', question):
        return (
            (today - timedelta(days=14)).strftime('%Y-%m-%d'),
            today.strftime('%Y-%m-%d'),
        )

    # 最近一个月 / 近一个月 / 近一月 / 本月 / 这个月
    if re.search(r'最近[一1]个?月|近[一1]个?月|近[一1]月|本月|这个月', question):
        return (
            (today - timedelta(days=30)).strftime('%Y-%m-%d'),
            today.strftime('%Y-%m-%d'),
        )

    # 最近三个月 / 近三个月 / 本季度
    if re.search(r'最近[三3]个?月|近[三3]个?月|本季度|这个季度', question):
        return (
            (today - timedelta(days=90)).strftime('%Y-%m-%d'),
            today.strftime('%Y-%m-%d'),
        )

    # 最近半年 / 近半年
    if re.search(r'最近半年|近半年', question):
        return (
            (today - timedelta(days=180)).strftime('%Y-%m-%d'),
            today.strftime('%Y-%m-%d'),
        )

    # 最近一年 / 近一年 / 今年 / 这一年
    if re.search(r'最近[一1]年|近[一1]年|今年|这[一]?年', question):
        return (
            (today - timedelta(days=365)).strftime('%Y-%m-%d'),
            today.strftime('%Y-%m-%d'),
        )

    # ── 上一个周期 ──
    # 上个月 / 上月
    if re.search(r'上个?月|上月', question):
        first_this_month = today.replace(day=1)
        last_prev_month = first_this_month - timedelta(days=1)
        first_prev_month = last_prev_month.replace(day=1)
        return (
            first_prev_month.strftime('%Y-%m-%d'),
            last_prev_month.strftime('%Y-%m-%d'),
        )

    # 上周
    if re.search(r'上周|上个?星期', question):
        days_since_monday = today.weekday()
        this_monday = today - timedelta(days=days_since_monday)
        last_sunday = this_monday - timedelta(days=1)
        last_monday = this_monday - timedelta(days=7)
        return (
            last_monday.strftime('%Y-%m-%d'),
            last_sunday.strftime('%Y-%m-%d'),
        )

    # 去年
    if re.search(r'去年', question):
        return (
            f"{today.year - 1}-01-01",
            f"{today.year - 1}-12-31",
        )

    # ── 指定年月日 ──
    # "X月X日 到 X月X日" 这种日期区间
    date_range = re.search(
        r'(\d{1,2})\s*月\s*(\d{1,2})\s*日?\s*[到至-]\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?',
        question,
    )
    if date_range:
        m1, d1, m2, d2 = map(int, date_range.groups())
        y = today.year
        # 如果 m2 < m1，跨年了
        if m2 < m1:
            y2 = y + 1
        else:
            y2 = y
        return (
            f"{y}-{m1:02d}-{d1:02d}",
            f"{y2}-{m2:02d}-{d2:02d}",
        )

    # "2026年X月X日" 或 "2026年X月" 或 "2026年"
    ym_match = re.search(r'(\d{4})\s*年', question)
    if ym_match:
        year = int(ym_match.group(1))
        # 找月份
        mon_match = re.search(rf'{year}\s*年\s*(\d{{1,2}})\s*月', question)
        if mon_match:
            month = int(mon_match.group(1))
            # 找日期
            day_match = re.search(
                rf'{year}\s*年\s*{month}\s*月\s*(\d{{1,2}})\s*日', question
            )
            if day_match:
                day = int(day_match.group(1))
                d = f"{year}-{month:02d}-{day:02d}"
                return (d, d)
            else:
                # 某年某月，整个月
                import calendar
                last_day = calendar.monthrange(year, month)[1]
                return (
                    f"{year}-{month:02d}-01",
                    f"{year}-{month:02d}-{last_day:02d}",
                )
        else:
            # 只有年份
            return (f"{year}-01-01", f"{year}-12-31")

    # "X月"（简短形式，没有年份，默认为当前年）
    mon_only = re.search(r'(?<!\d)(\d{1,2})\s*月(?:份)?(?![份日])', question)
    if mon_only:
        month = int(mon_only.group(1))
        if 1 <= month <= 12:
            import calendar
            y = today.year
            last_day = calendar.monthrange(y, month)[1]
            return (
                f"{y}-{month:02d}-01",
                f"{y}-{month:02d}-{last_day:02d}",
            )

    # ── 没识别到 ──
    return (None, None)


# ---------------------------------------------------------------------------
# CLI 测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        "请帮我梳理最近这一周天津日报刊登了哪些和经济发展有关的内容？",
        "今天天津日报有哪些重要新闻？",
        "最近一个月天津在人工智能方面有什么进展？",
        "2026年3月有天津港的报道吗？",
        "上个月有哪些关于教育的文章？",
        "去年天津GDP增速如何？",
        "请梳理最近一年天津日报上关于天津港的内容",
        "最近两周有什么体育新闻？",
        "1月到3月期间有哪些民生相关的报道？",
        "这周有什么文化活动的报道？",
    ]
    for q in tests:
        f, t = parse_time_range(q)
        print(f"{'✅' if f else '❌'} {f or '-':>10} ~ {t or '-':<10}  ← {q}")
