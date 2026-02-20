#!/usr/bin/env python3
"""Build RSS feeds from public Telegram channels via https://t.me/s/<channel>."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

DEFAULT_KEYWORDS = [
    "вакансия",
    "ищем",
    "нужен",
    "требуется",
    "копирайтер",
    "smm",
    "pr",
    "seo",
    "дизайнер",
]

CATEGORY_RULES = {
    "design": [
        "дизайнер",
        "design",
        "designer",
        "ui",
        "ux",
        "figma",
        "графдизайнер",
        "graphic designer",
        "motion",
        "иллюстратор",
    ],
    "content": [
        "копирайтер",
        "редактор",
        "editor",
        "author",
        "контент",
        "контент-менедж",
        "сценарист",
        "scriptwriter",
    ],
    "marketing": [
        "маркетолог",
        "marketing",
        "smm",
        "pr",
        "seo",
        "контекст",
        "таргет",
        "performance",
        "lead generation",
        "директ",
        "ads",
    ],
}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


class MessageTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.capture_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = dict(attrs)
        classes = attrs_map.get("class", "") or ""
        if tag == "div" and "tgme_widget_message_text" in classes:
            self.capture_depth = 1
            return

        if self.capture_depth > 0:
            if tag in {"br", "p", "li"}:
                self.parts.append("\n")
            if tag == "div":
                self.capture_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self.capture_depth > 0 and tag == "div":
            self.capture_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.capture_depth > 0:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        return re.sub(r"\s+", " ", raw).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect public Telegram posts by keywords and build RSS"
    )
    parser.add_argument("--channels", nargs="+", required=True, help="Public channels")
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=DEFAULT_KEYWORDS,
        help="Keywords for filtering posts",
    )
    parser.add_argument(
        "--pages-per-channel",
        type=int,
        default=6,
        help="How many public archive pages to scan per channel",
    )
    parser.add_argument("--max-items", type=int, default=200, help="Max RSS items")
    parser.add_argument(
        "--min-salary-rub",
        type=int,
        default=None,
        help="Keep only vacancies with salary >= N rubles",
    )
    parser.add_argument(
        "--since-days",
        type=float,
        default=None,
        help="Only keep posts from the last N days (e.g. 3)",
    )
    parser.add_argument(
        "--today-only",
        action="store_true",
        help="Keep only posts from current day in selected timezone",
    )
    parser.add_argument(
        "--timezone",
        default="Europe/Moscow",
        help="Timezone for --today-only window",
    )
    parser.add_argument(
        "--output",
        default="scripts/telegram_jobs_feed.xml",
        help="Output RSS XML path",
    )
    parser.add_argument(
        "--category-output-dir",
        default=None,
        help="If set, creates separate RSS files by category in this directory",
    )
    parser.add_argument("--feed-title", default="Telegram Jobs Feed")
    parser.add_argument("--feed-link", required=True, help="Your site URL")
    parser.add_argument("--feed-description", default="Filtered Telegram vacancies")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON and exit")
    return parser.parse_args()


def normalize_channel(value: str) -> str:
    value = value.strip()
    value = value.replace("https://t.me/", "", 1).replace("http://t.me/", "", 1)
    value = value.removeprefix("s/").lstrip("@")
    if not value:
        raise ValueError("Empty channel value")
    return value


def build_public_url(channel: str, before: str | None = None) -> str:
    base = f"https://t.me/s/{quote(channel)}"
    return f"{base}?before={quote(before)}" if before else base


def compile_keyword_pattern(keywords: Iterable[str]) -> re.Pattern[str]:
    parts: list[str] = []
    for raw in keywords:
        keyword = raw.strip()
        if not keyword:
            continue
        escaped = re.escape(keyword)
        # For short latin-only markers like "pr", "smm", "seo" match full word only.
        if re.fullmatch(r"[A-Za-z]{2,5}", keyword):
            parts.append(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])")
        else:
            parts.append(escaped)

    if not parts:
        raise ValueError("Keywords cannot be empty")
    return re.compile(r"(" + "|".join(parts) + r")", re.IGNORECASE)


def fetch_html(url: str, timeout_sec: int = 20) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"HTTP error for {url}: {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc.reason}") from exc


def extract_message_text(block_html: str) -> str:
    parser = MessageTextParser()
    parser.feed(block_html)
    return parser.text()


def extract_first_message_text(page_html: str) -> str | None:
    messages = extract_messages(page_html)
    if not messages:
        return None
    return messages[0].get("raw_text") or None


def extract_datetime(block_html: str) -> datetime | None:
    match = re.search(r'<time[^>]*datetime="([^"]+)"', block_html)
    if not match:
        return None
    value = html.unescape(match.group(1))
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def extract_next_before(page_html: str) -> str | None:
    match = re.search(r'href="/s/[^"?]+\?before=([^"]+)"', page_html)
    return html.unescape(match.group(1)) if match else None


def extract_messages(page_html: str) -> list[dict]:
    matches = list(
        re.finditer(
            r'<div class="tgme_widget_message\b[^>]*data-post="([^"]+)/(\d+)"[^>]*>',
            page_html,
        )
    )
    items: list[dict] = []

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(page_html)
        block = page_html[start:end]

        channel = match.group(1).strip()
        message_id = int(match.group(2))
        text = extract_message_text(block)
        dt_obj = extract_datetime(block)
        if not text or not dt_obj:
            continue

        items.append(
            {
                "channel": channel,
                "message_id": message_id,
                "raw_text": text,
                "url": f"https://t.me/{quote(channel)}/{message_id}",
                "dt": dt_obj,
            }
        )

    return items


def normalize_for_fingerprint(text: str) -> str:
    value = text.lower()
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"@\w+", " ", value)
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def detect_category(text: str) -> str:
    value = text.lower()

    def marker_match(marker: str) -> bool:
        if re.fullmatch(r"[A-Za-z]{2,5}", marker):
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(marker)}(?![A-Za-z0-9_])"
            return re.search(pattern, value, flags=re.IGNORECASE) is not None
        return marker in value

    for category, markers in CATEGORY_RULES.items():
        if any(marker_match(marker) for marker in markers):
            return category
    return "other"


def salary_values_rub(text: str) -> list[int]:
    values: list[int] = []
    value_text = text.lower().replace("\u00a0", " ")

    for match in re.finditer(r"(\d{2,3}(?:\s\d{3})+|\d{2,3})\s*(к|k|тыс)?\b", value_text):
        raw_num = match.group(1).replace(" ", "")
        suffix = match.group(2)
        try:
            base = int(raw_num)
        except ValueError:
            continue

        # 100к / 100k / 100 тыс
        if suffix in {"к", "k", "тыс"}:
            values.append(base * 1000)
            continue

        # Plain large numbers like 120000 or 120 000
        if base >= 10000:
            values.append(base)

    return values


def in_today_window(dt_obj: datetime, tz_name: str) -> bool:
    tz = ZoneInfo(tz_name)
    now_local = datetime.now(tz)
    day_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    local_dt = dt_obj.astimezone(tz)
    return day_start <= local_dt < day_end


def to_rfc2822(dt_obj: datetime) -> str:
    return dt_obj.astimezone(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def looks_truncated(text: str) -> bool:
    t = clean_text(text)
    return t.endswith("...") or t.endswith("…")


def fetch_full_message_text(channel: str, message_id: int) -> str | None:
    full_url = f"https://t.me/{quote(channel)}/{message_id}"
    try:
        page_html = fetch_html(full_url)
    except RuntimeError:
        return None

    full_text = extract_first_message_text(page_html)
    if not full_text:
        return None
    return clean_text(full_text)


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def write_rss(path: Path, feed_title: str, args: argparse.Namespace, items: list[dict]) -> None:
    rss = ET.Element("rss", attrib={"version": "2.0"})
    channel = ET.SubElement(rss, "channel")
    ET.SubElement(channel, "title").text = feed_title
    ET.SubElement(channel, "link").text = args.feed_link
    ET.SubElement(channel, "description").text = args.feed_description
    ET.SubElement(channel, "lastBuildDate").text = to_rfc2822(datetime.now(timezone.utc))

    for row in items:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = row["title"]
        ET.SubElement(item, "link").text = row["url"]
        ET.SubElement(item, "guid", attrib={"isPermaLink": "true"}).text = row["url"]
        ET.SubElement(item, "pubDate").text = row["pub_date"]
        ET.SubElement(item, "category").text = row["category"]
        ET.SubElement(item, "description").text = row["description"]

    indent_xml(rss)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(rss).write(path, encoding="utf-8", xml_declaration=True)


def collect_items(args: argparse.Namespace) -> list[dict]:
    channels = [normalize_channel(c) for c in args.channels]
    pattern = compile_keyword_pattern(args.keywords)
    collected: list[dict] = []

    min_dt = None
    if args.since_days is not None:
        min_dt = datetime.now(timezone.utc) - timedelta(days=args.since_days)

    seen_urls: set[str] = set()
    seen_fingerprints: set[str] = set()
    full_text_cache: dict[tuple[str, int], str | None] = {}

    for channel in channels:
        before: str | None = None
        visited_urls: set[str] = set()

        for _ in range(args.pages_per_channel):
            page_url = build_public_url(channel, before)
            if page_url in visited_urls:
                break
            visited_urls.add(page_url)

            page_html = fetch_html(page_url)
            for msg in extract_messages(page_html):
                text = clean_text(msg["raw_text"])
                if looks_truncated(text):
                    cache_key = (msg["channel"], msg["message_id"])
                    if cache_key not in full_text_cache:
                        full_text_cache[cache_key] = fetch_full_message_text(
                            msg["channel"], msg["message_id"]
                        )
                    full_text = full_text_cache[cache_key]
                    if full_text and len(full_text) > len(text):
                        text = full_text

                if not pattern.search(text):
                    continue

                dt_obj = msg["dt"]
                if min_dt and dt_obj.astimezone(timezone.utc) < min_dt:
                    continue
                if args.today_only and not in_today_window(dt_obj, args.timezone):
                    continue
                if args.min_salary_rub is not None:
                    salaries = salary_values_rub(text)
                    if not salaries or max(salaries) < args.min_salary_rub:
                        continue

                fingerprint = hashlib.sha256(
                    normalize_for_fingerprint(text).encode("utf-8")
                ).hexdigest()

                if msg["url"] in seen_urls or fingerprint in seen_fingerprints:
                    continue

                category = detect_category(text)
                collected.append(
                    {
                        "title": f"Вакансия из @{channel}",
                        "url": msg["url"],
                        "pub_date": to_rfc2822(dt_obj),
                        "sort_ts": dt_obj.timestamp(),
                        "category": category,
                        "description": text,
                        "raw_text": text,
                        "channel": channel,
                        "message_id": msg["message_id"],
                    }
                )
                seen_urls.add(msg["url"])
                seen_fingerprints.add(fingerprint)

            before = extract_next_before(page_html)
            if not before:
                break

    collected.sort(key=lambda x: x["sort_ts"], reverse=True)
    return collected[: args.max_items]


def write_category_feeds(args: argparse.Namespace, items: list[dict]) -> list[str]:
    if not args.category_output_dir:
        return []

    base_dir = Path(args.category_output_dir)
    created: list[str] = []

    for category in ["content", "design", "marketing", "other"]:
        cat_items = [row for row in items if row["category"] == category]
        if not cat_items:
            continue
        out = base_dir / f"telegram_jobs_{category}.xml"
        write_rss(out, f"{args.feed_title} - {category}", args, cat_items)
        created.append(str(out))

    return created


def main() -> None:
    args = parse_args()
    items = collect_items(args)

    if args.dry_run:
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return

    main_out = Path(args.output)
    write_rss(main_out, args.feed_title, args, items)
    cat_files = write_category_feeds(args, items)

    print(f"Generated {len(items)} RSS items at {main_out}")
    if cat_files:
        print("Category feeds:")
        for path in cat_files:
            print(f"- {path}")


if __name__ == "__main__":
    main()
