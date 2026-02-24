#!/usr/bin/env python3
import hashlib
import json
import os
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
API_URL = "https://openrouter.ai/api/v1/chat/completions"
STATE_FILE = Path(".rewrite_state.json")
REWRITE_VERSION = "v2"

SYSTEM = (
    "Ты редактор вакансий для русскоязычного job-сайта. "
    "Сохраняй факты строго 1:1: суммы, даты, города, форматы занятости, контакты, ссылки, @username. "
    "Ничего не выдумывай и не добавляй. "
    "Верни JSON с полями title и description. "
    "title: короткий и информативный, до 95 символов. "
    "description: 5 строк в таком порядке, каждая с новой строки: "
    "1) Зарплата: ... "
    "2) Формат: ... "
    "3) Задачи: ... "
    "4) Требования: ... "
    "5) Контакты: ... "
    "Если данных нет, пиши 'не указано'. "
    "Не используй markdown, html и эмодзи."
)


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_title(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:95] if len(s) > 95 else s


def clean_description(s: str) -> str:
    s = s.replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return "Зарплата: не указано\nФормат: не указано\nЗадачи: не указано\nТребования: не указано\nКонтакты: не указано"

    prefixes = ("Зарплата:", "Формат:", "Задачи:", "Требования:", "Контакты:")
    normalized = []
    for i, pfx in enumerate(prefixes):
        if i < len(lines):
            line = lines[i]
            if not line.startswith(pfx):
                if ":" in line:
                    line = pfx + " " + line.split(":", 1)[1].strip()
                else:
                    line = f"{pfx} {line}"
        else:
            line = f"{pfx} не указано"
        normalized.append(line)
    return "\n".join(normalized)


def call_or(raw_title: str, raw_desc: str):
    payload = {
        "model": MODEL,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"TITLE: {raw_title}\nTEXT: {raw_desc}"},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/pereverzevart/jobs-feed-public",
            "X-Title": "jobs-feed-public",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.loads(r.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    obj = json.loads(content)
    title = clean_title(obj.get("title") or raw_title)
    desc = clean_description(obj.get("description") or raw_desc)
    return title, desc


def rewrite_file(path: str, state: dict, limit=30):
    tree = ET.parse(path)
    root = tree.getroot()
    items = root.findall("./channel/item")
    changed = 0
    processed = 0

    for item in items:
        if processed >= limit:
            break

        title_el = item.find("title")
        desc_el = item.find("description")
        link_el = item.find("link")
        if title_el is None or desc_el is None:
            continue

        link = (link_el.text or "").strip() if link_el is not None else ""
        raw_title = (title_el.text or "").strip()
        raw_desc = (desc_el.text or "").strip()
        if not raw_desc:
            continue

        src_hash = hashlib.sha256((REWRITE_VERSION + "\n" + raw_title + "\n" + raw_desc).encode("utf-8")).hexdigest()
        key = link or hashlib.sha256(raw_title.encode("utf-8")).hexdigest()

        if state.get(key) == src_hash:
            continue

        try:
            new_title, new_desc = call_or(raw_title, raw_desc)
            title_el.text = new_title
            desc_el.text = new_desc
            state[key] = src_hash
            changed += 1
            processed += 1
            time.sleep(0.35)
        except Exception as e:
            print(f"[warn] {path}: {e}")

    tree.write(path, encoding="utf-8", xml_declaration=True)
    print(f"[ok] {path}: changed={changed}")


def main():
    if not API_KEY:
        raise SystemExit("OPENROUTER_API_KEY missing")

    state = load_state()

    for feed in [
        "telegram_jobs_feed.xml",
        "telegram_jobs_crypto.xml",
        "telegram_jobs_100k.xml",
    ]:
        if Path(feed).exists():
            rewrite_file(feed, state, limit=30)

    save_state(state)


if __name__ == "__main__":
    main()
