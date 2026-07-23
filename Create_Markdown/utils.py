#!/usr/bin/env python3
"""
utils.py — Общие функции для пайплайна конвертации документов в Markdown.

Содержит:
- call_claude() — универсальный вызов Claude с fallback и проверкой finish_reason
- call_vision() — универсальный вызов Vision API (formula / classify)
- call_vision_formula() — обратно-совместимая обёртка для формул
- check_model_available() — проверка доступности модели
- chunk_text() — разбивка текста на батчи
- save_intermediate() — безопасное сохранение с backup
- find_formula_images() — поиск ссылок на формулы в тексте
- needs_cleanup() — проверка на нераспознанные формулы
- fix_subscripts() — regex-чистка LaTeX индексов

Использование:
  from utils import call_claude, call_vision, check_model_available, chunk_text
"""

import base64
import httpx
import io
import logging
import os
import re
import shutil
import time
from pathlib import Path
from PIL import Image
from logging.handlers import RotatingFileHandler

log = logging.getLogger("utils")

# ── Настройка логирования ───────────────────────────────────────────────

def init_logging(log_dir: str | Path | None = None):
    """Настроить логирование: консоль + файл (с ротацией).

    Вызывается из main() каждого скрипта. Если log_dir не указан,
    используется папка, где лежит вызывающий скрипт.
    """
    if log_dir is None:
        # Определяем папку вызывающего скрипта
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller = frame.f_back.f_globals.get("__file__", "")
            log_dir = Path(caller).parent.resolve() if caller else Path.cwd()
        else:
            log_dir = Path.cwd()
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "create_markdown.log"

    # Настройка корневого логгера
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Формат
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Удаляем старые handler-ы, если init_logging вызывается повторно
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Консоль (только INFO и выше)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Файл (все уровни, включая DEBUG)
    file_handler = RotatingFileHandler(
        str(log_file), maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    log.info(f"Лог-файл: {log_file}")

# ── Configuration ───────────────────────────────────────────────────────
PROVOD_URL = "https://api.provod.ai/v1/chat/completions"
PROVOD_API_KEY = os.environ.get("PROVOD_API_KEY", "")

# Цепочка fallback для текстовой постобработки
TEXT_MODELS = [
    "anthropic/claude-sonnet-5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.8",
    "openai/gpt-5.4",
]

# Цепочка fallback для Vision API
VISION_MODELS = [
    "google/gemini-3.5-flash",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.4",
]

HTTP_TIMEOUT = 900
MAX_TOKENS_OUT = 64000

# ── Счётчики токенов ────────────────────────────────────────────────────
# Накопление статистики за сессию обработки одного документа
_token_stats = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "vision_calls": 0,
}

def reset_token_stats():
    """Сбросить накопленную статистику токенов (для нового документа)."""
    _token_stats["prompt_tokens"] = 0
    _token_stats["completion_tokens"] = 0
    _token_stats["vision_calls"] = 0

def get_token_stats() -> dict:
    """Вернуть копию накопленной статистики."""
    return dict(_token_stats)

# ── Vision промпты ──────────────────────────────────────────────────────

VISION_FORMULA_PROMPT = r"""Read the mathematical formula in this image carefully.
Output ONLY the LaTeX code, using $$...$$ for display formulas.
Pay attention to:
- Cyrillic subscripts like эк, расч, ном, доп
- Greek letters like \tau (tau), \Theta (Theta)
- Mathematical operators like \sqrt, \sum, \int
- Fractions and subscripts
Do NOT add any commentary, explanations, or extra text.
Do NOT wrap in code fences."""

VISION_CLASSIFY_PROMPT = r"""Look at this image. Is it a mathematical formula/equation?
- If YES: output FORMULA: followed by the LaTeX code using $$...$$
  Example: FORMULA: $$I_{кз} = \frac{U_{ном}}{\sqrt{3} \cdot Z_{т}}$$
- If NO (it's a diagram, graph, schematic, photo, table, or any other non-formula image): output just IMAGE
Do NOT add any commentary."""


# ═══════════════════════════════════════════════════════════════════════════
# 0. Проверка доступности модели
# ═══════════════════════════════════════════════════════════════════════════

def check_model_available(model: str, timeout: int = 10) -> bool:
    """Проверить доступность модели коротким запросом (ping).

    Отправляет минимальный chat completion (1 токен) и проверяет HTTP 200.
    Если модель перегружена (503) или недоступна — возвращает False.
    """
    if not PROVOD_API_KEY:
        return False

    headers = {
        "Authorization": f"Bearer {PROVOD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "."}
        ],
        "temperature": 0.0,
        "max_tokens": 1,
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(PROVOD_URL, json=payload, headers=headers)

        if resp.status_code == 503:
            log.debug(f"  ping {model}: 503 (перегрузка)")
            return False
        if resp.status_code == 200:
            log.debug(f"  ping {model}: OK")
            return True

        log.debug(f"  ping {model}: HTTP {resp.status_code}")
        return False

    except (httpx.TimeoutException, httpx.ConnectError):
        log.debug(f"  ping {model}: недоступен (таймаут/ошибка соединения)")
        return False
    except Exception as e:
        log.debug(f"  ping {model}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 1. Универсальный вызов Claude (текстовый) с fallback
# ═══════════════════════════════════════════════════════════════════════════

def call_claude(
    system_prompt: str,
    user_text: str,
    label: str = "Claude",
    max_tokens: int = MAX_TOKENS_OUT,
    timeout: int = HTTP_TIMEOUT,
    model_chain: list | None = None,
) -> str:
    """Универсальный вызов Claude с fallback-цепочкой и проверкой finish_reason.

    Args:
        system_prompt: Системный промпт
        user_text: Текст для обработки
        label: Метка для логирования
        max_tokens: Максимум токенов в ответе
        timeout: Таймаут запроса
        model_chain: Список моделей для fallback (по умолчанию TEXT_MODELS)

    Returns:
        Ответ модели или исходный текст при ошибке
    """
    if not PROVOD_API_KEY:
        log.warning(f"  {label}: PROVOD_API_KEY не задан, пропускаю")
        return user_text

    if model_chain is None:
        model_chain = TEXT_MODELS

    headers = {
        "Authorization": f"Bearer {PROVOD_API_KEY}",
        "Content-Type": "application/json",
    }

    for model in model_chain:
        for attempt in range(3):
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                }

                start = time.time()
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(PROVOD_URL, json=payload, headers=headers)

                if resp.status_code == 503:
                    log.warning(f"  {label}/{model}: 503, попытка {attempt+1}/3")
                    time.sleep(5)
                    continue

                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                content = choice["message"]["content"].strip()
                finish_reason = choice.get("finish_reason", "")

                elapsed = time.time() - start
                usage = data.get("usage", {})
                pt = usage.get("prompt_tokens", 0)
                ct = usage.get("completion_tokens", 0)
                _token_stats["prompt_tokens"] += pt
                _token_stats["completion_tokens"] += ct
                log.info(
                    f"  {label}/{model}: {elapsed:.1f}с, "
                    f"in={pt} "
                    f"out={ct} "
                    f"finish={finish_reason}"
                )

                # Убираем обёртку в код-фенсы
                content = re.sub(r"^```(?:markdown)?\s*\n?", "", content, flags=re.MULTILINE)
                content = re.sub(r"\n```\s*$", "", content, flags=re.MULTILINE)

                # Проверка finish_reason
                if finish_reason == "length":
                    log.warning(
                        f"  {label}/{model}: ответ оборван (finish_reason=length)! "
                        f"Получено {len(content)} символов, возможно неполный результат"
                    )
                    # Возвращаем что есть — лучше неполный ответ, чем ничего

                return content.strip()

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 503:
                    log.warning(f"  {label}/{model}: 503, попытка {attempt+1}/3")
                    time.sleep(5)
                    continue
                log.warning(f"  {label}/{model}: HTTP {status} — {e.response.text[:200]}")
                time.sleep(3)
            except httpx.TimeoutException:
                log.warning(f"  {label}/{model}: таймаут {timeout}с, попытка {attempt+1}/3")
                time.sleep(5)
            except Exception as e:
                log.warning(f"  {label}/{model}: {e}")
                time.sleep(3)

        log.warning(f"  {label}/{model}: исчерпаны попытки, пробую следующую модель...")

    log.error(f"  {label}: все модели недоступны, возвращаю исходный текст")
    return user_text


# ═══════════════════════════════════════════════════════════════════════════
# 2. Универсальный вызов Vision API
# ═══════════════════════════════════════════════════════════════════════════

def call_vision(
    image_path: str,
    mode: str = "formula",
    label: str = "",
    model_chain: list | None = None,
    upscale_threshold: int = 200,
) -> dict:
    """Универсальный вызов Vision API с апскейлом, fallback и ping-проверкой.

    Args:
        image_path: Путь к файлу изображения
        mode: "formula" — распознать формулу и вернуть LaTeX
              "classify" — классифицировать: FORMULA (с LaTeX) или IMAGE
        label: Метка для логирования
        model_chain: Список моделей для fallback (по умолчанию VISION_MODELS)
        upscale_threshold: Минимальный размер меньшей стороны для апскейла

    Returns:
        Для mode="formula":
            {"type": "FORMULA", "latex": "$$...$$", "model": "..."}
        Для mode="classify":
            {"type": "FORMULA", "latex": "$$...$$", "model": "..."}
            или
            {"type": "IMAGE", "model": "..."}
        При ошибке:
            {"type": "ERROR", "error": "..."}
    """
    if not PROVOD_API_KEY:
        return {"type": "ERROR", "error": "PROVOD_API_KEY не задан"}

    if model_chain is None:
        model_chain = VISION_MODELS

    # Апскейл маленьких изображений
    img = Image.open(image_path)
    if img.width < upscale_threshold or img.height < upscale_threshold:
        scale = max(4, upscale_threshold * 2 // min(img.width, img.height))
        new_size = (img.width * scale, img.height * scale)
        img = img.resize(new_size, Image.LANCZOS)
        log.debug(f"  {label}: апскейл {img.width//scale}x{img.height//scale} → {new_size}")

    # Ресайз больших изображений до 1024px по большей стороне (экономия токенов)
    MAX_VISION_SIZE = 1024
    if img.width > MAX_VISION_SIZE or img.height > MAX_VISION_SIZE:
        ratio = MAX_VISION_SIZE / max(img.width, img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        log.debug(f"  {label}: ресайз {img.width}x{img.height} → {new_size}")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # Выбор промпта
    if mode == "classify":
        prompt_text = VISION_CLASSIFY_PROMPT
    else:
        prompt_text = VISION_FORMULA_PROMPT
        if label:
            prompt_text += f"\nFormula from: {label}."

    headers = {
        "Authorization": f"Bearer {PROVOD_API_KEY}",
        "Content-Type": "application/json",
    }

    for model in model_chain:
        # Ping перед отправкой
        if not check_model_available(model):
            log.warning(f"  {label}: {model} недоступен, пропускаю")
            continue

        for attempt in range(3):
            try:
                payload = {
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                            },
                        ],
                    }],
                    "temperature": 0.0,
                    "max_tokens": 2048,
                }

                start = time.time()
                with httpx.Client(timeout=60) as c:
                    r = c.post(PROVOD_URL, json=payload, headers=headers)

                if r.status_code == 503:
                    log.warning(f"  {label}/{model}: 503, попытка {attempt+1}/3")
                    time.sleep(5)
                    continue

                r.raise_for_status()
                data = r.json()
                response_text = data["choices"][0]["message"]["content"].strip()
                response_text = re.sub(r"^```(?:latex)?\s*", "", response_text)
                response_text = re.sub(r"\s*```$", "", response_text)

                elapsed = time.time() - start
                log.info(f"  {label}/{model}: {elapsed:.1f}s → {response_text[:120]}")

                # Учитываем Vision вызов
                _token_stats["vision_calls"] += 1

                # Парсинг ответа для mode="classify"
                if mode == "classify":
                    if response_text.upper().startswith("FORMULA:"):
                        latex = response_text[len("FORMULA:"):].strip()
                        return {"type": "FORMULA", "latex": latex, "model": model}
                    elif response_text.upper().startswith("FORMULA"):
                        latex = response_text[len("FORMULA"):].strip()
                        # Убираем возможное двоеточие
                        latex = latex.lstrip(":")
                        return {"type": "FORMULA", "latex": latex, "model": model}
                    elif response_text.upper().strip() == "IMAGE":
                        return {"type": "IMAGE", "model": model}
                    else:
                        # Неопределённый ответ — считаем IMAGE для безопасности
                        log.warning(f"  {label}/{model}: неопределённый ответ: {response_text[:80]}")
                        return {"type": "IMAGE", "model": model}

                # mode="formula" — возвращаем LaTeX
                else:
                    return {"type": "FORMULA", "latex": response_text, "model": model}

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 503:
                    log.warning(f"  {label}/{model}: 503, попытка {attempt+1}/3")
                    time.sleep(5)
                    continue
                log.warning(f"  {label}/{model}: HTTP {e.response.status_code}")
                time.sleep(3)
            except httpx.TimeoutException:
                log.warning(f"  {label}/{model}: таймаут, попытка {attempt+1}/3")
                time.sleep(5)
            except Exception as e:
                log.warning(f"  {label}/{model}: {e}")
                time.sleep(3)

        log.warning(f"  {label}/{model}: исчерпаны попытки, пробую следующую...")

    log.error(f"  {label}: все Vision модели недоступны")
    return {"type": "ERROR", "error": "Все Vision модели недоступны"}


# ═══════════════════════════════════════════════════════════════════════════
# 2b. Обратно-совместимая обёртка для распознавания формул
# ═══════════════════════════════════════════════════════════════════════════

def call_vision_formula(
    image_path: str,
    formula_num: int = 0,
    model_chain: list | None = None,
) -> str:
    """Распознать формулу через Vision API (обратно-совместимая обёртка).

    Вызывает call_vision(mode="formula") и возвращает LaTeX-строку.

    Raises:
        RuntimeError: Если все модели недоступны
    """
    label = f"formula_{formula_num}" if formula_num else "formula"
    result = call_vision(image_path, mode="formula", label=label, model_chain=model_chain)

    if result["type"] == "FORMULA":
        return result["latex"]
    elif result["type"] == "ERROR":
        raise RuntimeError(result["error"])
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# 3. Чанкование текста
# ═══════════════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    separator: str = r'\n\n---\n\n',
    max_chars: int = 20000,
    filter_pattern: str | None = None,
) -> list[str]:
    """Разбить текст на батчи.

    Сначала пытается разделить по разделителю (---).
    Если не найден — делит по двойным переносам строк.
    Если и их нет — режет по символам на границе слов.

    Args:
        text: Исходный текст
        separator: Regex-разделитель секций
        max_chars: Максимальный размер батча в символах
        filter_pattern: Если задан — включать только секции, содержащие паттерн

    Returns:
        Список батчей (готовых к отправке в Claude)
    """
    sections = re.split(separator, text)

    if len(sections) <= 1:
        sections = re.split(r'\n\n', text)

    if len(sections) <= 1:
        words = text.split()
        sections = []
        current = []
        current_len = 0
        for word in words:
            if current_len + len(word) + 1 > max_chars and current:
                sections.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += len(word) + 1
        if current:
            sections.append(" ".join(current))

    if filter_pattern:
        sections = [s for s in sections if re.search(filter_pattern, s)]

    if not sections:
        return []

    batches = []
    current = []
    current_size = 0
    for sec in sections:
        if current_size + len(sec) > max_chars and current:
            batches.append("\n\n---\n\n".join(current))
            current = [sec]
            current_size = len(sec)
        else:
            current.append(sec)
            current_size += len(sec)
    if current:
        batches.append("\n\n---\n\n".join(current))

    return batches


# ═══════════════════════════════════════════════════════════════════════════
# 4. Сохранение с backup
# ═══════════════════════════════════════════════════════════════════════════

def save_intermediate(path: str, content: str, label: str = "") -> None:
    """Сохранить файл, сделав backup предыдущей версии.

    Предыдущая версия сохраняется как path.bak
    """
    path_obj = Path(path)
    if path_obj.exists():
        bak_path = path_obj.with_suffix(path_obj.suffix + ".bak")
        shutil.copy2(str(path_obj), str(bak_path))

    path_obj.write_text(content, encoding="utf-8")
    if label:
        log.info(f"  💾 {label}: {path_obj} ({len(content)} символов)")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Поиск формул в тексте
# ═══════════════════════════════════════════════════════════════════════════

def find_formula_images(text: str, img_dir: str | Path) -> list[dict]:
    """Найти все ссылки на изображения формул в тексте.

    Returns:
        Список словарей: {img_ref, img_name, img_path, full_match}
    """
    img_dir = Path(img_dir)
    pattern = r'!\[Формула\]\(image/(equation-[^)]+)\)'
    matches = list(re.finditer(pattern, text))

    if not matches:
        pattern2 = r'!\[.*?\]\(image/(equation-[^)]+)\)'
        matches = list(re.finditer(pattern2, text))

    results = []
    for m in matches:
        img_name = m.group(1)
        img_path = img_dir / img_name
        results.append({
            "img_ref": m.group(0),
            "img_name": img_name,
            "img_path": str(img_path),
            "full_match": m,
        })

    return results


def needs_cleanup(text: str) -> bool:
    """Проверить, есть ли в тексте нераспознанные формулы."""
    markers = ["<!-- formula-not-decoded -->", "formula-not-decoded", "Тг"]
    return any(m in text for m in markers)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Склейка разорванных таблиц
# ═══════════════════════════════════════════════════════════════════════════

def merge_broken_tables(md_text: str) -> str:
    """Склеить разорванные таблицы.

    Общее правило: если после таблицы (строки с |) идёт продолжение с | —
    даже через пустые строки, произвольный текст, маркеры «Продолжение»,
    «Окончание», «Таблица (продолжение)», <!-- image --> и т.п. —
    считать одной таблицей. Повторные шапки удалять.
    """
    HEADER_KEYWORDS = (
        'Наименование', 'Помещения', 'Производственные',
        'Температура', 'Степень огнестойкости',
        'Выделяющиеся вредности', 'Кратность/расход',
        'Показатель', 'Размерность',
    )
    CONTINUE_KEYWORDS = (
        'продолжение', 'прод', 'окончание', 'оконч',
        'Продолжение', 'Прод', 'Окончание', 'Оконч',
        'Таблица (продолжение)', 'продолжение предыдущей страницы',
    )

    def _is_table_header(line: str) -> bool:
        stripped = line.strip()
        if not stripped.startswith('|'):
            return False
        return any(kw in stripped for kw in HEADER_KEYWORDS)

    def _is_separator(line: str) -> bool:
        return line.strip().startswith('|-')

    def _is_continuation_marker(line: str) -> bool:
        stripped = line.strip().strip('#').strip()
        return any(kw.lower() in stripped.lower() for kw in CONTINUE_KEYWORDS)

    def _is_new_section_header(line: str) -> bool:
        stripped = line.strip()
        if not stripped.startswith('#'):
            return False
        return not _is_continuation_marker(stripped)

    lines = md_text.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.strip().startswith('|'):
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1

            # Ищем продолжение таблицы через разрыв
            while i < len(lines):
                if lines[i].strip() == '':
                    i += 1
                    continue

                stripped = lines[i].strip()

                # Пропускаем <!-- ... -->
                if stripped.startswith('<!--') and stripped.endswith('-->'):
                    i += 1
                    continue

                # Пропускаем маркеры «Продолжение», «Окончание»
                if _is_continuation_marker(stripped):
                    i += 1
                    continue

                # Если нашли строку с | — это продолжение таблицы
                if stripped.startswith('|'):
                    is_dup_header = (_is_table_header(stripped) and
                                     i + 1 < len(lines) and
                                     _is_separator(lines[i + 1]))
                    if is_dup_header:
                        i += 2  # шапка + разделитель
                        while i < len(lines) and lines[i].strip() == '':
                            i += 1
                        table_lines.append('')
                        continue
                    else:
                        # Это не шапка, а данные
                        if not (table_lines and table_lines[-1].strip() == ''):
                            table_lines.append('')
                        while i < len(lines) and lines[i].strip().startswith('|'):
                            table_lines.append(lines[i])
                            i += 1
                        continue

                if _is_new_section_header(stripped):
                    break
                break

            result.extend(table_lines)
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Промпты (общие для всех скриптов)
# ═══════════════════════════════════════════════════════════════════════════
# Перенесено из fix_formulas.py для прямого использования в pdf_to_md.py

def fix_subscripts(text: str) -> str:
    """Исправить подстрочные индексы в формулах ПУЭ.

    - Греческие буквы: α → \\\\alpha, с индексами αw → \\\\alpha_{w}
    - Русские буквы-индексы: Iн → I_{н}, Uпр → U_{пр}
    - Многосимвольные латинские индексы: Ki → K_{i}
    - Цифровые индексы: W0 → W_{0}, I1 → I_{1}

    ВАЖНО: не трогает содержимое Markdown-ссылок ![caption](path) и URL.
    Также не трогает текст вне math mode ($...$ или $$...$$).
    """
    # Изолируем Markdown-ссылки от замен
    LINK_PLACEHOLDER = "___FIX_SUB_MD_LINK___"
    links = []

    def save_link(m):
        links.append(m.group(0))
        return LINK_PLACEHOLDER

    text = re.sub(r'!\[.*?\]\(.*?\)', save_link, text)

    # Разбиваем на сегменты: внутри math mode и вне его
    # Применяем замены только внутри $$...$$ и $...$
    MATH_PLACEHOLDER = "___MATH_SEGMENT___"
    non_math_segments = []
    math_segments = []

    def split_math(text):
        """Разделить текст на math и non-math сегменты."""
        # Сначала двухдолларовые, потом однодолларовые
        parts = re.split(r'(\$\$.*?\$\$)', text, flags=re.DOTALL)
        result_parts = []
        for part in parts:
            if part.startswith('$$') and part.endswith('$$'):
                # Math mode $$...$$
                math_segments.append(part)
                result_parts.append(MATH_PLACEHOLDER)
            else:
                # Non-math — ищем однодолларовые
                subparts = re.split(r'(\$[^$]*?\$)', part)
                for sp in subparts:
                    if sp.startswith('$') and sp.endswith('$') and len(sp) > 2:
                        math_segments.append(sp)
                        result_parts.append(MATH_PLACEHOLDER)
                    else:
                        non_math_segments.append(sp)
                        result_parts.append(sp)
        return ''.join(result_parts)

    # Изолируем non-math текст
    isolated_text = split_math(text)

    # Применяем замены ТОЛЬКО к math-сегментам
    fixed_math = []
    for seg in math_segments:
        fixed = seg

        # Замена точки как умножения (но не десятичный разделитель между цифрами)
        fixed = re.sub(r'(?<=[A-Za-z])\.(?=[A-Za-z])', r'\\cdot ', fixed)

        # 1. Греческие буквы с латинскими индексами: αw → \\alpha_{w}
        greek_map = {
            'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
            'ε': 'epsilon', 'ζ': 'zeta', 'η': 'eta', 'θ': 'theta',
            'ι': 'iota', 'κ': 'kappa', 'λ': 'lambda', 'μ': 'mu',
            'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron', 'π': 'pi',
            'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
            'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
            'Θ': 'Theta', 'Φ': 'Phi', 'Γ': 'Gamma', 'Δ': 'Delta',
            'Λ': 'Lambda', 'Σ': 'Sigma', 'Ω': 'Omega', 'Π': 'Pi',
        }
        for gl, latex_name in sorted(greek_map.items(), key=lambda x: -len(x[0])):
            fixed = re.sub(rf'({gl})([a-zA-Z]{{1,4}})(?=\s|[+\-*/=,);]|$)', rf'\\{latex_name}_{{\2}}', fixed)
            fixed = re.sub(rf'({gl})(?=\s|[+\-*/=,);]|$)', rf'\\{latex_name}', fixed)

        # 2. Русские буквы: Iн → I_{н}, Uпр → U_{пр}
        fixed = re.sub(r'([A-Za-z])([а-яА-Я]{1,4})(?=\s|[+\-*/=,);]|$)', r'\1_{\2}', fixed)

        # 3. Латинская буква + 2+ строчные латинские: Ki → K_{i}
        fixed = re.sub(r'(?<!_)([A-Z])([a-z]{2,4})(?=\s|[+\-*/=,);]|$)', r'\1_{\2}', fixed)

        # 4. Цифровой индекс после буквы: W0 → W_{0}, I1 → I_{1}
        fixed = re.sub(r'([A-Za-zа-яА-Я])(\d)(?=\s|[+\-*/=,);]|$)', r'\1_{\2}', fixed)

        fixed_math.append(fixed)

    # Собираем текст обратно: заменяем плейсхолдеры на исправленные math-сегменты
    math_iter = iter(fixed_math)
    def restore_math(m):
        return next(math_iter)
    text = re.sub(MATH_PLACEHOLDER, restore_math, isolated_text)

    # Восстанавливаем Markdown-ссылки
    for link in links:
        text = text.replace(LINK_PLACEHOLDER, link, 1)

    return text


# ═══════════════════════════════════════════════════════════════════════════
# 7. Промпты (общие для всех скриптов)
# ═══════════════════════════════════════════════════════════════════════════

POSTPROCESS_PROMPT = r"""You are a professional document formatter. Clean up the following Markdown document.

CRITICAL: Preserve ALL document content — every paragraph, every heading, every table row, every list item.
Do NOT delete, summarize, or rephrase anything. Do NOT skip sections.

Rules:
- Fix broken or misaligned Markdown tables — ensure proper column alignment
- Merge table rows that were split across lines
- Fix nested lists and indentation
- Convert line-wrapped paragraphs into proper single-line paragraphs
- Preserve ALL text content — do not delete, summarize, or rephrase anything
- Remove only obvious page-level artifacts: standalone page numbers, running headers like
  "Продолжение таблицы X", "Таблица (продолжение)", and document-wide headers/footers only.
  Do NOT remove or alter section/subsection headings like 1.1.1, 2.3.5 — those are real document structure.
- KEEP all existing $...$ and $$...$$ LaTeX formulas exactly as they are — do not remove or alter them
- KEEP all existing subscript/superscript notation using _ and ^ exactly as in the source
- Do NOT add any new formulas or equations that are not already in the text
- IMPORTANT: PRESERVE ALL markdown image references like ![caption](path/to/image) — do NOT remove, move, or alter them
- Convert section headings to proper hierarchy (keep their text and numbering unchanged)
- Output ONLY the cleaned markdown, no commentary"""

FINAL_CLEANUP_PROMPT = r"""You are a document quality inspector. Review the following Markdown document and fix ONLY the following issues:

1. If you see "<!-- formula-not-decoded -->", "formula-not-decoded", garbled symbols like "Тг" — replace with correct LaTeX based on context.
2. If a sentence is broken into separate lines with hard breaks, merge them into proper paragraphs.
3. Do NOT change tables, lists, image references ![caption](path), or any other content.
4. Output ONLY the fixed markdown, no commentary."""
