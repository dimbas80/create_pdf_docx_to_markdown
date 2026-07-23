#!/usr/bin/env python3
"""
Create_Markdown.py — Единая точка входа для конвертации PDF/DOCX в Markdown.

Режимы:
  1. Одиночный файл:  python3 Create_Markdown.py -i document.pdf
                       python3 Create_Markdown.py -i document.docx
  2. Пакетный (batch): python3 Create_Markdown.py (с config.yaml batch.enabled=true)

Выходная папка: Markdown/имя_файла/ (рядом с исходным файлом)
                 или batch.output/имя_файла/ (при batch-режиме)
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
from utils import init_logging

init_logging(SCRIPT_DIR)
log = logging.getLogger("create-markdown")

CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def load_config() -> dict:
    """Загрузить config.yaml."""
    import yaml
    if not CONFIG_PATH.exists():
        log.warning(f"Конфиг не найден: {CONFIG_PATH}, используются дефолты")
        return {"mode": "auto", "batch": {"enabled": False}}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_file(file_path: Path, output_dir: Path, config: dict):
    """Обработать один файл через соответствующий скрипт."""
    ext = file_path.suffix.lower()
    script_name = "pdf_to_md.py" if ext == ".pdf" else "docx_to_md.py"
    script_path = SCRIPT_DIR / script_name

    if not script_path.exists():
        log.error(f"Скрипт не найден: {script_path}")
        return

    mode = config.get("mode", "auto")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Запускаем скрипт с тем же mode (manual → manual, auto → auto)
    child_mode = mode

    # Экспортируем API-ключи из конфига в окружение для дочерних процессов
    env = os.environ.copy()
    for section_key in ("vision", "postprocess"):
        section = config.get(section_key, {})
        for role in ("primary", "fallback"):
            sub = section.get(role, {})
            api_key_raw = sub.get("api_key", "").strip()
            if not api_key_raw:
                continue

            # Пытаемся разрешить ${VAR_NAME} — os.path.expandvars может не сработать,
            # если переменная не экспортирована, а задана через shopt/hermes tui
            api_key = os.path.expandvars(api_key_raw)

            # Если expandvars не смог (вернул то же самое или пусто),
            # пробуем прочитать из os.environ напрямую
            if not api_key or api_key == api_key_raw:
                import re as _re
                m = _re.match(r'\$\{?(\w+)\}?', api_key_raw)
                if m:
                    var_name = m.group(1)
                    api_key = os.environ.get(var_name, api_key_raw)

            # Если ключ всё ещё ссылка на переменную — пропускаем
            if api_key.startswith("${") or api_key.startswith("$"):
                log.warning(f"Переменная {api_key} не найдена в окружении")
                continue

            # Устанавливаем ключ в окружение для дочернего процесса
            provider = sub.get("provider", "")
            if provider == "provod" or not provider:
                env["PROVOD_API_KEY"] = api_key
            elif provider == "openai":
                env["OPENAI_API_KEY"] = api_key
            elif provider == "anthropic":
                env["ANTHROPIC_API_KEY"] = api_key

    cmd = [
        sys.executable, str(script_path),
        "-i", str(file_path),
        "-o", str(output_dir),
        "--mode", child_mode,
        "--config", str(CONFIG_PATH),
    ]

    log.info(f"{'='*60}")
    log.info(f"Файл: {file_path.name}")
    log.info(f"Скрипт: {script_name} | Режим: {child_mode}")
    log.info(f"Выход: {output_dir}/")

    result = subprocess.run(cmd, cwd=SCRIPT_DIR, env=env)
    if result.returncode != 0:
        log.error(f"Ошибка обработки {file_path.name} (exit={result.returncode})")


def main():
    config = load_config()
    batch_cfg = config.get("batch", {})
    mode = config.get("mode", "auto")

    # Парсим CLI аргументы
    parser = argparse.ArgumentParser(
        description="Конвертация PDF/DOCX в Markdown",
        epilog="Примеры:\n"
               "  python3 Create_Markdown.py -i document.pdf\n"
               "  python3 Create_Markdown.py (batch-режим из config.yaml)",
    )
    parser.add_argument("-i", "--input", help="Входной файл (.pdf / .docx)")
    parser.add_argument("-o", "--output", default=None,
                        help="Выходная папка (по умолчанию Markdown/имя_файла/)")
    args = parser.parse_args()

    # ── Режим 1: Одиночный файл ──────────────────────────────────────
    if args.input:
        file_path = Path(args.input).resolve()
        if not file_path.exists():
            log.error(f"Файл не найден: {file_path}")
            sys.exit(1)

        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = file_path.parent / "Markdown" / file_path.stem

        process_file(file_path, output_dir, config)
        return

    # ── Режим 2: Пакетная обработка ──────────────────────────────────
    if batch_cfg.get("enabled"):
        input_dir = Path(batch_cfg["input"])
        output_base = Path(batch_cfg["output"])

        if not input_dir.exists():
            log.error(f"Входная папка не найдена: {input_dir}")
            sys.exit(1)

        files = sorted(input_dir.rglob("*.pdf")) + sorted(input_dir.rglob("*.docx"))
        log.info(f"Найдено файлов: {len(files)}")
        log.info(f"Вход:  {input_dir}")
        log.info(f"Выход: {output_base}/")

        if not files:
            log.info("Нет PDF/DOCX для обработки")
            return

        # Если mode=manual — запрос подтверждения для batch
        if mode == "manual":
            print(f"\nБудет обработано: {len(files)} файлов")
            ans = input("Начать обработку? [Y/n]: ")
            if ans.lower() == "n":
                log.info("Отменено")
                return

        for f in files:
            output_dir = output_base / f.stem
            process_file(f, output_dir, config)

        log.info(f"\n✅ Пакетная обработка завершена: {len(files)} файлов")
        return

    # ── Режим 3: Нет аргументов, batch отключён → справка ───────────
    parser.print_help()
    print(f"\nИли включите batch в {CONFIG_PATH}")


if __name__ == "__main__":
    main()
