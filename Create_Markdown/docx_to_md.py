#!/usr/bin/env python3
"""
Конвертация .docx → Markdown.
Docling парсинг + извлечение изображений + Vision API классификация + Claude постобработка.

Использование:
  python3 docx_to_md.py -i document.docx
  python3 docx_to_md.py -i document.docx --mode auto
"""

import logging, os, re, sys, time
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from utils import (
    call_vision,
    call_claude,
    fix_subscripts,
    init_logging,
    merge_broken_tables,
    save_intermediate,
    chunk_text,
    reset_token_stats,
    get_token_stats,
    POSTPROCESS_PROMPT,
)

init_logging()
log = logging.getLogger("docx-to-md")


def _resolve_env(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_config(config_path: str) -> dict:
    cfg = {}
    if yaml is None:
        log.warning("yaml не установлен, используются дефолты")
        return cfg
    path = Path(config_path)
    if not path.exists():
        log.warning(f"Конфиг не найден: {config_path}")
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    vis = data.get("vision", {})
    for role in ("primary", "fallback"):
        section = vis.get(role, {})
        cfg[f"vision_{role}_model"] = _resolve_env(section.get("model", ""))
    post = data.get("postprocess", {})
    for role in ("primary", "fallback"):
        section = post.get(role, {})
        cfg[f"post_{role}_model"] = _resolve_env(section.get("model", ""))
    cfg["mode"] = data.get("mode", "auto")
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# 1. Парсинг .docx через Docling
# ═══════════════════════════════════════════════════════════════════════════

def parse_docx(docx_path: str) -> tuple[str, list[dict]]:
    """Парсинг .docx через Docling. Возвращает (raw_markdown, список изображений)."""
    from docling.document_converter import DocumentConverter

    log.info("Парсинг .docx через Docling...")
    start = time.time()
    conv = DocumentConverter()
    result = conv.convert(docx_path)
    elapsed = time.time() - start

    md = result.document.export_to_markdown()
    log.info(f"  Docling: {elapsed:.1f}с, {len(md)} символов raw")

    pics = list(result.document.pictures)
    images = []
    for i, pic in enumerate(pics):
        ref = pic.image
        if hasattr(ref, 'pil_image') and ref.pil_image is not None:
            images.append({
                "idx": i + 1,
                "pil_image": ref.pil_image,
                "size": ref.pil_image.size,
            })

    log.info(f"  Изображений: {len(images)}")
    return md, images, result


def estimate_tokens_claude(md_text: str) -> tuple[int, int]:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = len(enc.encode(md_text))
    except Exception:
        prompt_tokens = len(md_text) // 3
    completion_tokens = int(prompt_tokens * 0.85)
    return prompt_tokens, completion_tokens


# ═══════════════════════════════════════════════════════════════════════════
# 2. Обработка изображений (классификация через Vision API)
# ═══════════════════════════════════════════════════════════════════════════

def save_and_classify_images(images: list[dict], img_dir: Path) -> list[dict]:
    """Сохранить изображения, классифицировать через Vision API."""
    img_dir.mkdir(exist_ok=True)
    results = []

    for img in images:
        w, h = img["size"]
        fname = img_dir / f"pic_{img['idx']}.png"
        img["pil_image"].save(str(fname))

        log.info(f"  pic_{img['idx']}.png: {w}x{h} — Vision классификация...")

        try:
            result = call_vision(str(fname), mode="classify", label=f"pic_{img['idx']}")
        except Exception as e:
            log.warning(f"  Vision API ошибка: {e}, сохраняю как изображение")
            result = {"type": "IMAGE"}

        if result.get("type") == "ERROR":
            log.warning(f"  Vision API: {result.get('error')}, сохраняю как изображение")
            result = {"type": "IMAGE"}

        if result["type"] == "FORMULA":
            latex = result["latex"]
            # Фильтр: отбрасываем названия переменных без операторов
            has_op = bool(re.search(r'[=+\-*/\\^<>]|\\cdot|\\frac|\\sum|\\prod|\\int', latex))
            if not has_op and len(latex.strip("$ ;,.")) < 15:
                log.info(f"    → ПРОПУЩЕН (без оператора): {latex[:60]}")
                continue
            results.append({"type": "latex", "latex": latex, "idx": img["idx"]})
            log.info(f"    → ФОРМУЛА: {latex[:120]}")
        else:
            rel_path = f"image/pic_{img['idx']}.png"
            results.append({"type": "image", "filename": rel_path, "idx": img["idx"]})
            log.info(f"    → ИЗОБРАЖЕНИЕ")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. Вставка изображений и формул в MD
# ═══════════════════════════════════════════════════════════════════════════

def inject_into_md(content: str, results: list[dict]) -> str:
    """Заменить <!-- image --> плейсхолдеры на изображения/формулы."""
    img_iter = iter(results)

    def replace_placeholder(m):
        try:
            item = next(img_iter)
            if item["type"] == "latex":
                return f"\n{item['latex']}\n"
            else:
                return f"![Рисунок]({item['filename']})"
        except StopIteration:
            return m.group(0)

    return re.sub(r'<!--\s*image\s*-->', replace_placeholder, content)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DOCX → Markdown через Docling + Vision API")
    parser.add_argument("-i", "--input", required=True, help="Входной .docx файл")
    parser.add_argument("-o", "--output", default=None,
                        help="Выходная папка (по умолчанию Markdown/имя_файла)")
    parser.add_argument("--mode", choices=["auto", "manual"], default=None,
                        help="auto — без подтверждения, manual — с оценкой токенов")
    parser.add_argument("--config", default=None, help="Путь к config.yaml")
    parser.add_argument("--debug", action="store_true", help="Подробное логирование")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    reset_token_stats()

    # Конфиг
    cfg = {}
    if args.config:
        cfg = load_config(args.config)
        # Применяем настройки из конфига
        if cfg:
            for key_name, cfg_key in [("PROVOD_API_KEY", "vision_primary_api_key"),
                                       ("OPENAI_API_KEY", "vision_fallback_api_key")]:
                val = cfg.get(cfg_key)
                if val:
                    os.environ[key_name] = val
                    import utils as _u
                    _u.PROVOD_API_KEY = os.environ.get("PROVOD_API_KEY", "")

            vision_models = []
            for role in ("vision_primary_model", "vision_fallback_model"):
                model = cfg.get(role)
                if model:
                    vision_models.append(model)
            if vision_models:
                import utils as _u
                _u.VISION_MODELS = vision_models

            post_models = []
            for role in ("post_primary_model", "post_fallback_model"):
                model = cfg.get(role)
                if model:
                    post_models.append(model)
            if post_models:
                import utils as _u
                _u.TEXT_MODELS = post_models

    mode = args.mode or cfg.get("mode", "auto")

    docx_path = Path(args.input)
    if not docx_path.exists():
        log.error(f"Файл не найден: {docx_path}")
        sys.exit(1)

    out_dir = Path(args.output) if args.output else \
        docx_path.parent / "Markdown" / docx_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "image"

    stem = docx_path.stem
    raw_path = out_dir / f"{stem}_raw.md"
    final_path = out_dir / f"{stem}.md"

    log.info(f"{'='*60}")
    log.info(f"DOCX → Markdown конвертер")
    log.info(f"Режим: {mode}")
    log.info(f"Вход:  {docx_path}")
    log.info(f"Выход: {final_path}")
    log.info(f"{'='*60}")

    # Шаг 1: Парсинг .docx
    log.info("Шаг 1: Парсинг .docx")
    raw_md, images, result = parse_docx(str(docx_path))
    save_intermediate(str(raw_path), raw_md, "Raw Docling")
    log.info(f"  Raw размер: {len(raw_md)} символов")

    # Оценка токенов (если manual)
    if mode == "manual":
        prompt_tok, comp_tok = estimate_tokens_claude(raw_md)
        vision_tok = len(images) * 250
        print(f"\n{'='*56}")
        print(f"  ОЦЕНКА РАСХОДА ТОКЕНОВ")
        print(f"  {'─'*54}")
        print(f"  Vision API (классификация):    ~{vision_tok:>6} токенов ({len(images)} изобр.)")
        print(f"  Claude (постобработка in):     ~{prompt_tok:>6} токенов")
        print(f"  Claude (постобработка out):    ~{comp_tok:>6} токенов")
        print(f"  {'─'*54}")
        print(f"  Всего: ~{prompt_tok + comp_tok + vision_tok} токенов")
        print(f"{'='*56}")
        choice = input("  Продолжить обработку? [Y/n]: ")
        if choice.lower() == "n":
            log.info("Отменено пользователем")
            return

    # Шаг 2: Классификация изображений
    log.info("=" * 50)
    log.info("Шаг 2: Классификация изображений через Vision API")
    vision_results = save_and_classify_images(images, img_dir)

    # Шаг 3: Вставка в MD
    log.info("=" * 50)
    log.info("Шаг 3: Вставка изображений и формул")
    content = inject_into_md(raw_md, vision_results)
    log.info(f"  После inject: {len(content)} символов")

    formulas = [r for r in vision_results if r["type"] == "latex"]
    pictures = [r for r in vision_results if r["type"] == "image"]
    log.info(f"  Формул (LaTeX): {len(formulas)}, изображений: {len(pictures)}")

    # Шаг 4: Regex-чистка формул
    log.info("=" * 50)
    log.info("Шаг 4: Regex-чистка LaTeX формул")
    content = fix_subscripts(content)
    log.info(f"  После fix_subscripts: {len(content)} символов")

    # Шаг 4b: Склейка разорванных таблиц
    content = merge_broken_tables(content)
    log.info(f"  После merge_broken_tables: {len(content)} символов")

    # Шаг 5: Claude постобработка
    log.info("=" * 50)
    log.info("Шаг 5: Claude постобработка")

    if len(content) > 25000:
        log.info(f"  Документ большой ({len(content)} символов), чанкование")
        batches = chunk_text(content, max_chars=20000)
        log.info(f"  Батчей: {len(batches)}")

        chunked_results = []
        for i, batch in enumerate(batches):
            log.info(f"  Батч [{i+1}/{len(batches)}] ({len(batch)} символов)")
            cleaned = call_claude(POSTPROCESS_PROMPT, batch, f"Claude батч {i+1}")
            chunked_results.append(cleaned)

        cleaned = "\n\n---\n\n".join(chunked_results)
    else:
        cleaned = call_claude(POSTPROCESS_PROMPT, content, "Claude постобработка")

    final_path.write_text(cleaned.strip(), "utf-8")
    log.info(f"✅ {final_path} ({len(cleaned)} chars)")

    # Cleanup промежуточных файлов — сохраняем в tmp/
    import shutil
    from datetime import datetime
    tmp_dir = Path(__file__).parent.resolve() / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    patterns = [
        f"{stem}_raw.md",
        f"{stem}_raw.md.bak",
    ]
    moved = 0
    for name in patterns:
        p = out_dir / name
        if p.exists():
            tmp_name = f"{timestamp}_{name}"
            shutil.copy2(str(p), str(tmp_dir / tmp_name))
            p.unlink()
            moved += 1
            log.debug(f"  Промежуточный {name} → tmp/{tmp_name}")
    if moved:
        log.info(f"  Промежуточных файлов перемещено в tmp/: {moved}")

    # Статистика
    stats = get_token_stats()
    log.info(f"  Фактический расход: in={stats['prompt_tokens']} out={stats['completion_tokens']} vision={stats['vision_calls']}×")
    log.info(f"\nГотово: {final_path}")


if __name__ == "__main__":
    main()
