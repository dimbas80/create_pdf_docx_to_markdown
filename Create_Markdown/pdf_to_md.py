#!/usr/bin/env python3
"""
pdf_to_md.py — Конвертация PDF в Markdown (Docling + Claude).

Режимы:
  --mode postprocess (1) Docling парсит локально, затем Claude чистит (рекомендуемый)
  --mode vlm         (2) Claude смотрит каждую страницу как картинку (точнее для схем)

Изображения:
  - С подписью "Рисунок X.Y" → сохраняются как PNG, ссылка в MD
  - Без подписи → Vision API классификация (формула или схема)
    - Формула → LaTeX через Vision API
    - Схема/график → PNG, ссылка в MD

Оценка токенов (до обработки):
  После Docling (бесплатно) показывает примерный расход токенов и запрашивает
  подтверждение. Флаг --yes для автоматического режима.

Использование:
  python3 pdf_to_md.py -i document.pdf
  python3 pdf_to_md.py -i document.pdf --yes
  python3 pdf_to_md.py -i document.pdf --mode vlm
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from utils import (
    call_claude,
    call_vision,
    chunk_text,
    fix_subscripts,
    init_logging,
    merge_broken_tables,
    needs_cleanup,
    save_intermediate,
    reset_token_stats,
    get_token_stats,
    POSTPROCESS_PROMPT,
    FINAL_CLEANUP_PROMPT,
)

init_logging()
log = logging.getLogger("pdf-to-md")


# ═══════════════════════════════════════════════════════════════════════════
# 0. Загрузка конфига
# ═══════════════════════════════════════════════════════════════════════════

def _resolve_env(value):
    """Подставить ${VAR_NAME} из переменных окружения."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_config(config_path: str) -> dict:
    """Загрузить конфиг, вернуть словарь с настройками."""
    cfg = {}
    if yaml is None:
        log.warning("yaml не установлен (`pip install pyyaml`), используются дефолты")
        return cfg

    path = Path(config_path)
    if not path.exists():
        log.warning(f"Конфиг не найден: {config_path}, используются дефолты")
        return cfg

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Vision
    vis = data.get("vision", {})
    for role in ("primary", "fallback"):
        section = vis.get(role, {})
        cfg[f"vision_{role}_model"] = _resolve_env(section.get("model", ""))
        cfg[f"vision_{role}_api_key"] = _resolve_env(section.get("api_key", ""))
        cfg[f"vision_{role}_base_url"] = _resolve_env(section.get("base_url", ""))

    # Postprocess
    post = data.get("postprocess", {})
    for role in ("primary", "fallback"):
        section = post.get(role, {})
        cfg[f"post_{role}_model"] = _resolve_env(section.get("model", ""))
        cfg[f"post_{role}_api_key"] = _resolve_env(section.get("api_key", ""))
        cfg[f"post_{role}_base_url"] = _resolve_env(section.get("base_url", ""))

    # Mode
    cfg["mode"] = data.get("mode", "auto")

    log.info(f"Конфиг загружен: {config_path}")
    return cfg

# ═══════════════════════════════════════════════════════════════════════════
# 0. Оценка токенов
# ═══════════════════════════════════════════════════════════════════════════

def count_image_blocks(pdf_path: str) -> int:
    """Быстро подсчитать количество IMAGE-блоков в PDF (без Vision API)."""
    import fitz
    doc = fitz.open(pdf_path)
    count = 0
    for pno in range(len(doc)):
        page = doc[pno]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 1:
                bbox = block["bbox"]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w >= 10 and h >= 10 and max(w, h) <= 30:
                    pass  # мелкий мусор — не считаем для Vision
                elif w >= 10 and h >= 10:
                    count += 1
    doc.close()
    return count

def estimate_tokens_claude(md_text: str) -> tuple[int, int]:
    """Оценить токены для Claude постобработки: (prompt, completion)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = len(enc.encode(md_text))
    except Exception:
        # fallback: грубая оценка для кириллицы (~3.5 символа/токен)
        prompt_tokens = len(md_text) // 3
    
    completion_tokens = int(prompt_tokens * 0.85)  # эмпирически: out ~85% от in
    return prompt_tokens, completion_tokens


# ═══════════════════════════════════════════════════════════════════════════
# 1. Извлечение и классификация изображений из PDF
# ═══════════════════════════════════════════════════════════════════════════

def _extract_pictures(result, output_dir: Path, input_path: str) -> list[dict]:
    """Извлечь изображения из PDF, классифицировать и обработать.

    Ищет IMAGE-блоки напрямую через fitz (Docling не находит мелкие объекты).
    Для каждого изображения:
    1. Есть подпись "Рисунок X.Y" рядом? → IMAGE, сохраняется PNG
    2. Нет подписи? → Vision API классификация:
       - FORMULA → LaTeX встраивается в MD
       - IMAGE + размер > 30×30 → PNG, ссылка в MD
       - IMAGE + размер ≤ 30×30 → игнорируется (шум/мусор)

    Returns:
        Список словарей: {"type": "image"/"latex", "filename": ..., 
                          "latex": ..., "caption": ..., "page_no": ...}
    """
    import fitz

    img_dir = output_dir / "images"
    img_dir.mkdir(exist_ok=True)
    pdf_doc = fitz.open(input_path)
    saved = []
    DPI = 300

    for pno in range(len(pdf_doc)):
        page = pdf_doc[pno]
        blocks = page.get_text("dict")["blocks"]

        for bidx, block in enumerate(blocks):
            if block["type"] != 1:
                continue

            bbox = block["bbox"]
            img_w = bbox[2] - bbox[0]
            img_h = bbox[3] - bbox[1]

            # Мусор: слишком мелкие блоки (12x13 и меньше) — без Vision
            if max(img_w, img_h) < 15:
                log.debug(f"  block[{bidx}]: пропущен (мелкий {img_w:.0f}x{img_h:.0f})")
                continue

            # Anchor: текст непосредственно перед изображением (5-40px выше)
            anchor_clip = fitz.Rect(
                max(0, bbox[0] - 5),
                max(0, bbox[1] - 40),
                min(page.rect.width, bbox[2] + 5),
                max(0, bbox[1] - 2)
            )
            anchor_text = ""
            if anchor_clip.height > 0:
                raw = page.get_text("text", clip=anchor_clip).strip()
                # Берём ПОСЛЕДНЮЮ непустую строку целиком (без обрезания)
                if raw:
                    lines = [l.strip() for l in raw.split('\n') if l.strip()]
                    anchor_text = lines[-1] if lines else ""

            # Контекст вокруг изображения (поиск подписи "Рисунок")
            context_clip = fitz.Rect(
                max(0, bbox[0] - 10),
                max(0, bbox[1] - 100),
                min(page.rect.width, bbox[2] + 10),
                min(page.rect.height, bbox[3] + 50)
            )
            context_text = page.get_text("text", clip=context_clip).strip()
            fig_match = re.search(r'(?:Рисунок|Figure|fig\.)\s*([\d.]+[\w]?)', context_text)

            # Вырезаем область
            clip = fitz.Rect(bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5) & page.rect

            if fig_match:
                # Есть подпись → схема/график
                fig_number = fig_match.group(1)
                safe = fig_number.replace(".", "_").lower()
                fname = img_dir / f"fig_{safe}.png"
                pix = page.get_pixmap(dpi=DPI, clip=clip)
                pix.save(str(fname))
                rel_path = f"images/{fname.name}"
                saved.append({
                    "type": "image",
                    "filename": rel_path,
                    "caption": f"Рисунок {fig_number}",
                    "page_no": pno + 1,
                    "anchor": anchor_text,
                })
                log.info(f"  {fname.name}: СХЕМА — стр.{pno+1} {img_w:.0f}x{img_h:.0f} [{context_text[:80]}]")
            else:
                # Нет подписи → Vision классификация
                tmp_fname = img_dir / f"_tmp_p{pno+1}_b{bidx}.png"
                pix = page.get_pixmap(dpi=DPI, clip=clip)
                pix.save(str(tmp_fname))

                log.info(f"  _tmp_p{pno+1}_b{bidx}.png: стр.{pno+1} {img_w:.0f}x{img_h:.0f} — Vision классификация...")

                try:
                    result_vision = call_vision(
                        str(tmp_fname),
                        mode="classify",
                        label=f"p{pno+1}_b{bidx}",
                    )
                except Exception as e:
                    log.warning(f"  Vision API ошибка: {e}, сохраняю как изображение")
                    result_vision = {"type": "IMAGE"}

                if result_vision.get("type") == "ERROR":
                    log.warning(f"  Vision API: {result_vision.get('error', 'неизвестная ошибка')}, сохраняю как изображение")
                    result_vision = {"type": "IMAGE"}

                if result_vision["type"] == "FORMULA":
                    latex = result_vision["latex"]
                    # Фильтр: отбрасываем "формулы" без операторов (названия переменных)
                    has_operator = bool(re.search(r'[=+\-*/\\^<>]|\\cdot|\\frac|\\sum|\\prod|\\int', latex))
                    if not has_operator and len(latex.strip("$ ;,.")) < 15:
                        log.info(f"    → ПРОПУЩЕН (без оператора): {latex[:60]}")
                        tmp_fname.unlink(missing_ok=True)
                        continue
                    saved.append({
                        "type": "latex",
                        "latex": latex,
                        "page_no": pno + 1,
                        "anchor": anchor_text,
                    })
                    log.info(f"    → ФОРМУЛА: {latex[:120]}")
                    tmp_fname.unlink(missing_ok=True)
                elif max(img_w, img_h) > 30:
                    fname = img_dir / f"img_p{pno+1}_b{bidx}.png"
                    pix = page.get_pixmap(dpi=DPI, clip=clip)
                    pix.save(str(fname))
                    rel_path = f"images/{fname.name}"
                    tmp_fname.unlink(missing_ok=True)
                    saved.append({
                        "type": "image",
                        "filename": rel_path,
                        "caption": f"Иллюстрация — стр. {pno+1}",
                        "page_no": pno + 1,
                        "anchor": anchor_text,
                    })
                    log.info(f"    → ИЗОБРАЖЕНИЕ: {fname.name}")
                else:
                    tmp_fname.unlink(missing_ok=True)
                    log.info(f"    → МУСОР: пропущен ({img_w:.0f}x{img_h:.0f})")

    pdf_doc.close()

    if not saved:
        log.info("  Изображений не найдено")

    return saved


def _inject_images_into_md(content: str, images: list[dict]) -> str:
    """Вставить изображения и формулы в markdown.

    Сначала заменяет плейсхолдеры <!-- image --> (от Docling).
    Оставшиеся (найденные через fitz без плейсхолдеров) — вставляет
    по anchor-тексту (последняя строка текста перед изображением).
    """
    if not images:
        return content

    # Собираем все замены в порядке original order
    all_replacements = []
    for img in images:
        if img["type"] == "latex":
            all_replacements.append(f"\n{img['latex']}\n")
        else:
            all_replacements.append(f"![{img['caption']}]({img['filename']})")

    # Шаг 1: Заменяем <!-- image --> плейсхолдеры по порядку
    replace_iter = iter(all_replacements)
    def replace_placeholder(m):
        try:
            return next(replace_iter)
        except StopIteration:
            return m.group(0)

    new_content = re.sub(r'<!--\s*image\s*-->', replace_placeholder, content)

    # Шаг 2: Оставшиеся (без плейсхолдеров) — вставляем по anchor-тексту
    remaining_replacements = list(replace_iter)
    if not remaining_replacements:
        return new_content

    remaining_images = images[len(images) - len(remaining_replacements):]

    lines = new_content.split("\n")
    extra_lines = []

    for img, repl in zip(remaining_images, remaining_replacements):
        anchor = img.get("anchor", "")
        page_no = img.get("page_no")

        if anchor:
            found = False
            for i, line in enumerate(lines):
                # Ищем точное совпадение КОНЦА строки
                if line.strip().endswith(anchor) and i < len(lines) - 1:
                    # Проверяем, что не вставили уже после этой строки
                    if i + 2 < len(lines) and repl.strip() in lines[i + 2]:
                        continue  # уже вставлено
                    # Пропускаем пустые строки после найденной
                    insert_at = i + 1
                    while insert_at < len(lines) and lines[insert_at].strip() == "":
                        insert_at += 1  # пропускаем пустые строки после найденной
                    lines.insert(insert_at, "")
                    lines.insert(insert_at + 1, repl.strip())
                    found = True
                    break
            if found:
                continue

        if page_no:
            extra_lines.append(f"<!-- стр. {page_no} -->\n{repl.strip()}")
        else:
            extra_lines.append(repl.strip())

    if extra_lines:
        lines.extend([""] + extra_lines)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Режим: VLM-бэкенд
# ═══════════════════════════════════════════════════════════════════════════

VLM_PROMPT = r"""You are a professional document converter. Convert this document page to clean Markdown.

Rules:
- Preserve ALL text exactly as written, including headers, footers, page numbers
- Convert tables to Markdown table syntax (pipe-delimited)
- Preserve lists, numbering, bullet points
- CRITICAL: Render ALL formulas as LaTeX using $$...$$ or $...$ — pay attention to subscripts, superscripts, Greek letters, fractions, square roots, integrals
- DO NOT add any commentary outside the markdown
- DO NOT wrap the output in code fences"""

VLM_MODELS = [
    "anthropic/claude-sonnet-5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.8",
    "openai/gpt-5.4",
]


def mode_vlm(input_path: str, output_dir: Path) -> str:
    """VLM pipeline: Claude видит каждую страницу как картинку.

    Пробует модели из VLM_MODELS по порядку. Если ни одна не доступна,
    падает в обычный postprocess.
    """
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
    from docling.document_converter import DocumentConverter
    from utils import PROVOD_URL as URL, PROVOD_API_KEY as KEY

    stem = Path(input_path).stem

    for model in VLM_MODELS:
        log.info(f"Режим VLM: пробую модель {model}...")
        try:
            vlm_opts = ApiVlmOptions(
                prompt=VLM_PROMPT,
                response_format=ResponseFormat.MARKDOWN,
                url=URL,
                headers={"Authorization": f"Bearer {KEY}"},
                params={"model": model},
                temperature=0.0,
                timeout=300,
                concurrency=1,
                scale=2.0,
            )

            pipeline_opts = VlmPipelineOptions()
            pipeline_opts.vlm_options = vlm_opts
            pipeline_opts.generate_page_images = True

            converter = DocumentConverter(pipeline_options=pipeline_opts)
            start = time.time()
            result = converter.convert(input_path)
            log.info(f"VLM ({model}): {time.time()-start:.0f}с")

            content = result.document.export_to_markdown()
            raw_path = output_dir / f"{stem}_vlm_raw.md"
            raw_path.write_text(content, encoding="utf-8")
            log.info(f"VLM raw: {raw_path} ({len(content)} символов)")

            if needs_cleanup(content):
                content = call_claude(FINAL_CLEANUP_PROMPT, content, "Финальная чистка")

            final_path = output_dir / f"{stem}_vlm.md"
            final_path.write_text(content, encoding="utf-8")
            log.info(f"VLM final: {final_path} ({len(content)} символов)")

            _cleanup_intermediate(output_dir, stem)

            return str(final_path)

        except Exception as e:
            log.warning(f"VLM ({model}): {e}")
            continue

    log.warning("VLM модели недоступны, переключаюсь в postprocess")
    return mode_postprocess(input_path, output_dir)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Режим: Docling (локально) + Formula regex + Vision классификация + Claude
# ═══════════════════════════════════════════════════════════════════════════

def run_docling(input_path: str):
    """Запустить Docling, вернуть (result, raw_md)."""
    from docling.document_converter import DocumentConverter

    log.info("Docling парсинг...")
    converter = DocumentConverter()
    start = time.time()
    result = converter.convert(input_path)
    log.info(f"Docling: {time.time()-start:.0f}с")
    raw_md = result.document.export_to_markdown()
    return result, raw_md


def mode_postprocess(input_path: str, output_dir: Path) -> str:
    """Docling парсит → fix_subscripts → Vision классификация → Claude."""
    from utils import reset_token_stats, get_token_stats

    reset_token_stats()

    # Шаг 1: Docling
    result, raw_md = run_docling(input_path)
    stem = Path(input_path).stem

    # ── Шаг 2: raw Markdown + изображения ─────────────────────────────
    log.info("Шаг 2: Извлечение и классификация изображений")
    images = _extract_pictures(result, output_dir, input_path)
    raw_md = _inject_images_into_md(raw_md, images)

    # ── Шаг 2b: Склейка разорванных таблиц ──────────────────────────
    raw_md = merge_broken_tables(raw_md)

    raw_path = output_dir / f"{stem}_docling_raw.md"
    raw_path.write_text(raw_md, encoding="utf-8")
    log.info(f"  Raw: {raw_path} ({len(raw_md)} символов)")

    # ── Шаг 3: Regex-чистка формул ────────────────────────────────────
    log.info("Шаг 3: Regex-чистка LaTeX формул")
    fixed_md = fix_subscripts(raw_md)
    fixed_path = output_dir / f"{stem}_fixed.md"
    save_intermediate(str(fixed_path), fixed_md, "Fixed subscripts")
    log.info(f"  Fix subscripts: готово")

    # ── Шаг 4: Claude постобработка с чанкованием ────────────────────
    log.info("Шаг 4: Claude постобработка")

    if len(fixed_md) > 25000:
        log.info(f"  Документ большой ({len(fixed_md)} символов), чанкование")
        batches = chunk_text(fixed_md, max_chars=20000)
        log.info(f"  Батчей: {len(batches)}")

        chunked_results = []
        for i, batch in enumerate(batches):
            log.info(f"  Батч [{i+1}/{len(batches)}] ({len(batch)} символов)")
            cleaned = call_claude(POSTPROCESS_PROMPT, batch, f"Claude батч {i+1}")
            chunked_results.append(cleaned)

        cleaned_md = "\n\n---\n\n".join(chunked_results)
    else:
        cleaned_md = call_claude(POSTPROCESS_PROMPT, fixed_md, "Claude постобработка")

    post_path = output_dir / f"{stem}_postprocessed.md"
    save_intermediate(str(post_path), cleaned_md, "Postprocessed")

    # ── Шаг 5: Финальная чистка (с чанкованием) ──────────────────────
    log.info("Шаг 5: Финальная чистка")
    if needs_cleanup(cleaned_md):
        log.info("  Есть нераспознанные формулы, запускаю финальную чистку...")
        if len(cleaned_md) > 25000:
            batches = chunk_text(cleaned_md, max_chars=20000)
            log.info(f"  Батчей финальной чистки: {len(batches)}")
            chunked_results = []
            for i, batch in enumerate(batches):
                log.info(f"  Финальный батч [{i+1}/{len(batches)}] ({len(batch)} символов)")
                cleaned = call_claude(FINAL_CLEANUP_PROMPT, batch, f"Финальная чистка батч {i+1}")
                chunked_results.append(cleaned)
            final_md = "\n\n---\n\n".join(chunked_results)
        else:
            final_md = call_claude(FINAL_CLEANUP_PROMPT, cleaned_md, "Финальная чистка")
        if len(final_md) < len(cleaned_md) * 0.5:
            log.warning("  Финальная чистка слишком короткая, использую postprocessed")
            final_md = cleaned_md
    else:
        log.info("  Нет маркеров нераспознанных формул")
        final_md = cleaned_md

    # ── Шаг 6: Сохранение финала ─────────────────────────────────────
    final_path = output_dir / f"{stem}.md"
    final_path.write_text(final_md, encoding="utf-8")
    log.info(f"✅ Final: {final_path} ({len(final_md)} символов)")

    img_dir = output_dir / "images"
    if img_dir.exists():
        png_count = len(list(img_dir.glob("*.png")))
        formula_count = sum(1 for img in images if img["type"] == "latex")
        image_count = sum(1 for img in images if img["type"] == "image")
        log.info(f"  Изображения: {img_dir}/ ({png_count} PNG, {image_count} изображений, {formula_count} формул LaTeX)")

    # ── Статистика токенов ───────────────────────────────────────────
    stats = get_token_stats()
    log.info(f"  Токены: in={stats['prompt_tokens']} out={stats['completion_tokens']} vision={stats['vision_calls']}×")

    _cleanup_intermediate(output_dir, stem)
    _print_stats(result)
    return str(final_path)


def _cleanup_intermediate(output_dir: Path, stem: str) -> None:
    """Скопировать промежуточные файлы в tmp/ и удалить из output_dir."""
    import shutil
    from datetime import datetime

    tmp_dir = Path(__file__).parent.resolve() / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    patterns = [
        f"{stem}_docling_raw.md",
        f"{stem}_docling_raw.md.bak",
        f"{stem}_vlm_raw.md",
        f"{stem}_vlm_raw.md.bak",
        f"{stem}_vlm.md",
        f"{stem}_vlm.md.bak",
        f"{stem}_fixed.md",
        f"{stem}_fixed.md.bak",
        f"{stem}_postprocessed.md",
        f"{stem}_postprocessed.md.bak",
    ]
    moved = 0
    for name in patterns:
        p = output_dir / name
        if p.exists():
            tmp_name = f"{timestamp}_{name}"
            shutil.copy2(str(p), str(tmp_dir / tmp_name))
            p.unlink()
            moved += 1
            log.debug(f"  Промежуточный {name} → tmp/{tmp_name}")
    if moved:
        log.info(f"  Промежуточных файлов перемещено в tmp/: {moved}")


def process_with_confirm(input_path: str, output_dir: Path, args) -> str | None:
    """Полный пайплайн с оценкой токенов и подтверждением.

    Returns:
        Путь к выходному файлу или None при отмене.
    """
    reset_token_stats()

    # ── Шаг 1: Docling (бесплатно) ────────────────────────────────────
    result, raw_md = run_docling(input_path)
    stem = Path(input_path).stem

    # ── Шаг 2: Оценка токенов (без Vision API) ───────────────────────
    if not args.yes:
        img_count = count_image_blocks(input_path)
        prompt_tok, comp_tok = estimate_tokens_claude(raw_md)
        vision_tok = img_count * 250

        print(f"\n{'='*56}")
        print(f"  ОЦЕНКА РАСХОДА ТОКЕНОВ")
        print(f"  {'─'*54}")
        print(f"  Vision API (классификация):    ~{vision_tok:>6} токенов ({img_count} изобр.)")
        print(f"  Claude (постобработка in):     ~{prompt_tok:>6} токенов")
        print(f"  Claude (постобработка out):    ~{comp_tok:>6} токенов")
        print(f"  {'─'*54}")
        print(f"  Всего: ~{prompt_tok + comp_tok + vision_tok} токенов")
        print(f"{'='*56}")
        choice = input("  Продолжить обработку? [Y/n]: ")
        if choice.lower() == "n":
            log.info("Отменено пользователем")
            return None

    # ── Шаг 3: Полный пайплайн ────────────────────────────────────────
    images = _extract_pictures(result, output_dir, input_path)
    raw_md = _inject_images_into_md(raw_md, images)

    # ── Склейка разорванных таблиц ───────────────────────────────────
    raw_md = merge_broken_tables(raw_md)

    raw_path = output_dir / f"{stem}_docling_raw.md"
    raw_path.write_text(raw_md, encoding="utf-8")

    fixed_md = fix_subscripts(raw_md)
    fixed_path = output_dir / f"{stem}_fixed.md"
    save_intermediate(str(fixed_path), fixed_md, "Fixed subscripts")

    if len(fixed_md) > 25000:
        batches = chunk_text(fixed_md, max_chars=20000)
        chunked = []
        for i, batch in enumerate(batches):
            log.info(f"  Батч [{i+1}/{len(batches)}]")
            cleaned = call_claude(POSTPROCESS_PROMPT, batch, f"Claude батч {i+1}")
            chunked.append(cleaned)
        cleaned_md = "\n\n---\n\n".join(chunked)
    else:
        cleaned_md = call_claude(POSTPROCESS_PROMPT, fixed_md, "Claude постобработка")

    post_path = output_dir / f"{stem}_postprocessed.md"
    save_intermediate(str(post_path), cleaned_md, "Postprocessed")

    if needs_cleanup(cleaned_md):
        if len(cleaned_md) > 25000:
            batches = chunk_text(cleaned_md, max_chars=20000)
            chunked = []
            for i, batch in enumerate(batches):
                log.info(f"  Финальный батч [{i+1}/{len(batches)}]")
                cleaned = call_claude(FINAL_CLEANUP_PROMPT, batch, f"Финальная чистка батч {i+1}")
                chunked.append(cleaned)
            final_md = "\n\n---\n\n".join(chunked)
        else:
            final_md = call_claude(FINAL_CLEANUP_PROMPT, cleaned_md, "Финальная чистка")
        if len(final_md) < len(cleaned_md) * 0.5:
            final_md = cleaned_md
    else:
        final_md = cleaned_md

    final_path = output_dir / f"{stem}.md"
    final_path.write_text(final_md, encoding="utf-8")
    log.info(f"✅ Final: {final_path} ({len(final_md)} символов)")

    # Статистика
    img_dir = output_dir / "images"
    if img_dir.exists():
        png_count = len(list(img_dir.glob("*.png")))
        formula_count = sum(1 for img in images if img["type"] == "latex")
        image_count = sum(1 for img in images if img["type"] == "image")
        log.info(f"  Изображения: {png_count} PNG, {image_count} изображений, {formula_count} формул")

    stats = get_token_stats()
    log.info(f"  Фактический расход: in={stats['prompt_tokens']} out={stats['completion_tokens']} vision={stats['vision_calls']}×")

    _print_stats(result)

    # Cleanup промежуточных файлов
    _cleanup_intermediate(output_dir, stem)

    return str(final_path)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Вспомогательные
# ═══════════════════════════════════════════════════════════════════════════

def _print_stats(result):
    doc = result.document
    pages = doc.pages if hasattr(doc, "pages") else {}
    table_count = 0
    for item in doc.iterate_items():
        if isinstance(item, tuple):
            _, obj = item
        else:
            obj = item
        if hasattr(obj, 'label') and obj.label and "table" in str(obj.label).lower():
            table_count += 1
    log.info(f"Страниц: {len(pages)}, таблиц: ~{table_count}")


def collect_pdf_files(path: Path) -> list[Path]:
    """Собрать все PDF-файлы из директории (рекурсивно)."""
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.pdf"))


# ═══════════════════════════════════════════════════════════════════════════
# 5. CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PDF → Markdown конвертер (Docling + Claude)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Примеры:
  python3 pdf_to_md.py -i document.pdf
  python3 pdf_to_md.py -i document.pdf --mode auto
        """,
    )
    parser.add_argument("-i", "--input", required=True, help="Входной PDF файл")
    parser.add_argument("-o", "--output", default=None, help="Выходная папка")
    parser.add_argument("--mode", choices=["auto", "manual"], default=None,
                        help="auto — без подтверждения, manual — с оценкой токенов")
    parser.add_argument("--config", default=None, help="Путь к config.yaml")
    parser.add_argument("--debug", action="store_true", help="Подробное логирование")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Загружаем конфиг
    cfg = {}
    if args.config:
        cfg = load_config(args.config)

    # Применяем настройки из конфига к модулю utils
    if cfg:
        # API-ключи
        for key_name, cfg_key in [("PROVOD_API_KEY", "vision_primary_api_key"),
                                   ("OPENAI_API_KEY", "vision_fallback_api_key")]:
            val = cfg.get(cfg_key)
            if val:
                os.environ[key_name] = val
                # Обновляем в уже импортированном модуле utils
                import utils as _u
                _u.PROVOD_API_KEY = os.environ.get("PROVOD_API_KEY", "")

        # Vision models
        vision_models = []
        for role in ("vision_primary_model", "vision_fallback_model"):
            model = cfg.get(role)
            if model:
                vision_models.append(model)
        if vision_models:
            import utils as _u
            _u.VISION_MODELS = vision_models
            log.info(f"  Vision models: {vision_models}")

        # Postprocess models
        post_models = []
        for role in ("post_primary_model", "post_fallback_model"):
            model = cfg.get(role)
            if model:
                post_models.append(model)
        if post_models:
            import utils as _u
            _u.TEXT_MODELS = post_models
            log.info(f"  Post models: {post_models}")

        # Base URL (из vision или postprocess конфига)
        base_url = cfg.get("vision_primary_base_url") or cfg.get("post_primary_base_url")
        if base_url:
            import utils as _u
            if not base_url.endswith("/chat/completions"):
                base_url = base_url.rstrip("/") + "/chat/completions"
            _u.PROVOD_URL = base_url
            log.info(f"  API URL: {base_url}")

    # Определяем mode (приоритет: CLI > config)
    mode = args.mode or cfg.get("mode", "auto")
    args.yes = (mode == "auto")

    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"Не найдено: {input_path}")
        sys.exit(1)

    # ── Режим одного файла ────────────────────────────────────────────
    output_dir = Path(args.output) if args.output else \
        input_path.parent / "Markdown" / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"{'='*60}")
    log.info(f"PDF → Markdown конвертер")
    log.info(f"Режим: {mode}")
    log.info(f"Вход:  {input_path}")
    log.info(f"Выход: {output_dir}/")
    log.info(f"{'='*60}")

    out = process_with_confirm(str(input_path), output_dir, args)

    if out:
        log.info(f"\n✅ Готово: {out}")
    else:
        log.info(f"\nОбработка отменена")


if __name__ == "__main__":
    main()
