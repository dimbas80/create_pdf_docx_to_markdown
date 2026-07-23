# Create_Markdown

Конвертация PDF и DOCX в чистый Markdown через Docling + Claude + Vision API.

## Возможности

- **PDF → Markdown**: Docling (layout detection) + Vision API (формулы/схемы) + Claude (постобработка)
- **DOCX → Markdown**: Docling SimplePipeline + Vision API + Claude
- **Склейка таблиц**: автоматическое объединение таблиц, разорванных между страницами
- **Чанкование Claude**: большие документы разбиваются на части по 20K символов
- **Распознавание формул**: через Gemini/Claude Vision API
- **Поддержка нормативной документации**: ПУЭ, ГОСТ, СП

## Установка

```bash
git clone https://github.com/dimbas80/create_pdf_docx_to_markdown.git
cd Create_Markdown
pip install -r requirements.txt  # или см. зависимости в CHANGELOG.md
```

## Использование

```bash
# PDF
python3 pdf_to_md.py -i document.pdf -o output/ --mode auto

# DOCX
python3 docx_to_md.py -i document.docx -o output/ --mode auto
```

## Версия

**0.11b** — подробнее в [CHANGELOG.md](CHANGELOG.md)
