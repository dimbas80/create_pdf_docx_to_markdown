# Create_Markdown

Пакет скриптов для конвертации нормативной документации (ПУЭ, ГОСТ, СП) из PDF и DOCX в чистый Markdown через Docling + Claude + Vision API.

Особенности:
- Распознавание иерархии разделов, таблиц, списков
- Автоматическое распознавание формул через Vision API (Gemini/Claude)
- Склейка таблиц, разорванных между страницами
- Чанкование больших документов для обработки Claude
- Оценка расхода токенов перед запуском

---

## Состав пакета

### Create_Markdown/pdf_to_md.py
**Конвертация PDF → Markdown.** Основной скрипт.

Процесс:
1. **Docling** — парсинг PDF (layout detection, OCR, таблицы) — бесплатно, локально
2. **Извлечение изображений** — поиск IMAGE-блоков через PyMuPDF
3. **Vision API** — классификация: формула (→ LaTeX) или схема (→ PNG)
4. **fix_subscripts** — regex-чистка LaTeX-индексов (греческие буквы, русские подстрочные)
5. **Склейка таблиц** — merge_broken_tables (объединение через разрывы страниц)
6. **Claude постобработка** — чанкование по 20K символов, каждый батч → чистый Markdown
7. **Финальная чистка** — только если есть маркеры нераспознанных формул

```bash
python3 pdf_to_md.py -i document.pdf -o output/ --mode auto
python3 pdf_to_md.py -i document.pdf --config my_config.yaml --debug
```

Опции:
| Параметр | Описание |
|---|---|
| `-i, --input` | Путь к входящему PDF (обязательно) |
| `-o, --output` | Папка для результата (по умолч. Markdown/имя_файла) |
| `--mode auto/manual` | auto — без подтверждения, manual — с оценкой токенов |
| `--config` | Путь к config.yaml (по умолч. config.yaml в папке скрипта) |
| `--debug` | Подробное логирование (DEBUG) |

---

### Create_Markdown/docx_to_md.py
**Конвертация DOCX → Markdown.**

Процесс:
1. **Docling** — парсинг DOCX (SimplePipeline) — бесплатно, локально
2. **Vision API** — классификация изображений и формул
3. **fix_subscripts** — чистка LaTeX
4. **Склейка таблиц** — merge_broken_tables
5. **Claude постобработка** — чанкование по 20K символов

```bash
python3 docx_to_md.py -i document.docx -o output/ --mode auto
```

Опции те же, что у pdf_to_md.py.

---

### Create_Markdown/utils.py
**Вспомогательные функции**, общие для всех скриптов:

| Функция | Назначение |
|---|---|
| `call_claude()` | Универсальный вызов Claude с fallback-цепочкой |
| `call_vision()` | Универсальный вызов Vision API (formula / classify) |
| `chunk_text()` | Разбивка текста на батчи по разделителю |
| `fix_subscripts()` | Regex-чистка LaTeX-индексов |
| `merge_broken_tables()` | Склейка разорванных таблиц |
| `save_intermediate()` | Сохранение с backup (.bak) |
| `needs_cleanup()` | Проверка на нераспознанные формулы |

Промпты:
- `POSTPROCESS_PROMPT` — инструкция для Claude: структура, таблицы, заголовки
- `FINAL_CLEANUP_PROMPT` — финальная проверка качества

---

### Create_Markdown/Create_Markdown.py
**Пакетный режим** — обрабатывает все PDF/DOCX из папки `input/` и сохраняет в `output/`.

```bash
python3 Create_Markdown.py
```

Настройки batch-режима — в config.yaml.

---

### index_search/index_chromaDB.py
**Индексация Markdown-документов в векторную базу ChromaDB** для RAG-поиска.

- Эмбеддинг через **SiliconFlow Qwen3-Embedding-8B** (4096 dim)
- Чанкование: по разделам (`##`) и таблицам, лимит 8000 токенов, с перекрытием 200 токенов
- Коллекция: `gost_docs_qwen`

```bash
export SILICONFLOW_API_KEY="sk-..."
python3 index_chromaDB.py --md-path document.md --dry-run
python3 index_chromaDB.py --md-path document.md --source-tag "Мой документ"
python3 index_chromaDB.py --md-path document.md --recreate  # пересоздать коллекцию
```

Опции:
| Параметр | Описание |
|---|---|
| `--md-path` | Путь к Markdown-файлу |
| `--source-tag` | Тег источника (по умолч. имя файла) |
| `--collection` | Имя коллекции ChromaDB (по умолч. gost_docs_qwen) |
| `--max-tokens` | Лимит чанка в токенах (по умолч. 8000) |
| `--overlap-tokens` | Перекрытие между чанками (по умолч. 200) |
| `--dry-run` | Только разбить на чанки, без эмбеддинга |
| `--recreate` | Удалить коллекцию перед индексацией |
| `--delete-old` | Удалить старые чанки с тем же source-tag |
| `--api-key` | API-ключ SiliconFlow (или SILICONFLOW_API_KEY) |

---

### index_search/search_chromaDB.py
**Поиск по векторной базе знаний.**

Двухэтапный поиск:
1. **ChromaDB** — косинусная близость через Qwen3-Embedding-8B (top-4)
2. **Реранкер** — Qwen3-Reranker-8B (top-100 → top-4), автоматически включается при слабой уверенности эмбеддинга

```bash
./search_chromaDB.py "допустимый ток кабеля 4х35 медь ПВХ"
./search_chromaDB.py "запрос" --json
./search_chromaDB.py "запрос" --force-rerank
./search_chromaDB.py "запрос" --no-rerank
./search_chromaDB.py --list-collections
./search_chromaDB.py --test  # проверка связи с API
```

---

## Конфигурация (config.yaml)

```yaml
vision:              # Vision API для распознавания формул
  primary:
    provider: provod # провайдер
    api_key: "..."   # API-ключ (⚠ не хранить в git!)
    base_url: https://api.provod.ai/v1
    model: google/gemini-3.5-flash
  fallback:          # резервная модель
    ...
postprocess:         # Claude для постобработки
  primary:
    provider: provod
    api_key: "..."
    base_url: https://api.provod.ai/v1
    model: google/gemini-3.5-flash
  fallback:
    ...
mode: manual         # auto / manual — режим подтверждения
batch:               # пакетный режим
  enabled: false
  input: /path/to/docs
  output: /path/to/output
```

**API-ключи** можно задать тремя способами (приоритет сверху):
1. Переменная окружения `export PROVOD_API_KEY="sk-..."`
2. Параметр `--config` с заполненным config.yaml
3. config.yaml по умолчанию в папке скрипта

---

## Установка

### Зависимости

```bash
pip install docling chromadb httpx openai pyyaml pillow tiktoken
```

| Библиотека | Для чего |
|---|---|
| `docling` | Парсинг PDF/DOCX (IBM) |
| `chromadb` | Векторная база для RAG |
| `httpx` | HTTP-клиент для API |
| `openai` | Эмбеддинги через SiliconFlow |
| `PyYAML` | Чтение config.yaml |
| `Pillow` | Обработка изображений |
| `tiktoken` | Оценка токенов (опционально) |

### Проверка

```bash
python3 pdf_to_md.py -i test.pdf -o /tmp/test/ --mode auto
```

---

## Структура репозитория

```
Electro/
├── .gitignore
├── CHANGELOG.md         # История версий
├── LICENSE              # GPL v3
├── README.md            # Этот файл
├── Create_Markdown/
│   ├── config.yaml      # Конфигурация (ключи удалены)
│   ├── pdf_to_md.py     # PDF → Markdown
│   ├── docx_to_md.py    # DOCX → Markdown
│   ├── Create_Markdown.py # Пакетный режим
│   ├── utils.py         # Общие функции
│   └── tmp/             # Промежуточные файлы (в gitignore)
└── index_search/        # RAG-индексация (отдельный проект)
    ├── index_chromaDB.py
    └── search_chromaDB.py
```

---

## Версия

**0.11b** — подробнее в [CHANGELOG.md](CHANGELOG.md).

Лицензия: **GNU General Public License v3.0** — см. [LICENSE](LICENSE).
