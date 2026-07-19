# Yle News JSON to Markdown Converter

This repository includes a module (`sulku.converter`) that processes Yle news JSON archives and converts each article into a structured Markdown file.

## Features

- **YAML Front Matter**: Every generated Markdown file starts with a YAML block containing article metadata (ID, title, URL, date published, date modified, language, authors, and subjects/tags).
- **Directory Mirroring**: Mirrors the directory layout of the JSON archive (e.g. `2021/01/0000/*.md`) in the destination folder to prevent flattening tens of thousands of files into one folder.
- **Configurable Block Types**: You can configure which Yle news block types are included or excluded from the generated files. Unwanted types (like video/audio/ads/embeds) can be filtered out easily.
- **AI-Ready Defaults**: By default, the converter only translates block types suitable for AI text training (`heading`, `text`, `quote`, `bullet-list`, `numbered-list`, `table`, `interactive-table`, `aside`, `feature`). This filters out multimedia placeholders and social media widget noise.
- **Internationalization (i18n)**: Uses standard Python `gettext` compatible classes to automatically translate generated section names and labels (e.g., "Gallery" -> "Galleria"/"Galleri", "Audio" -> "Äänite"/"Ljudspår", "Caption" -> "Kuvateksti"/"Bildtext") to match the specific language of each article (supporting Finnish, Swedish, and English).
- **Multiprocessing**: Automatically processes the JSON batch files in parallel across available CPU cores for high-speed conversion.

## Code Structure

- [converter.py](file:///app/src/sulku/converter.py): Main converter module containing logic for block mapping, YAML formatting, parallel extraction, and the CLI.
- [__init__.py](file:///app/src/sulku/__init__.py): Exposes package conversion entrypoints.

## API Usage

You can import the module functions within Python:

```python
from pathlib import Path
from sulku import convert_dataset

# Convert entire dataset using the default AI-ready text filter
convert_dataset(
    src_dir=Path("data/ylenews-fi-2021-src"),
    dest_dir=Path("output_markdown")
)

# Convert entire dataset including ALL Yle block types
convert_dataset(
    src_dir=Path("data/ylenews-fi-2021-src"),
    dest_dir=Path("output_markdown"),
    allowed_types={"*"}
)
```

## CLI Usage

The module can be invoked directly from the command line:

### Basic Conversion (Default AI training filter applied)
```bash
uv run python -m sulku.converter --src data/ylenews-fi-2021-src --dest output_markdown
```

### Full Conversion (Include all block types)
```bash
uv run python -m sulku.converter --src data/ylenews-fi-2021-src --dest output_markdown --include-types all
```

### Excluding Specific Block Types (Bypasses defaults, excludes specified ones)
```bash
uv run python -m sulku.converter \
  --src data/ylenews-fi-2021-src \
  --dest output_markdown \
  --exclude-types video,audio,some-posting,survey
```

### Exclusively Including Specific Block Types
```bash
uv run python -m sulku.converter \
  --src data/ylenews-fi-2021-src \
  --dest output_markdown \
  --include-types heading,text,quote,bullet-list,numbered-list,table,image
```
