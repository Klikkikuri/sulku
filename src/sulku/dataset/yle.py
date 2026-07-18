"""
Yle News JSON Archive to Markdown Converter
==========================================

This module parses large JSON files from the Yle news archive dataset and generates
individual Markdown files for each article. It mirrors the directory structure of
the input dataset in the destination directory to avoid flattening thousands of files
into a single folder.

The converter translates Yle-specific content blocks (such as headings, text, quotes,
lists, tables, galleries, embeds, and asides) into standard Markdown formatting
and includes a YAML front matter containing key metadata for each article.

It utilizes process-level parallelism to accelerate batch processing of large archives.
"""

import json
import logging
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import gettext

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default block types suitable for AI training (excluding audio, video, dynamic embeds, etc.)
DEFAULT_ALLOWED_TYPES = {
    "heading",
    "text",
    "quote",
    "bullet-list",
    "numbered-list",
    "table",
    "interactive-table",
    "aside",
    "feature"
}

# Translation catalogs for Yle news languages (Finnish, Swedish, English)
TRANSLATIONS = {
    "fi": {
        "Gallery": "Galleria",
        "Feature Page": "Erikoissivu",
        "Video": "Video",
        "Audio": "Äänite",
        "Survey": "Kysely",
        "External Content": "Ulkoinen sisältö",
        "Source": "Lähde",
        "Infogram": "Infograafi",
        "Exam": "Tentti",
        "Livefeed": "Live-seuranta",
        "Interactive Table": "Interaktiivinen taulukko",
        "Flockler Stream": "Flockler-virta",
        "post": "julkaisu",
        "Post": "Julkaisu",
        "social media": "sosiaalinen media",
        "twitter": "Twitter",
        "instagram": "Instagram",
        "facebook": "Facebook",
        "youtube": "YouTube",
        "Image": "Kuva",
        "Caption": "Kuvateksti",
        "Link": "Linkki"
    },
    "sv": {
        "Gallery": "Galleri",
        "Feature Page": "Erikoissida",
        "Video": "Video",
        "Audio": "Ljudspår",
        "Survey": "Enkät",
        "External Content": "Externt innehåll",
        "Source": "Källa",
        "Infogram": "Infograf",
        "Exam": "Prov",
        "Livefeed": "Direktrapportering",
        "Interactive Table": "Interaktiv tabell",
        "Flockler Stream": "Flockler-flöde",
        "post": "inlägg",
        "Post": "Inlägg",
        "social media": "sociala medier",
        "twitter": "Twitter",
        "instagram": "Instagram",
        "facebook": "Facebook",
        "youtube": "YouTube",
        "Image": "Bild",
        "Caption": "Bildtext",
        "Link": "Länk"
    }
}


class DictTranslations(gettext.NullTranslations):
    """
    In-memory catalog translations class compatible with standard Python gettext.
    """
    def __init__(self, catalog: dict):
        super().__init__()
        self._catalog = catalog

    def gettext(self, message: str) -> str:
        """
        Translate a message using the dictionary catalog.

        :param message: The message string to translate.
        :type message: str
        :return: Translated message string.
        :rtype: str
        """
        return self._catalog.get(message, message)


def get_translator(lang: str) -> gettext.NullTranslations:
    """
    Get a gettext NullTranslations compatible instance matching the language catalog.

    :param lang: Language code of the Yle news article (e.g. 'fi', 'sv', 'en').
    :type lang: str
    :return: An active translation instance catalog matching the language.
    :rtype: gettext.NullTranslations
    """
    if not lang:
        return gettext.NullTranslations()
    catalog = TRANSLATIONS.get(lang.lower())
    if catalog:
        return DictTranslations(catalog)
    return gettext.NullTranslations()


def block_to_markdown(
    block: dict,
    allowed_types: set[str] = None,
    exclude_types: set[str] = None,
    _ = None
) -> str:
    """
    Convert a content block dictionary to a markdown string.

    :param block: Dictionary representing a content block from the Yle article.
    :type block: dict
    :param allowed_types: Optional set of block types to exclusively allow. Wildcard '*' allows all.
    :type allowed_types: set[str], optional
    :param exclude_types: Optional set of block types to exclude.
    :type exclude_types: set[str], optional
    :param _: Optional translation function.
    :return: Markdown formatted string of the block.
    :rtype: str
    """
    if not block or not isinstance(block, dict):
        return ""
    b_type = block.get("type")
    if not b_type:
        return ""

    # Check configuration limits
    if allowed_types is not None and b_type not in allowed_types:
        if "*" not in allowed_types:
            return ""
    if exclude_types is not None and b_type in exclude_types:
        return ""

    if _ is None:
        def _(x):
            return x

    if b_type == "heading":
        level = block.get("level", 1)
        text = block.get("text", "").strip()
        if not text:
            return ""
        hashes = "#" * max(1, min(6, level + 1))
        return f"{hashes} {text}"

    elif b_type == "text":
        return block.get("text", "").strip()

    elif b_type == "quote":
        text = block.get("text", "").strip()
        if not text:
            return ""
        return f"> {text}"

    elif b_type in ("bullet-list", "numbered-list"):
        items = block.get("items", [])
        if not items:
            return ""
        res = []
        for i, item in enumerate(items):
            item_text = item.strip() if isinstance(item, str) else str(item).strip()
            if b_type == "bullet-list":
                res.append(f"- {item_text}")
            else:
                res.append(f"{i+1}. {item_text}")
        return "\n".join(res)

    elif b_type in ("table", "interactive-table") and "rows" in block:
        rows = block.get("rows", [])
        if not rows:
            return ""
        max_cols = 0
        for r in rows:
            cells = r.get("cells", []) if isinstance(r, dict) else []
            if len(cells) > max_cols:
                max_cols = len(cells)
        if max_cols == 0:
            return ""
        
        md_lines = []
        def get_row_cells(row_dict):
            cells = row_dict.get("cells", []) if isinstance(row_dict, dict) else []
            cell_texts = []
            for j in range(max_cols):
                if j < len(cells):
                    cell_val = cells[j].get("text", "") if isinstance(cells[j], dict) else str(cells[j])
                    cell_texts.append(cell_val.replace("\n", " ").strip())
                else:
                    cell_texts.append("")
            return cell_texts

        header_cells = get_row_cells(rows[0])
        md_lines.append("| " + " | ".join(header_cells) + " |")
        md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
        for r in rows[1:]:
            md_lines.append("| " + " | ".join(get_row_cells(r)) + " |")
        return "\n".join(md_lines)

    elif b_type == "image":
        alt = block.get("alt") or block.get("caption") or _("Image")
        img_id = block.get("id") or ""
        fmt = block.get("recommendedFormat") or "jpg"
        caption = block.get("caption") or ""
        source = block.get("source") or ""
        
        img_str = f"![{alt}]({img_id}.{fmt})"
        if caption:
            caption_lbl = _("Caption")
            if source:
                img_str += f"\n\n*{caption_lbl}: {caption} ({source})*"
            else:
                img_str += f"\n\n*{caption_lbl}: {caption}*"
        elif source:
            source_lbl = _("Source")
            img_str += f"\n\n*{source_lbl}: {source}*"
        return img_str

    elif b_type == "gallery":
        images = block.get("images", [])
        res = [f"### {_('Gallery')}"]
        for img in images:
            img_md = block_to_markdown(img, allowed_types, exclude_types, _)
            if img_md:
                res.append(img_md)
        return "\n\n".join(res)

    elif b_type == "aside":
        content = block.get("content", [])
        if not content:
            return ""
        rendered_blocks = [block_to_markdown(b, allowed_types, exclude_types, _) for b in content if b]
        combined = "\n\n".join(b for b in rendered_blocks if b)
        lines = combined.split("\n")
        return "\n".join(f"> {line}" for line in lines)

    elif b_type == "feature":
        pages = block.get("pages", [])
        if not pages:
            return ""
        res = []
        for i, page in enumerate(pages):
            page_content = page.get("content", []) if isinstance(page, dict) else []
            if page_content:
                rendered_blocks = [block_to_markdown(b, allowed_types, exclude_types, _) for b in page_content if b]
                combined = "\n\n".join(b for b in rendered_blocks if b)
                res.append(f"### {_('Feature Page')} {i+1}\n\n{combined}")
        return "\n\n".join(res)

    elif b_type == "links":
        links = block.get("links", [])
        if not links:
            return ""
        res = []
        for lnk in links:
            if not isinstance(lnk, dict):
                continue
            title = lnk.get("title") or _("Link")
            url = lnk.get("external", {}).get("url")
            if not url and lnk.get("internal", {}).get("id"):
                url = f"#{lnk['internal']['id']}"
            if url:
                res.append(f"- [{title}]({url})")
            else:
                res.append(f"- {title}")
        return "\n".join(res)

    elif b_type == "video":
        v_id = block.get("id", "")
        alt = block.get("alt") or ""
        img = block.get("image")
        res = f"*🎥 {_('Video')} {v_id}*"
        if alt:
            res += f": {alt}"
        if img and isinstance(img, dict):
            img_md = block_to_markdown(img, allowed_types, exclude_types, _)
            if img_md:
                res += f"\n\n{img_md}"
        return res

    elif b_type == "audio":
        a_id = block.get("id", "")
        caption = block.get("caption") or ""
        source = block.get("source") or ""
        res = f"*🔊 {_('Audio')} {a_id}*"
        if caption:
            res += f": {caption}"
        if source:
            res += f" ({source})"
        return res

    elif b_type == "some-posting":
        posting_type = block.get("postingType") or "social media"
        desc = block.get("description") or _("Post")
        url = block.get("url")
        p_type_translated = _(posting_type).capitalize()
        p_word_translated = _("post")
        if url:
            return f"*{p_type_translated} {p_word_translated}*: [{desc}]({url})"
        return f"*{p_type_translated} {p_word_translated}*: {desc}"

    elif b_type == "survey":
        s_id = block.get("surveyId") or ""
        alt = block.get("alt") or ""
        survey_lbl = _("Survey")
        return f"*📊 {survey_lbl} {s_id}*: {alt}" if alt else f"*📊 {survey_lbl} {s_id}*"

    elif b_type == "external-content":
        html = block.get("html") or ""
        ext_lbl = _("External Content")
        if html:
            return f"*🔗 {ext_lbl}*: [{_('Source')}]({html})"
        return f"*🔗 {ext_lbl}*"

    elif b_type == "infogram":
        info_id = block.get("infogramId") or ""
        alt = block.get("alt") or ""
        info_lbl = _("Infogram")
        return f"*📊 {info_lbl} {info_id}*: {alt}" if alt else f"*📊 {info_lbl} {info_id}*"

    elif b_type == "tehtava-exam":
        exam_id = block.get("id") or ""
        return f"*📝 {_('Exam')} {exam_id}*"

    elif b_type == "livefeed":
        feed_id = block.get("livefeedId") or ""
        return f"*📢 {_('Livefeed')} {feed_id}*"

    elif b_type == "interactive-table":
        table_id = block.get("tableId") or ""
        alt = block.get("alt") or ""
        tbl_lbl = _("Interactive Table")
        return f"*📊 {tbl_lbl} {table_id}*: {alt}" if alt else f"*📊 {tbl_lbl} {table_id}*"

    elif b_type == "flockler":
        stream_id = block.get("streamId") or ""
        return f"*📱 {_('Flockler Stream')} {stream_id}*"

    else:
        text = block.get("text")
        if text:
            return str(text).strip()
        return f"*[{b_type}]*"


def format_front_matter(article: dict) -> str:
    """
    Format the article metadata into a YAML front matter string.

    :param article: Dictionary representing the article.
    :type article: dict
    :return: YAML front matter string enclosed in triple dashes.
    :rtype: str
    """
    lines = ["---"]
    
    # ID
    art_id = article.get("id")
    if art_id:
        lines.append(f"id: {art_id}")
        
    # Title
    headline = article.get("headline")
    title = ""
    if isinstance(headline, dict):
        title = headline.get("full") or ""
    elif isinstance(headline, str):
        title = headline
    if title:
        safe_title = title.replace('"', '\\"')
        lines.append(f'title: "{safe_title}"')
        
    # URL
    url_dict = article.get("url")
    url = ""
    if isinstance(url_dict, dict):
        url = url_dict.get("full") or url_dict.get("short") or ""
    elif isinstance(url_dict, str):
        url = url_dict
    if url:
        lines.append(f"url: {url}")
        
    # Dates
    date_pub = article.get("datePublished")
    if date_pub:
        lines.append(f"datePublished: {date_pub}")
        
    date_mod = article.get("dateContentModified") or article.get("dateJsonModified")
    if date_mod:
        lines.append(f"dateModified: {date_mod}")
        
    # Language
    lang = article.get("language")
    if lang:
        lines.append(f"language: {lang}")
        
    # Authors
    authors = article.get("authors")
    if authors and isinstance(authors, list):
        lines.append("authors:")
        for auth in authors:
            if isinstance(auth, dict):
                name = auth.get("name")
                org = auth.get("organization")
                if name:
                    if org:
                        lines.append(f"  - name: {name}")
                        lines.append(f"    organization: {org}")
                    else:
                        lines.append(f"  - name: {name}")
            elif isinstance(auth, str):
                lines.append(f"  - name: {auth}")
                
    # Subjects/Tags
    subjects = article.get("subjects")
    if subjects and isinstance(subjects, list):
        lines.append("subjects:")
        for sub in subjects:
            if isinstance(sub, dict):
                title_dict = sub.get("title")
                title_val = ""
                if isinstance(title_dict, dict):
                    title_val = title_dict.get("fi") or title_dict.get("en") or title_dict.get("sv") or ""
                elif isinstance(title_dict, str):
                    title_val = title_dict
                if title_val:
                    safe_val = title_val.replace('"', '\\"')
                    lines.append(f'  - "{safe_val}"')
            elif isinstance(sub, str):
                safe_val = sub.replace('"', '\\"')
                lines.append(f'  - "{safe_val}"')
                
    lines.append("---")
    return "\n".join(lines)


def article_to_markdown(
    article: dict,
    allowed_types: set[str] = None,
    exclude_types: set[str] = None
) -> str:
    """
    Render a complete Yle article dictionary to its Markdown representation.

    :param article: The dictionary representing the article.
    :type article: dict
    :param allowed_types: Optional set of block types to exclusively allow. Wildcard '*' allows all.
    :type allowed_types: set[str], optional
    :param exclude_types: Optional set of block types to exclude.
    :type exclude_types: set[str], optional
    :return: Complete Markdown string including front matter and body content.
    :rtype: str
    """
    lang = article.get("language") or "en"
    translator = get_translator(lang)
    _ = translator.gettext

    front_matter = format_front_matter(article)
    
    body_parts = []
    
    # Check if the title is already present as a heading at the start of body content
    content_blocks = article.get("content", [])
    has_title_in_content = False
    
    headline_dict = article.get("headline")
    title = ""
    if isinstance(headline_dict, dict):
        title = headline_dict.get("full") or ""
    elif isinstance(headline_dict, str):
        title = headline_dict
        
    for block in content_blocks[:2]:
        if block.get("type") == "heading" and block.get("text", "").strip() == title.strip():
            has_title_in_content = True
            break
            
    if not has_title_in_content and title:
        body_parts.append(f"# {title}")
        
    # Convert all content blocks in sequence
    for block in content_blocks:
        md = block_to_markdown(block, allowed_types, exclude_types, _)
        if md:
            body_parts.append(md)
            
    body_content = "\n\n".join(body_parts)
    
    return f"{front_matter}\n\n{body_content}\n"


def process_single_json(
    json_path: Path,
    src_dir: Path,
    dest_dir: Path,
    allowed_types: set[str] = None,
    exclude_types: set[str] = None
) -> int:
    """
    Process a single Yle news JSON file and write markdown files for each article.

    :param json_path: Path to the Yle news JSON file.
    :type json_path: Path
    :param src_dir: Source root directory to compute relative paths.
    :type src_dir: Path
    :param dest_dir: Destination root directory where markdown files will be created.
    :type dest_dir: Path
    :param allowed_types: Optional set of block types to exclusively allow.
    :type allowed_types: set[str], optional
    :param exclude_types: Optional set of block types to exclude.
    :type exclude_types: set[str], optional
    :return: The number of articles successfully converted and written.
    :rtype: int
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read/parse JSON file {json_path}: {e}")
        return 0

    articles = data.get("data", [])
    if not articles:
        return 0

    # Calculate target directory mirroring the source structure
    try:
        relative_path = json_path.relative_to(src_dir)
        relative_parent = relative_path.parent
        json_stem = json_path.stem
        target_dir = dest_dir / relative_parent / json_stem
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create target directory for {json_path}: {e}")
        return 0

    count = 0
    for article in articles:
        if not isinstance(article, dict):
            continue
        art_id = article.get("id")
        if not art_id:
            continue
        
        safe_id = "".join(c for c in str(art_id) if c.isalnum() or c in ("-", "_"))
        if not safe_id:
            continue
            
        dest_file = target_dir / f"{safe_id}.md"
        try:
            md_content = article_to_markdown(article, allowed_types, exclude_types)
            with open(dest_file, "w", encoding="utf-8") as out_f:
                out_f.write(md_content)
            count += 1
        except Exception as e:
            logger.error(f"Failed to write article {art_id} from {json_path} to {dest_file}: {e}")

    return count


def convert_dataset(
    src_dir: Path,
    dest_dir: Path,
    max_workers: int = None,
    allowed_types: set[str] = None,
    exclude_types: set[str] = None
) -> int:
    """
    Recursively find all JSON files under src_dir and convert them to Markdown.

    Processes JSON files in parallel using a process pool to maximize CPU usage.
    Supports src_dir being a directory or a single JSON file.

    :param src_dir: Path to the source directory containing Yle news JSON files or a single JSON file.
    :type src_dir: Path
    :param dest_dir: Path to the destination directory for markdown output files.
    :type dest_dir: Path
    :param max_workers: Optional maximum number of parallel workers.
    :type max_workers: int, optional
    :param allowed_types: Optional set of block types to exclusively allow. Wildcard '*' allows all.
    :type allowed_types: set[str], optional
    :param exclude_types: Optional set of block types to exclude.
    :type exclude_types: set[str], optional
    :return: The total number of articles processed.
    :rtype: int
    """
    src_dir = Path(src_dir).resolve()
    dest_dir = Path(dest_dir).resolve()

    if src_dir.is_file():
        json_files = [src_dir]
        src_root = src_dir.parent
    else:
        json_files = sorted(list(src_dir.glob("**/*.json")))
        src_root = src_dir

    if not json_files:
        logger.warning(f"No JSON files found at {src_dir}")
        return 0

    logger.info(f"Found {len(json_files)} JSON file(s) to process.")

    # Default to text-friendly blocks suitable for AI training if no configuration is set
    if allowed_types is None and exclude_types is None:
        allowed_types = DEFAULT_ALLOWED_TYPES

    total_articles = 0
    completed_files = 0

    # If max_workers is not specified, default to CPU count
    if not max_workers:
        max_workers = os.cpu_count() or 4

    logger.info(f"Starting conversion using up to {max_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_json,
                jf,
                src_root,
                dest_dir,
                allowed_types,
                exclude_types
            ): jf
            for jf in json_files
        }

        for future in as_completed(futures):
            jf = futures[future]
            try:
                articles_count = future.result()
                total_articles += articles_count
                completed_files += 1
                logger.info(
                    f"[{completed_files}/{len(json_files)}] Processed {jf.name}: "
                    f"wrote {articles_count} articles. (Total so far: {total_articles})"
                )
            except Exception as e:
                logger.error(f"Error processing JSON file {jf}: {e}")

    logger.info(f"Conversion finished. Total articles written: {total_articles}")
    return total_articles


def main() -> None:
    """
    Command line interface entry point.
    """
    parser = argparse.ArgumentParser(
        description="Convert Yle News JSON Archive dataset to Markdown files."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to the source data folder containing Yle news JSON files."
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="Path to the destination folder for generated markdown files."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers. Defaults to CPU count."
    )
    parser.add_argument(
        "--exclude-types",
        type=str,
        default=None,
        help="Comma-separated list of Yle block types to exclude (e.g. video,audio,some-posting)"
    )
    parser.add_argument(
        "--include-types",
        type=str,
        default=None,
        help="Comma-separated list of Yle block types to exclusively include. Use 'all' or '*' to include everything."
    )

    args = parser.parse_args()
    
    src_path = Path(args.src)
    dest_path = Path(args.dest)
    
    if not src_path.exists():
        logger.error(f"Source path does not exist: {src_path}")
        sys.exit(1)
        
    exclude_types = None
    if args.exclude_types:
        exclude_types = {t.strip() for t in args.exclude_types.split(",") if t.strip()}

    allowed_types = None
    if args.include_types:
        val = args.include_types.strip()
        if val in ("all", "*"):
            allowed_types = {"*"}
        else:
            allowed_types = {t.strip() for t in args.include_types.split(",") if t.strip()}
        
    convert_dataset(
        src_path,
        dest_path,
        max_workers=args.workers,
        allowed_types=allowed_types,
        exclude_types=exclude_types
    )


if __name__ == "__main__":
    main()
