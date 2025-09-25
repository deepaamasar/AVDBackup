from pathlib import Path
from typing import Optional
import re
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


APP_TITLE = "Docling PDF to HTML Service"
UPLOAD_DIR = Path(r"C:\Input\Inputs")
OUTPUT_DIR = Path(r"C:\Input\Inputs")


def build_converter() -> DocumentConverter:
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
            )
        }
    )


app = FastAPI(title=APP_TITLE)

# Allow local requests from browsers/tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

converter = build_converter()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/convert")
def convert_pdf(
    file: UploadFile = File(...),
    return_path: Optional[bool] = False,
    return_raw: Optional[bool] = False,
    wrap_code_fence: Optional[bool] = False,
    tables_mode: Optional[str] = "auto",  # no-op for HTML; kept for compatibility
):
    """
    Convert an uploaded PDF to HTML while maintaining document structure.

    - Returns the .html file as an attachment by default.
    - If return_path=true, returns a JSON with the saved path instead.
    - Single-shot conversion of the entire PDF (no per-page splitting).
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Save upload to disk
        upload_path = UPLOAD_DIR / file.filename
        with upload_path.open("wb") as f:
            f.write(file.file.read())

        # Single-shot conversion (all pages at once)
        conv_res = converter.convert(source=str(upload_path))
        doc = conv_res.document
        # Prefer native HTML export; fall back to Markdown->HTML rendering
        html_content = _export_document_to_html(doc)
        if not html_content:
            try:
                md_fallback = doc.export_to_markdown()
                html_content = _markdown_to_html(md_fallback)
            except Exception:
                html_content = "<html><body><p>Conversion succeeded, but HTML rendering failed.</p></body></html>"

        # Save HTML output next to outputs dir
        out_html = OUTPUT_DIR / (upload_path.stem + ".html")
        out_html.write_text(html_content, encoding="utf-8")

        # Log basic sizes to server console for quick inspection
        try:
            print(
                f"Converted {file.filename}: html chars={len(html_content)}; path={out_html}")
        except Exception:
            pass

        if return_path:
            return JSONResponse({"output": str(out_html)})

        # Optionally return raw HTML instead of sending a file
        if return_raw:
            # Return raw HTML; wrap_code_fence is ignored for HTML
            return HTMLResponse(content=html_content, media_type="text/html")

        # Return as file download
        return FileResponse(
            path=str(out_html),
            media_type="text/html",
            filename=out_html.name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")


if __name__ == "__main__":
    # Optional dev server runner: uvicorn docling_service:app --host 0.0.0.0 --port 8000
    import uvicorn
    host = os.environ.get("DOCSVC_HOST", "0.0.0.0")
    port = int(os.environ.get("DOCSVC_PORT", "8090"))
    uvicorn.run("docling_service:app", host=host, port=port, reload=False)


def _export_document_to_text(doc) -> str:
    """Export a Docling document to plain text.

    Tries known text exporters; falls back to converting Markdown to text.
    """
    # 1) Try direct text export if available
    for attr in ("export_to_text", "export_to_plaintext", "export_to_txt"):
        fn = getattr(doc, attr, None)
        if callable(fn):
            try:
                txt = fn()
                if isinstance(txt, str) and txt.strip():
                    return txt
            except Exception:
                pass

    # 2) Try HTML export and convert to text using BeautifulSoup for better fidelity
    for attr in ("export_to_html", "export_as_html", "to_html"):
        fn = getattr(doc, attr, None)
        if callable(fn):
            try:
                html = fn()
                if isinstance(html, str) and html:
                    try:
                        from bs4 import BeautifulSoup  # installed via docling deps
                        soup = BeautifulSoup(html, "html.parser")
                        # Get text with reasonable spacing
                        txt = soup.get_text(separator="\n")
                        if txt.strip():
                            return _normalize_whitespace(txt)
                    except Exception:
                        pass
            except Exception:
                pass

    # 3) Fallback: export to Markdown and conservatively strip formatting
    md = doc.export_to_markdown()
    return _markdown_to_text(md)


def _markdown_to_text(md: str) -> str:
    """A conservative Markdown->text converter that aims to keep content intact."""
    text = md
    # Code fences: drop backticks but keep inner content
    text = re.sub(r"```(.*?)```", lambda m: "\n" + m.group(1).strip() + "\n", text, flags=re.DOTALL)
    # Headers -> plain lines (remove leading # only)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Bold/italic/inline code markers
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Links: [text](url) -> text (url)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    # Images: ![alt](url) -> alt (url)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\1 (\2)", text)
    # Lists: keep bullets as hyphens for readability
    text = re.sub(r"^\s*[-*+]\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(\d+)\.\s+", r"\1. ", text, flags=re.MULTILINE)
    # Blockquotes: remove '>' but keep content
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    # Simple tables: convert pipes to tabs but keep row content
    text = re.sub(r"^\s*\|\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*\|\s*", "\t", text)
    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}$", "", text, flags=re.MULTILINE)
    # Normalize whitespace
    text = _normalize_whitespace(text)
    return text


def _normalize_whitespace(s: str) -> str:
    # Collapse 3+ newlines to 2, trim trailing spaces
    s = re.sub(r"[ \t]+$", "", s, flags=re.MULTILINE)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()



def _export_document_to_html(doc) -> str | None:
    for attr in ("export_to_html", "export_as_html", "to_html"):
        fn = getattr(doc, attr, None)
        if callable(fn):
            try:
                html = fn()
                if isinstance(html, str) and html.strip():
                    return html
            except Exception:
                pass
    return None


def _markdown_contains_table_pipes(md: str) -> bool:
    # Detect typical markdown table rows with pipes
    return bool(re.search(r"^\s*\|.*\|\s*$", md, flags=re.MULTILINE))


def _html_to_markdown_with_tables(html: str) -> str:
    """Convert HTML to Markdown with proper markdown tables and common tags.

    Uses BeautifulSoup to parse HTML and emits headings, paragraphs, lists,
    links, images, line breaks, and tables with pipe syntax.
    """
    try:
        from bs4 import BeautifulSoup
    except Exception:
        # Fallback: keep HTML fenced so content isn't lost
        return f"```html\n{html}\n```"

    soup = BeautifulSoup(html, "html.parser")

    def text_of(node) -> str:
        return " ".join(node.get_text(separator=" ", strip=True).split()) if node else ""

    def handle_table(table) -> str:
        rows = table.find_all("tr")
        if not rows:
            return ""
        # Build header from first row if th present; otherwise synthesize
        header_cells = rows[0].find_all(["th", "td"]) if rows else []
        headers = [text_of(th) for th in header_cells]
        if not table.find("th") and headers:
            headers = [f"Col {i+1}" for i in range(len(headers))]
        lines = []
        if headers:
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        data_rows = rows[1:] if len(rows) > 1 else []
        for r in data_rows:
            cells = [text_of(td) for td in r.find_all(["td", "th"])]
            if cells:
                lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def walk(node) -> str:
        out: list[str] = []
        for el in node.children:
            name = getattr(el, "name", None)
            if not name:
                txt = str(el)
                if txt.strip():
                    out.append(txt)
                continue
            if name in ("script", "style"):
                continue
            if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(name[1])
                out.append("#" * level + " " + text_of(el) + "\n\n")
            elif name == "p":
                out.append(text_of(el) + "\n\n")
            elif name == "br":
                out.append("\n")
            elif name in ("ul", "ol"):
                idx = 1
                for li in el.find_all("li", recursive=False):
                    prefix = (f"{idx}. " if name == "ol" else "- ")
                    out.append(prefix + text_of(li) + "\n")
                    idx += 1
                out.append("\n")
            elif name == "a":
                href = el.get("href", "")
                out.append(f"[{text_of(el)}]({href})")
            elif name == "img":
                alt = el.get("alt", "")
                src = el.get("src", "")
                out.append(f"![{alt}]({src})")
            elif name == "table":
                md_tbl = handle_table(el)
                if md_tbl:
                    out.append(md_tbl + "\n\n")
            else:
                out.append(walk(el))
        return "".join(out)

    content = walk(soup.body or soup)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def _markdown_to_basic_html(md: str) -> str:
    """Very simple Markdown -> HTML wrapper as a fallback.

    This is not a full markdown renderer; it wraps content in <pre> to preserve
    structure and replaces basic line breaks. Suitable only as a last resort.
    """
    safe = (
        md.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    # Convert triple backtick code fences to <pre><code>
    safe = re.sub(r"```(.*?)```", lambda m: f"<pre><code>{m.group(1)}</code></pre>", safe, flags=re.DOTALL)
    # Convert single newlines to <br>, preserve paragraphs roughly
    html_body = "\n".join(f"<p>{line}</p>" if line.strip() else "<br>" for line in safe.splitlines())
    return f"<html><body>{html_body}</body></html>"


def _markdown_to_html(md: str) -> str:
    """Convert Markdown to HTML using python-markdown if available; otherwise a basic wrapper.

    To improve fidelity (headings, lists, tables), install:
      pip install markdown
    """
    try:
        import markdown  # type: ignore
        # Enable common extensions, including tables
        html_body = markdown.markdown(md, extensions=["extra", "tables", "sane_lists", "toc"])
        return f"<html><body>{html_body}</body></html>"
    except Exception:
        return _markdown_to_basic_html(md)
