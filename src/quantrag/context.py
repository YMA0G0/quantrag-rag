"""Context-header helpers shared by ingestion and retrieval.

The header ``"{title} — {section}"`` is prepended to chunk text at index
time so that both the vector and the stored text carry citation context.

This module is intentionally free of any ingestion or runtime dependencies
(std-lib only) so that ``quantrag.retriever`` — and by extension the MCP
server and FastAPI app — never transitively import the heavy ingestion
package (pypdf, trafilatura, pdfplumber, …).
"""

from __future__ import annotations


def add_context_header(text: str, title: str, section: str) -> str:
    """Prepend the citation context header to *text* (body only).

    Returns ``"{header}\\n\\n{text}"`` where *header* is ``title`` alone when
    *section* is empty, otherwise ``"{title} — {section}"``.
    """
    header = title.strip()
    if section.strip():
        header = f"{header} — {section.strip()}"
    return f"{header}\n\n{text.strip()}"
