"""
Fetches content from a public URL -- either an HTML page (article,
e-paper edition, documentation page, blog post, public report page)
or a direct PDF link.

- HTML pages: uses `trafilatura`, which is purpose-built to strip nav
  bars, ads, cookie banners, and boilerplate, keeping just the main
  article/body content. This matters a lot for news/e-paper sites,
  which are otherwise full of junk that pollutes your chunks.
- PDF links (very common for e-papers, public reports, gov filings):
  downloaded in-memory and parsed the same way as an uploaded PDF.

Only use this against publicly accessible URLs you have the right to
access -- it respects normal HTTP semantics (status codes, redirects)
but does not bypass logins, paywalls, or robots.txt-disallowed content.
"""
import io
import requests
import trafilatura
from pypdf import PdfReader

USER_AGENT = "Mozilla/5.0 (compatible; KnowledgeAssistantBot/1.0; +local-demo)"
REQUEST_TIMEOUT = 20


def is_url(source: str) -> bool:
    return source.strip().lower().startswith(("http://", "https://"))


def fetch_url_text(url: str) -> str:
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()

    if "application/pdf" in content_type or url.lower().split("?")[0].endswith(".pdf"):
        reader = PdfReader(io.BytesIO(response.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif "text/html" in content_type or content_type == "":
        text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
    else:
        # Plain text, markdown, rst, etc. -- nothing to strip, use as-is
        text = response.text

    if not text or not text.strip():
        raise ValueError(
            f"Could not extract readable text from {url}. "
            f"It may be paywalled, JS-rendered, or blocking automated requests."
        )
    return text
