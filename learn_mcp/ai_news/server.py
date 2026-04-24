from mcp.server.fastmcp import FastMCP
import requests
import xml.etree.ElementTree as ET
import sys

mcp = FastMCP("ai_news")

RSS_FEEDS = {
    "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "The Verge AI": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "MIT Tech Review AI": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
    "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
    "Ars Technica AI": "https://arstechnica.com/ai/feed/",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ai-news-mcp/1.0)"}


@mcp.tool()
def get_hackernews_ai(limit: int = 15) -> str:
    """
    Fetch recent AI-related stories from HackerNews via the free Algolia API.
    :param limit (int): Max stories to return (default 15).
    """
    url = "https://hn.algolia.com/api/v1/search_by_date"
    params = {
        "query": "AI OR LLM OR GPT OR Claude OR Anthropic OR OpenAI",
        "tags": "story",
        "hitsPerPage": limit,
    }
    r = requests.get(url, params=params, timeout=15, headers=HEADERS)
    r.raise_for_status()
    hits = r.json().get("hits", [])
    lines = []
    for h in hits:
        title = (h.get("title") or "").strip()
        if not title:
            continue
        link = h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}"
        pts = h.get("points") or 0
        comments = h.get("num_comments") or 0
        lines.append(f"- [{pts}pts, {comments}c] {title}\n  {link}")
    return "\n".join(lines) if lines else "No HackerNews AI stories found."


@mcp.tool()
def get_arxiv_ai(limit: int = 10) -> str:
    """
    Fetch the latest cs.AI / cs.LG / cs.CL papers from arXiv (no key required).
    :param limit (int): Max papers to return (default 10).
    """
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": "cat:cs.AI OR cat:cs.LG OR cat:cs.CL",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": limit,
    }
    r = requests.get(url, params=params, timeout=20, headers=HEADERS)
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    lines = []
    for entry in root.findall("a:entry", ns):
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        title = " ".join(title.split())
        link = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        summary = " ".join(summary.split())[:220]
        lines.append(f"- {title}\n  {link}\n  {summary}...")
    return "\n".join(lines) if lines else "No arXiv results."


@mcp.tool()
def get_rss_ai(limit_per_feed: int = 4) -> str:
    """
    Fetch recent articles from a curated set of AI-focused RSS/Atom feeds.
    :param limit_per_feed (int): Max items per feed (default 4).
    """
    atom_ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for name, url in RSS_FEEDS.items():
        try:
            r = requests.get(url, timeout=15, headers=HEADERS)
            r.raise_for_status()
            root = ET.fromstring(r.content)
        except Exception as e:
            out.append(f"- [{name}] fetch failed: {type(e).__name__}")
            continue

        channel = root.find("channel")
        if channel is not None:
            items = channel.findall("item")[:limit_per_feed]
            for it in items:
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                if title:
                    out.append(f"- [{name}] {title}\n  {link}")
        else:
            entries = root.findall("a:entry", atom_ns)[:limit_per_feed]
            for e in entries:
                title = (e.findtext("a:title", default="", namespaces=atom_ns) or "").strip()
                link_el = e.find("a:link", atom_ns)
                link = link_el.get("href") if link_el is not None else ""
                if title:
                    out.append(f"- [{name}] {title}\n  {link}")
    return "\n".join(out) if out else "No RSS results."


if __name__ == "__main__":
    print("Starting ai_news MCP server...", file=sys.stderr)
    mcp.run(transport="stdio")
