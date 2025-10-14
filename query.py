from duckduckgo_search import DDGS
import json
import re
import html

def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_stackoverflow_duckduckgo(query: str, max_results=5):
    """
    在 DuckDuckGo 上模糊搜索 Stack Overflow 内容
    例如：site:stackoverflow.com firefox 0patch windows 10
    """
    search_query = f"site:stackoverflow.com {query}"
    results = []

    with DDGS() as ddgs:
        for r in ddgs.text(search_query, max_results=max_results):
            title = clean_text(r.get("title", ""))
            body = clean_text(r.get("body", ""))[:400]
            url = r.get("href", "")
            results.append({
                "title": title,
                "body": body,
                "url": url
            })
    return results


if __name__ == "__main__":
    query_item = {
        "id": "1539264",
        "query": "firefox compatibility with 0patch after windows 10 support ends"
    }

    results = fetch_stackoverflow_duckduckgo(query_item["query"], max_results=5)

    print(json.dumps({
        "id": query_item["id"],
        "query": query_item["query"],
        "results": results
    }, indent=2, ensure_ascii=False))
