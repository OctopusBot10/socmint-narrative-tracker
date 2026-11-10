"""JSON and CSV exports of a completed run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR
from models import NarrativeCluster, Post


def build_payload(
    *,
    keyword: str,
    start: datetime,
    end: datetime,
    mode: str,
    platforms: list[str],
    posts: list[Post],
    clusters: list[NarrativeCluster],
    alerts: list[dict],
    repeats: list[dict],
) -> dict:
    cluster_of = {}
    for cluster in clusters:
        for post in cluster.posts:
            cluster_of[post.post_id] = cluster

    post_rows = []
    for post in sorted(posts, key=lambda item: item.created_at):
        row = post.to_dict()
        cluster = cluster_of.get(post.post_id)
        row["cluster_id"] = cluster.cluster_id if cluster else None
        row["cluster_label"] = cluster.label if cluster else ""
        row.pop("text_preview", None)
        post_rows.append(row)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": (
            "Narrative clusters and coordination alerts are heuristics over the "
            "collected sample. They do not establish that a claim is false, and "
            "they do not establish that accounts are operated by the same person."
        ),
        "query": {
            "keyword": keyword,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "mode": mode,
            "platforms": platforms,
        },
        "summary": {
            "posts": len(posts),
            "clusters": len(clusters),
            "multi_post_clusters": sum(1 for cluster in clusters if cluster.size >= 2),
            "alerts": len(alerts),
            "platforms": sorted({post.platform for post in posts}),
        },
        "posts": post_rows,
        "clusters": [cluster.to_dict() for cluster in clusters],
        "alerts": alerts,
        "repeat_authors": repeats,
    }


def payload_to_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def posts_csv(payload: dict) -> str:
    frame = pd.DataFrame(payload.get("posts", []))
    return _frame_csv(frame)


def clusters_csv(payload: dict) -> str:
    rows = []
    for cluster in payload.get("clusters", []):
        origin = cluster.get("patient_zero", {})
        rows.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "label": cluster.get("label"),
                "top_terms": ", ".join(cluster.get("top_terms", [])),
                "size": cluster.get("size"),
                "platforms": json.dumps(cluster.get("platforms", {})),
                "patient_zero_post_id": origin.get("post_id"),
                "patient_zero_platform": origin.get("platform"),
                "patient_zero_author": origin.get("author"),
                "patient_zero_created_at": origin.get("created_at"),
                "patient_zero_url": origin.get("url"),
                "patient_zero_title": origin.get("title"),
            }
        )
    return _frame_csv(pd.DataFrame(rows))


def alerts_csv(payload: dict) -> str:
    rows = []
    for alert in payload.get("alerts", []):
        rows.append(
            {
                "alert_id": alert.get("alert_id"),
                "indicator": alert.get("indicator"),
                "severity": alert.get("severity"),
                "score": alert.get("score"),
                "summary": alert.get("summary"),
                "authors": ", ".join(alert.get("authors", [])),
                "platforms": ", ".join(alert.get("platforms", [])),
                "post_ids": ", ".join(alert.get("post_ids", [])),
                "reasons": "; ".join(alert.get("reasons", [])),
                "time_start": alert.get("time_start"),
                "time_end": alert.get("time_end"),
                "evidence": json.dumps(alert.get("evidence", {})),
                "limitation": alert.get("limitation"),
            }
        )
    return _frame_csv(pd.DataFrame(rows))


def payload_to_html(payload: dict) -> str:
    """A single HTML report of posts, narratives, origins, and account groups."""
    query = payload.get("query", {})
    summary = payload.get("summary", {})
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>SOCMINT results</title>",
        "<style>body{font-family:Georgia,serif;max-width:920px;margin:2rem auto;padding:0 1rem;color:#1c1917}",
        "h1,h2{font-family:system-ui,sans-serif} article,section{border-top:1px solid #e7e5e4;padding:1rem 0}",
        "img{max-width:280px;display:block;margin:.4rem 0} .reason{display:inline-block;background:#f5f5f4;border-radius:999px;padding:.15rem .6rem;margin:.15rem .3rem .15rem 0}",
        "</style></head><body>",
        f"<h1>Results for {_esc(query.get('keyword', ''))}</h1>",
        f"<p>{_esc(query.get('start', '')[:10])} to {_esc(query.get('end', '')[:10])} UTC · "
        f"{_esc(query.get('mode', ''))} · platforms: {_esc(', '.join(query.get('platforms', [])))}</p>",
        f"<p>{summary.get('posts', 0)} posts · {summary.get('multi_post_clusters', 0)} multi-post narratives · "
        f"{summary.get('alerts', 0)} account groups.</p>",
        f"<p>{_esc(payload.get('disclaimer', ''))}</p>",
        "<h2>Posts</h2>",
    ]
    for post in payload.get("posts", []):
        parts.append("<article>")
        parts.append(
            f"<p><strong>{_esc(post.get('platform'))}</strong> · {_esc(post.get('author'))} · "
            f"{_esc(str(post.get('created_at', ''))[:16])}</p>"
        )
        parts.append(f"<p>{_esc(post.get('text') or 'This post has no text.')}</p>")
        parts.append(
            f"<p>Likes {post.get('likes') or 0} · Retweets {post.get('retweets') or 0} · "
            f"Replies {post.get('replies') or 0} · Views {post.get('views') or 0}</p>"
        )
        if post.get("url"):
            parts.append(f"<p><a href='{_esc(post['url'])}'>Open post</a></p>")
        for photo in (post.get("photos") or [])[:1]:
            parts.append(f"<img src='{_esc(photo)}' alt='Post photo'>")
        for video in (post.get("videos") or [])[:1]:
            parts.append(f"<p><a href='{_esc(video)}'>Video</a></p>")
        parts.append("</article>")
    parts.append("<h2>Main narratives</h2>")
    for cluster in payload.get("clusters", []):
        if int(cluster.get("size") or 0) < 2:
            continue
        origin = cluster.get("patient_zero") or {}
        parts.append("<section>")
        parts.append(
            f"<h3>{_esc(cluster.get('label'))}</h3>"
            f"<p>{cluster.get('size')} posts · platforms {_esc(json.dumps(cluster.get('platforms', {})))}</p>"
        )
        parts.append(
            f"<p>Started in this sample with {_esc(origin.get('author'))} on "
            f"{_esc(origin.get('platform'))}: {_esc(origin.get('text') or '')}</p>"
        )
        if origin.get("url"):
            parts.append(f"<p><a href='{_esc(origin['url'])}'>First post</a></p>")
        parts.append("<p>Spread:</p><ol>")
        for step in cluster.get("propagation") or []:
            parts.append(
                "<li>"
                f"{_esc(step.get('platform'))} · {_esc(step.get('author'))} · "
                f"{_esc(str(step.get('created_at', ''))[:16])} · "
                f"<a href='{_esc(step.get('url') or '')}'>Open</a>"
                "</li>"
            )
        parts.append("</ol></section>")
    parts.append("<h2>Account groups</h2>")
    alerts = payload.get("alerts") or []
    if not alerts:
        parts.append("<p>No account group met the current settings.</p>")
    for alert in alerts:
        parts.append("<section>")
        parts.append(f"<h3>{_esc(alert.get('alert_id'))} · {_esc(alert.get('summary'))}</h3>")
        for reason in alert.get("reasons") or []:
            parts.append(f"<span class='reason'>{_esc(reason)}</span>")
        parts.append(f"<p>Accounts: {_esc(', '.join(alert.get('authors', [])))}</p>")
        parts.append(f"<p>{_esc(alert.get('limitation', ''))}</p>")
        parts.append("</section>")
    parts.append("</body></html>")
    return "".join(parts)


def _esc(value) -> str:
    text = str(value or "")
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def save_run(payload: dict, directory: Path = RESULTS_DIR) -> dict[str, str]:
    """Write JSON and three CSV files. Returns the file paths as strings."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    keyword = str(payload.get("query", {}).get("keyword", "query"))
    slug = "".join(char.lower() if char.isalnum() else "-" for char in keyword)
    slug = "-".join(part for part in slug.split("-") if part)[:40] or "query"
    prefix = directory / f"{stamp}-{slug}"
    paths = {
        "json": Path(str(prefix) + ".json"),
        "posts_csv": Path(str(prefix) + "-posts.csv"),
        "clusters_csv": Path(str(prefix) + "-clusters.csv"),
        "alerts_csv": Path(str(prefix) + "-alerts.csv"),
    }
    paths["json"].write_text(payload_to_json(payload), encoding="utf-8")
    paths["posts_csv"].write_text(posts_csv(payload), encoding="utf-8")
    paths["clusters_csv"].write_text(clusters_csv(payload), encoding="utf-8")
    paths["alerts_csv"].write_text(alerts_csv(payload), encoding="utf-8")
    return {name: str(path) for name, path in paths.items()}


def _frame_csv(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    safe = frame.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(_csv_safe)
    return safe.to_csv(index=False)


def _csv_safe(value: object) -> object:
    if not isinstance(value, str):
        return value
    if value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value
