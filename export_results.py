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
                "time_start": alert.get("time_start"),
                "time_end": alert.get("time_end"),
                "evidence": json.dumps(alert.get("evidence", {})),
                "limitation": alert.get("limitation"),
            }
        )
    return _frame_csv(pd.DataFrame(rows))


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
