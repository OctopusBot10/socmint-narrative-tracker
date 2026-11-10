"""Group posts into narrative themes and summarize volume over time.

The default method is TF-IDF (word and word-pair counts, weighted by how
distinctive a term is) plus DBSCAN on cosine distance. The sidebar distance
is the DBSCAN neighborhood radius (eps). The query keyword is removed before
vectorizing so the shared search term does not glue unrelated posts into one
cluster.

This step groups similar wording. It does not decide whether a claim is true.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

from coordination import detect_coordination
from models import NarrativeCluster, Post


def cluster_posts(
    posts: list[Post],
    *,
    keyword: str,
    distance_threshold: float = 0.62,
) -> tuple[list[NarrativeCluster], np.ndarray]:
    """Return clusters and a cosine-similarity matrix aligned with `posts`."""
    if not posts:
        return [], np.zeros((0, 0))

    texts = [_text_for_clustering(post.full_text(), keyword) for post in posts]
    if not any(text.strip() for text in texts):
        clusters = [
            _build_cluster(index + 1, [post], ["unclassified"])
            for index, post in enumerate(posts)
        ]
        return clusters, np.eye(len(posts))
    if len(posts) == 1:
        cluster = _build_cluster(1, posts, ["single post"])
        return [cluster], np.eye(1)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.85,
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        similarity = np.eye(len(posts))
        clusters = [
            _build_cluster(index + 1, [post], ["unclassified"])
            for index, post in enumerate(posts)
        ]
        return clusters, similarity

    if matrix.shape[1] == 0:
        similarity = np.eye(len(posts))
        clusters = [
            _build_cluster(index + 1, [post], ["unclassified"])
            for index, post in enumerate(posts)
        ]
        return clusters, similarity

    distances = np.nan_to_num(cosine_distances(matrix), nan=1.0)
    np.fill_diagonal(distances, 0.0)
    similarity = np.clip(1.0 - distances, 0.0, 1.0)

    model = DBSCAN(
        eps=float(distance_threshold),
        min_samples=2,
        metric="precomputed",
    )
    labels = model.fit_predict(distances)
    terms = vectorizer.get_feature_names_out()
    groups: dict[str, list[int]] = {}
    noise_count = 0
    for index, raw_label in enumerate(labels):
        if int(raw_label) == -1:
            key = f"noise-{noise_count}"
            noise_count += 1
        else:
            key = str(int(raw_label))
        groups.setdefault(key, []).append(index)
    clusters: list[NarrativeCluster] = []
    for key in groups:
        member_indexes = groups[key]
        members = [posts[i] for i in member_indexes]
        if key.startswith("noise-"):
            top_terms = ["unclustered"]
        else:
            top_terms = _top_terms(matrix, terms, member_indexes)
        clusters.append(_build_cluster(0, members, top_terms))

    clusters.sort(key=lambda cluster: (-cluster.size, cluster.patient_zero().created_at))
    for index, cluster in enumerate(clusters, start=1):
        cluster.cluster_id = index
    return clusters, similarity


def volume_frame(posts: list[Post], clusters: list[NarrativeCluster]) -> pd.DataFrame:
    """One row per post, with a day (or hour) bucket for the timeline chart."""
    if not posts:
        return pd.DataFrame(
            columns=[
                "post_id",
                "platform",
                "author",
                "created_at",
                "bucket",
                "cluster_id",
                "cluster_label",
                "title",
                "ups",
                "views",
                "comments",
            ]
        )

    cluster_by_post: dict[str, NarrativeCluster] = {}
    for cluster in clusters:
        for post in cluster.posts:
            cluster_by_post[post.post_id] = cluster

    rows = []
    for post in posts:
        cluster = cluster_by_post.get(post.post_id)
        rows.append(
            {
                "post_id": post.post_id,
                "platform": post.platform,
                "author": post.author,
                "created_at": pd.Timestamp(post.created_at),
                "cluster_id": cluster.cluster_id if cluster else None,
                "cluster_label": cluster.label if cluster else "",
                "title": post.title,
                "url": post.url,
                "ups": post.ups,
                "views": post.views,
                "comments": post.comments,
            }
        )
    frame = pd.DataFrame(rows)
    span = frame["created_at"].max() - frame["created_at"].min()
    if span <= pd.Timedelta(days=2):
        frame["bucket"] = frame["created_at"].dt.floor("h")
    else:
        frame["bucket"] = frame["created_at"].dt.floor("D")
    return frame


def analyze_records(
    records,
    *,
    keyword: str,
    distance_threshold: float = 0.62,
    window_hours: float = 6.0,
    text_similarity_threshold: float = 0.86,
) -> dict:
    """Cluster a unified post table and flag short-window copy or link patterns.

    `records` is a list of collector dicts or a DataFrame with columns
    id, platform, author, text, timestamp, url, and views. The coordination
    step is a coordinated-behavior heuristic: matching wording or a shared
    link inside the time window. It is not a determination of inauthentic
    behavior.
    """
    if isinstance(records, pd.DataFrame):
        rows = records.to_dict(orient="records")
    else:
        rows = list(records)
    posts = [_post_from_record(row) for row in rows]
    clusters, similarity = cluster_posts(
        posts, keyword=keyword, distance_threshold=distance_threshold
    )
    alerts = detect_coordination(
        posts,
        clusters,
        similarity,
        window_hours=window_hours,
        text_similarity_threshold=text_similarity_threshold,
    )
    _attach_claim_scores(clusters)
    return {
        "posts": posts,
        "clusters": clusters,
        "similarity": similarity,
        "frame": volume_frame(posts, clusters),
        "alerts": alerts,
    }


def _attach_claim_scores(clusters: list[NarrativeCluster]) -> None:
    """Store a ClaimBuster checkworthiness percent on each cluster.

    The call is skipped when CLAIMBUSTER_API_KEY is empty. The dashboard then
    uses its local checkworthiness score instead of showing every card at 0.
    """
    from collector_claimbuster import get_claim_score, has_api_key

    if not has_api_key():
        return
    for cluster in clusters:
        summary = cluster.patient_zero().full_text().strip()
        cluster.claim_score = get_claim_score(summary) if summary else 0.0


def _post_from_record(row: dict) -> Post:
    created = _created_at(row.get("timestamp") or row.get("created_at"))
    platform = str(row.get("platform") or "")
    url = str(row.get("url") or "")
    author = str(row.get("author") or "")
    engagement = row.get("engagement") if isinstance(row.get("engagement"), dict) else {}
    photos = _url_list(row.get("photos"))
    videos = _url_list(row.get("videos"))
    view_count = _count(engagement.get("views", row.get("views")))
    return Post(
        post_id=str(row.get("id") or row.get("post_id") or ""),
        platform=platform,
        author=author,
        community=_community_from_url(platform, url, author),
        title="",
        text=str(row.get("text") or ""),
        url=url,
        created_at=created,
        media_links=[*photos, *videos],
        photos=photos,
        videos=videos,
        views=view_count,
        likes=_count(engagement.get("likes")),
        retweets=_count(engagement.get("retweets")),
        replies=_count(engagement.get("replies")),
        outbound_urls=_url_list(row.get("outbound_urls")),
        account_created=_optional_time(row.get("account_created")),
        profile_image=_text(row.get("profile_image")),
        amplified_authors=[
            str(name).lstrip("@")
            for name in _name_list(row.get("amplified_authors"))
            if str(name).strip()
        ],
        authenticity_score=_optional_score(row.get("authenticity_score")),
        bot_risk_pct=_optional_score(row.get("bot_risk_pct")),
        risk_flags=_flag_list(row.get("risk_flags")),
    )


def _optional_time(value) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if isinstance(value, datetime):
        created = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return created.astimezone(timezone.utc)
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    parsed = None
    for attempt in (candidate, text):
        try:
            parsed = datetime.fromisoformat(attempt)
            break
        except ValueError:
            parsed = None
    if parsed is None:
        for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                parsed = None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _created_at(value) -> datetime:
    """Parse a collector timestamp. A missing or unreadable value stays sortable."""
    if isinstance(value, datetime):
        created = value
    else:
        text = str(value or "").strip()
        created = None
        if text:
            candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
            for attempt in (candidate, text):
                try:
                    created = datetime.fromisoformat(attempt)
                    break
                except ValueError:
                    created = None
            if created is None:
                for fmt in ("%a %b %d %H:%M:%S %z %Y", "%Y-%m-%d %H:%M:%S"):
                    try:
                        created = datetime.strptime(text, fmt)
                        break
                    except ValueError:
                        created = None
        if created is None:
            created = datetime(1970, 1, 1, tzinfo=timezone.utc)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.astimezone(timezone.utc)


def _flag_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None or isinstance(value, float):
        return []
    text = str(value).strip()
    return [text] if text else []


def _optional_score(value) -> int | None:
    if value is None or isinstance(value, float) and value != value:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _url_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item.startswith("http")]


def _name_list(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else []


def _text(value) -> str:
    if value is None or isinstance(value, float):
        return ""
    return str(value)


def _count(value) -> int:
    try:
        return int(value) if value not in (None, "") else 0
    except (TypeError, ValueError):
        return 0


def _community_from_url(platform: str, url: str, author: str) -> str:
    parts = [part for part in url.split("/") if part]
    if platform == "telegram" and len(parts) >= 2:
        return parts[-2]
    if platform == "twitter" and "status" in parts:
        return parts[parts.index("status") - 1]
    return author


def _build_cluster(
    cluster_id: int,
    posts: list[Post],
    top_terms: list[str],
) -> NarrativeCluster:
    label = ", ".join(top_terms[:4]) if top_terms else "unlabeled narrative"
    if len(posts) == 1 and top_terms == ["single post"]:
        label = _fallback_label(posts[0])
    if not top_terms or top_terms == ["unclassified"]:
        label = _fallback_label(posts[0])
        top_terms = []
    return NarrativeCluster(
        cluster_id=cluster_id,
        label=label,
        top_terms=top_terms[:6],
        posts=posts,
    )


def _top_terms(matrix, terms: np.ndarray, indexes: list[int], limit: int = 6) -> list[str]:
    if len(terms) == 0 or not indexes:
        return []
    weights = np.asarray(matrix[indexes].mean(axis=0)).ravel()
    if weights.size == 0 or float(weights.max()) <= 0:
        return []
    ranked = weights.argsort()[::-1]
    chosen: list[str] = []
    for index in ranked:
        term = str(terms[index])
        if weights[index] <= 0:
            break
        if any(term in other or other in term for other in chosen):
            continue
        chosen.append(term)
        if len(chosen) >= limit:
            break
    return chosen


def _text_for_clustering(text: str, keyword: str) -> str:
    cleaned = text.lower()
    for token in _keyword_tokens(keyword):
        cleaned = re.sub(rf"\b{re.escape(token)}\b", " ", cleaned)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip() or "empty"


def _keyword_tokens(keyword: str) -> list[str]:
    raw = keyword.strip().lower()
    tokens = [raw, raw.lstrip("#"), raw.replace("#", " ")]
    tokens.extend(part for part in re.split(r"[^a-z0-9]+", raw) if len(part) > 2)
    unique: list[str] = []
    for token in tokens:
        token = token.strip()
        if token and token not in unique:
            unique.append(token)
    return unique


def _fallback_label(post: Post) -> str:
    words = post.title.strip() or post.full_text()
    words = " ".join(words.split())
    if len(words) > 72:
        return words[:69] + "..."
    return words or "untitled post"
