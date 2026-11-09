"""Group posts into narrative themes and summarize volume over time.

The default method is TF-IDF (word and word-pair counts, weighted by how
distinctive a term is) plus average-linkage clustering on cosine distance.
The query keyword is removed before vectorizing so the shared search term
does not glue unrelated posts into one cluster.

This step groups similar wording. It does not decide whether a claim is true.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

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

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=float(distance_threshold),
    )
    labels = model.fit_predict(distances)
    terms = vectorizer.get_feature_names_out()
    clusters: list[NarrativeCluster] = []
    for raw_label in sorted(set(labels), key=lambda label: int(label)):
        member_indexes = [i for i, value in enumerate(labels) if value == raw_label]
        members = [posts[i] for i in member_indexes]
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
