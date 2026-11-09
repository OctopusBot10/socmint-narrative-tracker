"""Shared records for collected posts and analysis output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


@dataclass
class Post:
    """One public post or video normalized across platforms."""

    post_id: str
    platform: str
    author: str
    community: str
    title: str
    text: str
    url: str
    created_at: datetime
    media_links: list[str] = field(default_factory=list)
    outbound_urls: list[str] = field(default_factory=list)
    ups: int | None = None
    views: int | None = None
    comments: int | None = None
    is_demo: bool = False

    def full_text(self) -> str:
        title = (self.title or "").strip()
        body = (self.text or "").strip()
        if title and body and body.lower() != title.lower():
            return f"{title}\n{body}"
        return title or body

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["created_at"] = utc_iso(self.created_at)
        payload["text_preview"] = self.full_text()[:280]
        return payload


@dataclass
class NarrativeCluster:
    """A group of posts the clustering step treated as one narrative theme."""

    cluster_id: int
    label: str
    top_terms: list[str]
    posts: list[Post]

    @property
    def size(self) -> int:
        return len(self.posts)

    def sorted_posts(self) -> list[Post]:
        return sorted(self.posts, key=lambda post: post.created_at)

    def patient_zero(self) -> Post:
        return self.sorted_posts()[0]

    def platform_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for post in self.posts:
            counts[post.platform] = counts.get(post.platform, 0) + 1
        return counts

    def first_on_each_platform(self) -> list[Post]:
        seen: set[str] = set()
        path: list[Post] = []
        for post in self.sorted_posts():
            if post.platform not in seen:
                seen.add(post.platform)
                path.append(post)
        return path

    def to_dict(self) -> dict:
        origin = self.patient_zero()
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "top_terms": self.top_terms,
            "size": self.size,
            "platforms": self.platform_counts(),
            "post_ids": [post.post_id for post in self.sorted_posts()],
            "patient_zero": {
                "post_id": origin.post_id,
                "platform": origin.platform,
                "author": origin.author,
                "community": origin.community,
                "created_at": utc_iso(origin.created_at),
                "url": origin.url,
                "title": origin.title,
                "text": origin.full_text(),
            },
            "propagation": [
                {
                    "platform": post.platform,
                    "post_id": post.post_id,
                    "author": post.author,
                    "created_at": utc_iso(post.created_at),
                    "url": post.url,
                    "title": post.title,
                }
                for post in self.first_on_each_platform()
            ],
        }
