"""Streamlit dashboard for the SOCMINT narrative tracker."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import analyze_records
from analysis_claims import WAF_FALLBACK_NOTICE, score_narratives
from config import PLATFORM_LABELS, env
from coordination import repeat_authors
from export_results import (
    alerts_csv,
    build_payload,
    clusters_csv,
    payload_to_html,
    payload_to_json,
    posts_csv,
    save_run,
)
from models import NarrativeCluster, Post
import collector_telegram
import collector_twitter
import collector_youtube
from scrapers import CollectorError
from utils.social_analyzer_helper import run_social_analyzer_lookup

st.set_page_config(
    page_title="SOCMINT Narrative Tracker",
    page_icon="S",
    layout="wide",
)


def main() -> None:
    st.markdown(_claim_css(), unsafe_allow_html=True)
    st.title("SOCMINT Narrative Tracker")
    st.caption(
        "Posts in the date range, the narratives they form, where each narrative "
        "started, and groups of accounts that moved together. Clusters are not "
        "fact-checks. Account groups are not proof of a shared operator."
    )

    (
        keyword,
        start,
        end,
        platforms,
        channels,
        use_demo,
        max_posts,
        distance,
        window,
        text_threshold,
    ) = _sidebar()

    if st.button("Run collection", type="primary", use_container_width=True):
        with st.spinner("Collecting posts and running the heuristics..."):
            try:
                result = _run(
                    keyword=keyword,
                    start=start,
                    end=end,
                    platforms=platforms,
                    channels=channels,
                    use_demo=use_demo,
                    max_posts=max_posts,
                    distance=distance,
                    window=window,
                    text_threshold=text_threshold,
                )
            except CollectorError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"The run failed before producing results: {exc}")
                return
        for notice in _banner_notices(result):
            if notice == WAF_FALLBACK_NOTICE:
                st.toast(notice)
            elif _is_live_success(notice):
                st.toast(notice, icon="✅")
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if not result:
        st.info(
            "Demo mode is on. Leave the sample keyword and press Run collection. "
            "Switch demo off for a live X search and public Telegram channel previews."
        )
        _method_notes()
        return

    if result["query_key"] != _query_key(
        keyword,
        start,
        end,
        platforms,
        channels,
        use_demo,
        max_posts,
        distance,
        window,
        text_threshold,
    ):
        st.warning(
            "Sidebar settings differ from the results below. Press Run collection "
            "to refresh."
        )

    if WAF_FALLBACK_NOTICE in (result.get("notices") or []):
        st.sidebar.caption(WAF_FALLBACK_NOTICE)
    _render_dashboard(result)
    _method_notes()


def _sidebar() -> tuple:
    today = date.today()
    with st.sidebar:
        st.header("Query")
        keyword = st.text_input(
            "Keyword",
            value="Acme Bank",
            help="Quoted on X when it contains more than one word. Telegram channels are searched separately.",
        ).strip()
        platforms = st.multiselect(
            "Platforms",
            options=["twitter", "telegram", "youtube"],
            default=["twitter", "telegram", "youtube"],
            format_func=lambda key: PLATFORM_LABELS[key],
        )
        channels_text = ""
        if "telegram" in platforms:
            channels_text = st.text_area(
                "Telegram channels",
                value="acme_notice",
                help="Public t.me channel names, one per line. No login.",
            )
        st.caption(
            "X is a keyword search. Telegram reads t.me/s for each channel. YouTube reads the public results page."
        )
        start_date = st.date_input("Start date", value=today - timedelta(days=14))
        end_date = st.date_input("End date", value=today)
        use_demo = st.toggle(
            "Demo data",
            value=True,
            help="Synthetic posts for this keyword and date range. Live mode is off while this is on.",
        )
        max_posts = st.slider("Max posts", min_value=10, max_value=100, value=40)

        st.header("Heuristics")
        distance = st.slider(
            "Cluster distance",
            min_value=0.35,
            max_value=0.8,
            value=0.62,
            step=0.01,
            help="DBSCAN neighborhood radius on cosine distance. Higher values merge posts more readily.",
        )
        window = st.slider("Coordination window (hours)", min_value=1, max_value=48, value=6)
        text_threshold = st.slider(
            "Near-copy similarity",
            min_value=0.7,
            max_value=0.98,
            value=0.86,
            step=0.01,
        )
        if not use_demo and "twitter" in platforms:
            st.caption("Live X search reads a public Nitter HTML page. No X API key is used.")

    channels = [
        line.strip().lstrip("@")
        for line in channels_text.splitlines()
        if line.strip()
    ]
    if "telegram" in platforms and not channels:
        channels = ["telegram", "durov"]
    start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    end = datetime.combine(end_date, time.max.replace(microsecond=0), tzinfo=timezone.utc)
    if end < start:
        start, end = end, start
    return (
        keyword,
        start,
        end,
        platforms,
        channels,
        use_demo,
        max_posts,
        distance,
        float(window),
        float(text_threshold),
    )


def _run(
    *,
    keyword: str,
    start: datetime,
    end: datetime,
    platforms: list[str],
    channels: list[str],
    use_demo: bool,
    max_posts: int,
    distance: float,
    window: float,
    text_threshold: float,
) -> dict:
    search = keyword.strip()
    if not search:
        raise CollectorError("Enter a keyword before running.")
    if not platforms:
        raise CollectorError("Select at least one platform.")

    notices: list[str] = []
    diagnostics: list[str] = []
    records: list[dict] = []
    active: list[str] = []
    if use_demo:
        mode = "demo"
        records, active = _sample_records(search, start, end, platforms, channels, max_posts)
    else:
        mode = "live"
        failures: list[str] = []
        if "twitter" in platforms:
            try:
                twitter_rows = collector_twitter.collect(
                    keyword,
                    start_date=start,
                    end_date=end,
                    max_posts=max_posts,
                )
                if twitter_rows:
                    records.extend(twitter_rows)
                    active.append("twitter")
                elif collector_twitter.last_warning:
                    failures.append(collector_twitter.last_warning)
            except collector_twitter.MissingApiKeyError as exc:
                failures.append(str(exc))
            except CollectorError as exc:
                failures.append(str(exc))
            except Exception as exc:
                failures.append(f"Twitter: {exc}")
        if "telegram" in platforms and channels:
            try:
                telegram_rows = collector_telegram.collect(
                    channels,
                    start_date=start,
                    end_date=end,
                    max_posts=max_posts,
                    store=False,
                    query=keyword,
                )
                if telegram_rows:
                    records.extend(telegram_rows)
                    active.append("telegram")
                elif collector_telegram.last_warning:
                    failures.append(collector_telegram.last_warning)
            except CollectorError as exc:
                failures.append(str(exc))
            except Exception as exc:
                failures.append(f"Telegram: {exc}")
        if "youtube" in platforms:
            try:
                youtube_rows = collector_youtube.collect(
                    keyword,
                    start_date=start,
                    end_date=end,
                    max_posts=max_posts,
                )
                if youtube_rows:
                    records.extend(youtube_rows)
                    active.append("youtube")
                elif collector_youtube.last_warning:
                    failures.append(collector_youtube.last_warning)
            except CollectorError as exc:
                failures.append(str(exc))
            except Exception as exc:
                failures.append(f"YouTube: {exc}")
        live_youtube = sum(1 for row in records if str(row.get("platform", "")).lower() == "youtube")
        live_telegram = sum(1 for row in records if str(row.get("platform", "")).lower() == "telegram")
        if not records:
            notices.append(WAF_FALLBACK_NOTICE)
            diagnostics.extend(failures)
            from analysis_claims import synthetic_crisis_records

            names = []
            if "twitter" in platforms:
                names.append("Twitter")
            if "telegram" in platforms:
                names.append("Telegram")
            if "youtube" in platforms:
                names.append("YouTube")
            records = synthetic_crisis_records(search, min(max_posts, 5), names)
            active = sorted({str(row.get("platform", "")).lower() for row in records})
            mode = "demo"
        elif live_youtube or live_telegram:
            notices.append(f"Fetched {len(records)} live posts across platforms.")
            diagnostics.extend(failures)
        else:
            diagnostics.extend(failures)

    if not records:
        raise CollectorError(" ".join(notices) or "No posts were collected.")

    table = pd.DataFrame(records)
    analyzed = analyze_records(
        table,
        keyword=search,
        distance_threshold=distance,
        window_hours=window,
        text_similarity_threshold=text_threshold,
    )
    posts = analyzed["posts"]
    clusters = analyzed["clusters"]
    alerts = analyzed["alerts"]
    active = active or ["twitter"]
    repeats = repeat_authors(alerts)
    payload = build_payload(
        keyword=search,
        start=start,
        end=end,
        mode=mode,
        platforms=active,
        posts=posts,
        clusters=clusters,
        alerts=alerts,
        repeats=repeats,
    )
    return {
        "query_key": _query_key(
            keyword,
            start,
            end,
            platforms,
        channels,
        use_demo,
            max_posts,
            distance,
            window,
            text_threshold,
        ),
        "notices": notices,
        "diagnostics": diagnostics,
        "posts": posts,
        "clusters": clusters,
        "frame": analyzed["frame"],
        "payload": payload,
    }


def _sample_records(
    search: str,
    start: datetime,
    end: datetime,
    platforms: list[str],
    channels: list[str],
    max_posts: int,
) -> tuple[list[dict], list[str]]:
    """Synthetic posts used by demo mode and by the offline fallback."""
    records: list[dict] = []
    active: list[str] = []
    if "twitter" in platforms:
        records.extend(collector_twitter.demo_records(search, start, end, limit=max_posts))
        active.append("twitter")
    if "telegram" in platforms:
        records.extend(
            collector_telegram.demo_records(
                channels or ["telegram", "durov"],
                search,
                start,
                end,
                limit=max_posts,
            )
        )
        active.append("telegram")
    if "youtube" in platforms:
        records.extend(collector_youtube.demo_records(search, start, end, limit=max_posts))
        active.append("youtube")
    return records, active


def _is_live_success(notice: str) -> bool:
    return notice.startswith("Fetched ") and "live posts" in notice


def _banner_notices(result: dict) -> list[str]:
    """Top-of-page notices. A live fetch keeps only the green success line."""
    notices = list(result.get("notices") or [])
    if any(_is_live_success(notice) for notice in notices):
        return [notice for notice in notices if _is_live_success(notice)]
    return [notice for notice in notices if notice == WAF_FALLBACK_NOTICE or _is_live_success(notice)]


def _diagnostic_notices(result: dict) -> list[str]:
    """Platform scraper warnings, kept out of the main banner stack."""
    notices = list(result.get("notices") or [])
    diagnostics = list(result.get("diagnostics") or [])
    banners = set(_banner_notices(result))
    diagnostics.extend(notice for notice in notices if notice not in banners)
    unique: list[str] = []
    for item in diagnostics:
        if item and item not in unique:
            unique.append(item)
    return unique


def _render_dashboard(result: dict) -> None:
    payload = result["payload"]
    posts: list[Post] = result["posts"]
    clusters: list[NarrativeCluster] = result["clusters"]
    query = payload["query"]
    alerts = payload["alerts"]

    for notice in _banner_notices(result):
        if notice == WAF_FALLBACK_NOTICE:
            st.caption(notice)
        elif _is_live_success(notice):
            st.success(notice)
    diagnostics = _diagnostic_notices(result)
    if diagnostics:
        with st.expander("Scraper Diagnostics"):
            for item in diagnostics:
                st.markdown(f"- {item}")
    if query["mode"] == "demo":
        st.info("These posts are synthetic. The links were not fetched from X, Telegram, or YouTube.")

    if not posts:
        st.warning(
            "No posts came back for that keyword and date range. Widen the dates, "
            "check the spelling, or switch to demo data."
        )
        return

    assessments = score_narratives(clusters, posts)
    st.markdown(_claim_css(), unsafe_allow_html=True)
    selected = st.session_state.get("claim_detail")
    chosen = next((item for item in assessments if item["cluster_id"] == selected), None)
    if chosen is not None:
        cluster = next(item for item in clusters if item.cluster_id == chosen["cluster_id"])
        _claim_detail(chosen, cluster, alerts)
    else:
        _claim_home(assessments, posts, alerts, payload.get("repeat_authors", []))

    with st.expander("Posts and volume over time"):
        _charts(result["frame"])
        _posts_section(posts)
    _export_section(payload)


def _claim_css() -> str:
    return """
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
      background:#0E1117; color:#C9D1D9;
    }
    [data-testid="stSidebar"] {background:#0E1117; border-right:1px solid #30363D;}
    .stApp h1, .stApp h2, .stApp h3, .cm-hero h2, .cm-panel h3 {
      color:#F0F6FC;
    }
    .stApp p, .stApp li, .stMarkdown, .cm-hero p, .cm-evidence, .cm-feed td {
      color:#C9D1D9;
    }
    hr, .stApp hr {border:none; border-top:1px solid #30363D; background:#30363D; height:1px;}
    [data-testid="stVerticalBlockBorderWrapper"] {
      background:#161B22; border:1px solid #30363D; border-radius:12px;
    }
    [data-testid="stMetric"] {
      background:#161B22; border:1px solid #30363D; border-radius:12px; padding:0.8rem 1rem;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {color:#C9D1D9;}
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] p {color:#F0F6FC;}
    .cm-hero {text-align:center; margin:0.2rem 0 1rem;}
    .cm-hero h2 {margin:0; font-size:1.8rem;}
    .cm-hero p {margin:0.35rem 0 0;}
    .cm-stats, .cm-panel, .cm-key, .cm-fact, .cm-metric {
      background:#161B22; border:1px solid #30363D; color:#C9D1D9;
    }
    .cm-stats {display:flex; align-items:center; justify-content:space-between; gap:1rem;
      border-radius:16px; padding:1rem 1.4rem; margin-bottom:1rem;}
    .cm-live {color:#3FB950; font-weight:700; letter-spacing:0.06em;}
    .cm-stat {text-align:center; min-width:7rem;}
    .cm-stat b {display:block; color:#388BFD; font-size:1.8rem; line-height:1.1;}
    .cm-stat span, .cm-when, .cm-metric span {color:#C9D1D9;}
    .cm-panel {border-radius:16px; padding:0.8rem 1rem 0.4rem; margin-bottom:1rem;}
    .cm-panel h3 {margin:0 0 0.6rem; font-size:1rem;}
    .cm-feed {width:100%; border-collapse:collapse;}
    .cm-feed td {padding:0.55rem 0.35rem; border-top:1px solid #30363D; vertical-align:middle;}
    .cm-when {font-size:0.82rem; white-space:nowrap;}
    .cm-badge {display:inline-block; font-size:0.72rem; font-weight:700; letter-spacing:0.04em;
      padding:0.18rem 0.55rem; border-radius:999px; color:#F0F6FC;}
    .cm-FALSE, .cm-DISPUTED, .cm-auth-low {background:#DA3633; color:#F0F6FC;}
    .cm-VERIFIED, .cm-LOW_RISK, .cm-auth-high {background:#238636; color:#F0F6FC;}
    .cm-UNVERIFIED, .cm-auth-mid {background:#9E6A03; color:#F0F6FC;}
    .cm-up {color:#3FB950; font-weight:700;}
    .cm-down {color:#DA3633; font-weight:700;}
    .cm-flat {color:#C9D1D9; font-weight:700;}
    .cm-card-rule {height:4px; border-radius:4px; margin:-0.4rem 0 0.7rem;}
    .cm-rule-DISPUTED, .cm-rule-FALSE {background:#DA3633;}
    .cm-rule-UNVERIFIED {background:#388BFD;}
    .cm-rule-LOW_RISK, .cm-rule-VERIFIED {background:#238636;}
    .cm-evidence {font-size:0.92rem; margin:0.15rem 0;}
    .cm-confidence {display:flex; align-items:center; gap:0.6rem; color:#C9D1D9; font-size:0.85rem; margin-top:0.6rem;}
    .cm-confidence b {color:#F0F6FC;}
    .cm-bar {flex:1; height:8px; background:#30363D; border-radius:999px; overflow:hidden;}
    .cm-bar span {display:block; height:100%; border-radius:999px;}
    .cm-fill-DISPUTED, .cm-fill-FALSE {background:#DA3633;}
    .cm-fill-UNVERIFIED {background:#388BFD;}
    .cm-fill-LOW_RISK, .cm-fill-VERIFIED {background:#238636;}
    .cm-fact {border-left:3px solid #388BFD; border-radius:8px; padding:0.55rem 0.8rem; margin:0.35rem 0;}
    .cm-key {border-radius:12px; padding:0.8rem 1rem;}
    .cm-metric {border-radius:12px; padding:0.85rem 0.6rem; text-align:center;}
    .cm-metric span {display:block; font-size:0.72rem; letter-spacing:0.06em;}
    .cm-metric b {font-size:1.35rem; color:#F0F6FC;}
    </style>
    """


def _claim_home(
    assessments: list[dict],
    posts: list[Post],
    alerts: list[dict],
    repeats: list[dict],
) -> None:
    st.markdown(
        "<div class='cm-hero'><h2>Narratives</h2>"
        "<p>What is being said about the topic, grouped from the posts in this run.</p></div>",
        unsafe_allow_html=True,
    )
    _claim_metrics(assessments)
    _claim_feed(assessments)
    _claim_grid(assessments)
    _account_groups(alerts, repeats)
    _recent_timeline(assessments, posts)


def _claim_metrics(assessments: list[dict]) -> None:
    active = len(assessments)
    verified = sum(1 for item in assessments if item["status"] == "VERIFIED" and item["today"])
    disputed = sum(1 for item in assessments if item["status"] == "FALSE" and item["today"])
    st.markdown(
        "<div class='cm-stats'>"
        "<div class='cm-live'>● LIVE</div>"
        f"<div class='cm-stat'><b>{active}</b><span>Active Narratives</span></div>"
        f"<div class='cm-stat'><b>{verified}</b><span>Verified Today</span></div>"
        f"<div class='cm-stat'><b>{disputed}</b><span>Disputed Today</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _claim_feed(assessments: list[dict]) -> None:
    query = st.text_input("Search narratives", placeholder="Search title, status, or summary").strip().lower()
    rows = [
        item
        for item in assessments
        if not query or query in f"{item['title']} {item['summary']} {item['status']}".lower()
    ]
    if not rows:
        st.info("No narratives match that search.")
        return
    body = "".join(
        "<tr>"
        f"<td class='cm-when'>{_html(_ago(item['updated']))}</td>"
        f"<td><span class='cm-badge cm-{item['status']}'>{_badge_label(item['status'])}</span></td>"
        f"<td>{_html(item['title'])}</td>"
        "</tr>"
        for item in rows
    )
    st.markdown(
        "<div class='cm-panel'><h3>Live Narrative Feed</h3>"
        f"<table class='cm-feed'><tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _claim_grid(assessments: list[dict]) -> None:
    columns = st.columns(3)
    for index, item in enumerate(assessments):
        with columns[index % 3]:
            with st.container(border=True):
                st.markdown(
                    f"<div class='cm-card-rule cm-rule-{item['status']}'></div>"
                    "<div style='display:flex;justify-content:space-between;align-items:center'>"
                    f"<span class='cm-badge cm-{item['status']}'>{_badge_label(item['status'])}</span>"
                    f"{_authenticity_badge(item)}"
                    f"<span class='{_trend_class(item['trend_pct'])}'>↗ {_trend_label(item['trend_pct'])}</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{item['title']}**")
                st.write(item["summary"])
                platforms = item.get("platforms") or {}
                platform_text = ", ".join(
                    f"{name} {count}" for name, count in platforms.items()
                ) or "this sample"
                st.markdown(
                    f"<div class='cm-evidence'>Started with {_html(item.get('author') or 'unknown')} "
                    f"· {_html(platform_text)}</div>",
                    unsafe_allow_html=True,
                )
                for source in _source_names(item)[:3]:
                    st.markdown(f"<div class='cm-evidence'>▷ {_html(source)}</div>", unsafe_allow_html=True)
                st.markdown(_confidence_bar(item), unsafe_allow_html=True)
                actions = st.columns(2)
                if actions[0].button("Share", key=f"share-{item['cluster_id']}", use_container_width=True):
                    st.session_state["claim_share"] = item["cluster_id"]
                if actions[1].button("Details", key=f"details-{item['cluster_id']}", use_container_width=True):
                    st.session_state["claim_detail"] = item["cluster_id"]
                    st.rerun()
                if st.session_state.get("claim_share") == item["cluster_id"]:
                    st.code(_share_text(item), language=None)


def _recent_timeline(assessments: list[dict], posts: list[Post]) -> None:
    by_id = {item["cluster_id"]: item for item in assessments}
    rows = []
    for post in sorted(posts, key=lambda item: item.created_at, reverse=True)[:6]:
        status = "UNVERIFIED"
        for item in assessments:
            if any(source.startswith(post.author) for source in _source_names(item)):
                status = item["status"]
                break
        rows.append(
            "<div class='cm-evidence'>"
            f"<span class='cm-live'>●</span> {_html(_ago(post.created_at))} "
            f"<span class='cm-badge cm-{status}'>{_badge_label(status)}</span> "
            f"{_html(post.full_text()[:110] or 'This post has no text.')}"
            "</div>"
        )
    if not rows:
        return
    del by_id
    st.markdown(
        "<div class='cm-panel'><h3>Recent Narrative Timeline</h3>" + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


def _claim_detail(item: dict, cluster: NarrativeCluster, alerts: list[dict]) -> None:
    if st.button("← Back to Narratives"):
        st.session_state["claim_detail"] = None
        st.rerun()
    with st.container(border=True):
        st.header(item["title"])
        st.markdown(
            "<div style='display:flex;align-items:center;gap:0.8rem'>"
            f"<span class='cm-badge cm-{item['status']}'>{_badge_label(item['status'])}</span>"
            f"{_authenticity_badge(item)}"
            f"{_confidence_bar(item, suffix='/100')}"
            "</div>",
            unsafe_allow_html=True,
        )
        st.write(item["summary"])
        metrics = st.columns(3)
        for column, label, value in (
            (metrics[0], "FIRST SEEN", _ago(item["first_seen"])),
            (metrics[1], "TRENDING", _trend_label(item["trend_pct"])),
            (metrics[2], "MENTIONS", str(item["mentions"])),
        ):
            column.markdown(
                f"<div class='cm-metric'><span>{label}</span><b>{_html(value)}</b></div>",
                unsafe_allow_html=True,
            )
        _osint_panel(item)
    overview, evidence, timeline, correlation, audits = st.tabs(
        ["Overview", "Evidence", "Timeline", "Correlation", "Audits"]
    )
    with overview:
        st.subheader("Why this status")
        for fact in item["key_facts"] or [item["why"]]:
            st.markdown(f"<div class='cm-fact'>{_html(fact)}</div>", unsafe_allow_html=True)
        st.subheader("Key Facts")
        st.markdown(f"<div class='cm-key'>{_html(item['why'])}</div>", unsafe_allow_html=True)
    with evidence:
        if not item["evidence"]:
            st.info("This narrative has no source posts.")
        for row in item["evidence"]:
            with st.container(border=True):
                st.markdown(f"**{row['source']}** · `{row['stance']}` · {row['timestamp']}")
                st.write(row["text"])
                if row["url"]:
                    st.markdown(f"[Open source]({row['url']})")
    with timeline:
        for post in cluster.sorted_posts():
            st.markdown(
                f"**{_ago(post.created_at)}** · {post.platform} · {post.author}  \n"
                f"{post.full_text().strip() or 'This post has no text.'}"
            )
    with correlation:
        origin = cluster.patient_zero()
        st.write(f"First account in this sample: {origin.author} on {origin.platform}.")
        for step, post in enumerate(cluster.sorted_posts(), start=1):
            hours = (post.created_at - origin.created_at).total_seconds() / 3600.0
            link = f" · [Open post]({post.url})" if post.url else ""
            st.markdown(
                f"{step}. {post.platform} · {post.author} · "
                f"{post.created_at.strftime('%Y-%m-%d %H:%M UTC')} "
                f"({hours:.1f} h after the first post){link}"
            )
        related = _alerts_for_cluster(cluster, alerts)
        if related:
            _account_groups(related, [])
        else:
            st.info("No coordinated-account pattern matched the posts in this narrative.")
    with audits:
        st.markdown(f"<div class='cm-fact'>{_html(item['why'])}</div>", unsafe_allow_html=True)
        st.write(
            f"Score source: {item.get('score_source', 'local')} · "
            f"Confidence {item['confidence']}% · Mentions {item['mentions']}"
        )


def _auth_label(score: int) -> tuple[str, str]:
    if score > 75:
        return "AUTHENTIC PROXY", "high"
    if score >= 50:
        return "UNVERIFIED FOOTPRINT", "mid"
    return "SUSPICIOUS BOT PATTERN", "low"


def _authenticity_badge(item: dict) -> str:
    score = int(item.get("authenticity_score") or 0)
    label, tone = _auth_label(score)
    return (
        f"<span class='cm-badge cm-auth-{tone}'>{_html(label)} {score}%</span>"
    )


def _osint_panel(item: dict) -> None:
    score = int(item.get("authenticity_score") or 0)
    risk = int(item.get("bot_risk_pct") or 0)
    author = str(item.get("author") or "")
    row_id = item.get("cluster_id")
    with st.expander("OSINT & Social Analyzer Footprint"):
        st.markdown(_authenticity_badge(item), unsafe_allow_html=True)
        st.metric("Bot Risk Score", f"{risk}%")
        st.markdown(
            "<div class='cm-confidence'><span>Bot risk</span>"
            f"<div class='cm-bar'><span class='cm-fill-FALSE' style='width:{risk}%'></span></div>"
            f"<b>{risk}%</b></div>",
            unsafe_allow_html=True,
        )
        flags = item.get("risk_flags") or []
        if flags:
            for flag in flags:
                st.markdown(f"- {_html(flag)}")
        else:
            st.markdown("- No bot-risk flag matched this author.")
        if st.button("Run Social Analyzer OSINT Lookup", key=f"osint_{row_id}"):
            st.session_state[f"osint_result_{row_id}"] = run_social_analyzer_lookup(author)
        found = st.session_state.get(f"osint_result_{row_id}")
        if found is not None:
            st.caption(f"Cross-platform footprint for {author or 'this author'}: YouTube, Twitter, Telegram.")
            st.json(found)


def _badge_label(status: str) -> str:
    return status.replace("_", " ")


def _ago(value: datetime) -> str:
    moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    minutes = int((datetime.now(timezone.utc) - moment.astimezone(timezone.utc)).total_seconds() // 60)
    if minutes < 60:
        return f"{max(minutes, 1)} min"
    hours = minutes // 60
    if hours < 48:
        return "1 hour ago" if hours == 1 else f"{hours} hours ago"
    days = hours // 24
    return "1 day ago" if days == 1 else f"{days} days ago"


def _confidence_bar(item: dict, suffix: str = "%") -> str:
    score = max(0, min(int(item["confidence"]), 100))
    label = f"{score}{suffix}" if suffix == "/100" else f"{score}%"
    return (
        "<div class='cm-confidence'><span>Confidence:</span>"
        f"<div class='cm-bar'><span class='cm-fill-{item['status']}' style='width:{score}%'></span></div>"
        f"<b>{label}</b></div>"
    )


def _source_names(item: dict) -> list[str]:
    names: list[str] = []
    for row in item["evidence"]:
        if row["source"] not in names:
            names.append(row["source"])
    return names


def _trend_label(value: int) -> str:
    if value > 0:
        return f"+{value}%"
    return f"{value}%"


def _trend_class(value: int) -> str:
    if value > 0:
        return "cm-up"
    if value < 0:
        return "cm-down"
    return "cm-flat"


def _share_text(item: dict) -> str:
    return (
        f"{item['title']}\n"
        f"Status: {_badge_label(item['status'])} · Confidence {item['confidence']}% · "
        f"Trend {_trend_label(item['trend_pct'])}\n"
        f"{item['summary']}"
    )


def _html(value: object) -> str:
    text = str(value or "")
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _charts(frame: pd.DataFrame) -> None:
    st.subheader("Volume over time, by platform")
    volume = (
        frame.groupby(["bucket", "platform"], as_index=False)
        .size()
        .rename(columns={"size": "posts"})
    )
    timeline = px.bar(
        volume,
        x="bucket",
        y="posts",
        color="platform",
        barmode="stack",
        labels={"bucket": "Time (UTC)", "posts": "Posts", "platform": "Platform"},
    )
    timeline.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    st.plotly_chart(timeline, use_container_width=True)
    st.subheader("What is being said, by platform")

    by_cluster = (
        frame.groupby(["cluster_label", "platform"], as_index=False)
        .size()
        .rename(columns={"size": "posts"})
    )
    breakdown = px.bar(
        by_cluster,
        x="posts",
        y="cluster_label",
        color="platform",
        orientation="h",
        labels={"cluster_label": "Narrative", "posts": "Posts", "platform": "Platform"},
    )
    breakdown.update_layout(
        height=max(280, 42 * frame["cluster_label"].nunique()),
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis={"categoryorder": "total ascending"},
        legend_title_text="",
    )
    st.plotly_chart(breakdown, use_container_width=True)


def _cluster_section(clusters: list[NarrativeCluster]) -> None:
    st.subheader("Main narratives")
    st.caption(
        "Each row is one wording group. The earliest post is the first account "
        "in this sample that pushed it. Later platforms show how that wording spread."
    )
    show_singletons = st.checkbox("Show single-post clusters", value=False)
    visible = [
        cluster for cluster in clusters if cluster.size >= 2 or show_singletons
    ]
    table = pd.DataFrame(
        [
            {
                "Cluster": cluster.cluster_id,
                "Label": cluster.label,
                "Posts": cluster.size,
                "Platforms": ", ".join(
                    f"{name} {count}" for name, count in cluster.platform_counts().items()
                ),
                "First seen (UTC)": cluster.patient_zero().created_at.strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "Patient zero": cluster.patient_zero().author,
            }
            for cluster in visible
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    choices = {
        f"{cluster.cluster_id}. {cluster.label} ({cluster.size} posts)": cluster
        for cluster in visible
    }
    if not choices:
        return
    selected_label = st.selectbox("Inspect a narrative", list(choices))
    cluster = choices[selected_label]
    origin = cluster.patient_zero()

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Where this narrative started**")
        st.write(
            f"First account in this sample: {origin.author} on {origin.platform}."
        )
        _post_card(origin)
    with right:
        st.markdown("**How it spread**")
        for step, post in enumerate(cluster.sorted_posts(), start=1):
            hours = (post.created_at - origin.created_at).total_seconds() / 3600.0
            link = f" · [Open post]({post.url})" if post.url else ""
            st.markdown(
                f"{step}. {post.platform} · {post.author} · "
                f"{post.created_at.strftime('%Y-%m-%d %H:%M UTC')} "
                f"({hours:.1f} h after the first post){link}"
            )

    detail = pd.DataFrame(
        [
            {
                "Time (UTC)": post.created_at.strftime("%Y-%m-%d %H:%M"),
                "Author": post.author,
                "Text": post.full_text().strip() or "This post has no text.",
                "Views": post.views or 0,
                "Likes": getattr(post, "likes", 0) or 0,
                "Retweets": getattr(post, "retweets", 0) or 0,
                "Replies": getattr(post, "replies", 0) or 0,
                "Photo": (getattr(post, "photos", []) or [None])[0],
                "URL": post.url,
            }
            for post in cluster.sorted_posts()
        ]
    )
    path = px.scatter(
        detail,
        x="Time (UTC)",
        y="Author",
        hover_data=["Views", "Likes", "Retweets", "Replies"],
    )
    path.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    st.plotly_chart(path, use_container_width=True)
    st.dataframe(
        detail,
        use_container_width=True,
        hide_index=True,
        column_config={
            "URL": st.column_config.LinkColumn("Post", display_text="Open"),
            "Photo": st.column_config.ImageColumn("Photo"),
        },
    )
    st.markdown("**Posts in this narrative**")
    for post in cluster.sorted_posts():
        with st.container(border=True):
            _post_card(post)


def _posts_section(posts: list[Post]) -> None:
    st.subheader("Posts in the date range")
    st.caption("Text, photos, videos, date, engagement, and a link for each post.")
    for post in posts:
        with st.container(border=True):
            _post_card(post)


def _post_card(post: Post) -> None:
    """Show the canonical link, engagement counts, and the first media preview."""
    st.write(f"{post.platform} · {post.author}")
    st.write(post.created_at.strftime("%Y-%m-%d %H:%M UTC"))
    body = post.full_text().strip()
    st.write(body if body else "This post has no text.")
    photos = list(getattr(post, "photos", []) or [])
    videos = list(getattr(post, "videos", []) or [])
    views = post.views or 0
    likes = getattr(post, "likes", 0) or 0
    retweets = getattr(post, "retweets", 0) or 0
    replies = getattr(post, "replies", 0) or 0
    badges = st.columns(4)
    badges[0].metric("Views", views)
    badges[1].metric("Likes", likes)
    badges[2].metric("Retweets", retweets)
    badges[3].metric("Replies", replies)
    if post.url:
        st.markdown(f"[Open post]({post.url})")
    if photos:
        st.image(photos[0], width=280)
    if videos:
        st.caption("Video")
        st.video(videos[0])


def _account_groups(alerts: list[dict], repeats: list[dict]) -> None:
    st.markdown(
        "<div class='cm-panel'><h3>Groups of accounts that appear to act together</h3>"
        "<p class='cm-evidence'>A group is listed when the collected posts show one of these patterns: "
        "the same or near-identical text, posting within minutes of each other, "
        "accounts created around the same time, reused profile photos, or "
        "the same accounts amplifying each other. A pattern is not evidence of a shared operator.</p></div>",
        unsafe_allow_html=True,
    )
    if repeats:
        names = ", ".join(f"{item['author']} ({item['alert_count']} groups)" for item in repeats)
        st.caption("Accounts that appear in more than one group: " + names)
    if not alerts:
        st.info("No account group met the current window and similarity settings.")
        return
    for alert in alerts:
        with st.container(border=True):
            reason_html = "".join(
                f"<span class='cm-badge cm-DISPUTED'>{_html(reason)}</span> "
                for reason in (alert.get("reasons") or ["pattern in the collected posts"])
            )
            st.markdown(
                f"<div class='cm-card-rule cm-rule-DISPUTED'></div>{reason_html}",
                unsafe_allow_html=True,
            )
            st.markdown(f"**{alert['alert_id']} · {alert['summary']}**")
            st.write("Accounts: " + ", ".join(alert.get("authors") or []))
            st.caption(
                f"{alert['time_start'][:16].replace('T', ' ')} UTC to "
                f"{alert['time_end'][:16].replace('T', ' ')} UTC"
            )
            evidence = alert.get("evidence") or {}
            if evidence.get("shared_url"):
                st.markdown(f"Shared link: [{evidence['shared_url']}]({evidence['shared_url']})")
            if evidence.get("profile_image"):
                st.caption("Shared profile photo")
                st.image(evidence["profile_image"], width=64)
            for post in alert.get("posts") or []:
                link = f" [Open post]({post['url']})" if post.get("url") else ""
                st.markdown(
                    f"- {post.get('platform', '')} · {post.get('author', '')} · "
                    f"{str(post.get('created_at', ''))[:16].replace('T', ' ')} UTC{link}"
                )
            st.caption(alert.get("limitation") or "")


def _alerts_for_cluster(cluster: NarrativeCluster, alerts: list[dict]) -> list[dict]:
    post_ids = {post.post_id for post in cluster.posts}
    authors = {post.author.strip().lower() for post in cluster.posts}
    matched = []
    for alert in alerts:
        alert_ids = set(alert.get("post_ids") or [])
        alert_authors = {author.strip().lower() for author in alert.get("authors") or []}
        if alert_ids & post_ids or alert_authors & authors:
            matched.append(alert)
    return matched


def _alert_section(alerts: list[dict], repeats: list[dict]) -> None:
    st.subheader("Groups of accounts that appear to act together")
    st.caption(
        "Each group lists the reason it was formed. A matching pattern in public "
        "posts is not evidence of a shared operator."
    )
    if repeats:
        names = ", ".join(
            f"{item['author']} ({item['alert_count']} alerts)" for item in repeats
        )
        st.write(f"Accounts that appear in more than one alert: {names}")
    if not alerts:
        st.info("No account group met the current window and similarity settings.")
        return
    for alert in alerts:
        title = f"{alert['alert_id']} · {alert['severity']} · {alert['summary']}"
        with st.expander(title, expanded=alert["severity"] == "high"):
            for reason in alert.get("reasons") or []:
                st.markdown(f"`{reason}`")
            st.write(
                f"Score {alert['score']} · "
                f"{alert['time_start'][:16].replace('T', ' ')} UTC to "
                f"{alert['time_end'][:16].replace('T', ' ')} UTC"
            )
            st.write("Accounts: " + ", ".join(alert["authors"]))
            evidence = alert.get("evidence", {})
            if evidence.get("shared_url"):
                st.write("Shared link: " + evidence["shared_url"])
            st.dataframe(
                pd.DataFrame(alert["posts"]),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(alert["limitation"])


def _export_section(payload: dict) -> None:
    st.subheader("Export")
    json_data = payload_to_json(payload)
    posts_data = posts_csv(payload)
    clusters_data = clusters_csv(payload)
    alerts_data = alerts_csv(payload)
    columns = st.columns(5)
    columns[0].download_button(
        "Download JSON",
        data=json_data,
        file_name="socmint_results.json",
        mime="application/json",
        use_container_width=True,
    )
    columns[1].download_button(
        "Download posts CSV",
        data=posts_data,
        file_name="socmint_posts.csv",
        mime="text/csv",
        use_container_width=True,
    )
    columns[2].download_button(
        "Download clusters CSV",
        data=clusters_data,
        file_name="socmint_clusters.csv",
        mime="text/csv",
        use_container_width=True,
    )
    columns[3].download_button(
        "Download alerts CSV",
        data=alerts_data,
        file_name="socmint_alerts.csv",
        mime="text/csv",
        use_container_width=True,
    )
    columns[4].download_button(
        "Download HTML report",
        data=payload_to_html(payload),
        file_name="socmint_report.html",
        mime="text/html",
        use_container_width=True,
    )
    if st.button("Save copies into results/"):
        paths = save_run(payload)
        st.success("Saved " + ", ".join(paths.values()))


def _method_notes() -> None:
    with st.expander("How the heuristics work"):
        st.markdown(
            """
**Collection.** Twitter reads a public Nitter search page. Telegram reads public
`t.me` channel previews. Demo mode builds labeled synthetic posts on the selected platforms. Each row
has text, photos, videos, a timestamp, engagement, a canonical link, and, when
the source provides them, an account-creation time and a profile image.

**Narratives.** Post text is converted to TF-IDF vectors (words and word pairs).
DBSCAN groups posts inside the sidebar cosine-distance radius. Empty text stays
in the table and is left unclustered with its neighbors instead of stopping the
run. The earliest post in a group is where that wording started *inside this
sample*. The ordered posts show how it moved across accounts and platforms.

**Account groups.** A group can be formed because accounts posted the same or
near-identical text, posted within minutes of each other, were created around
the same time, reused a profile photo, amplified one another, or posted the
same outbound link. These are patterns in the collected posts. They are not a
finding of coordinated inauthentic behavior.

**What this cannot say.** The tool does not score whether a claim is true.
Copied headlines and ordinary reactions to the same event produce the same
signals. Posts outside the search results are missed.
            """
        )


def _configured_secret(name: str) -> str:
    """Prefer a Streamlit secret, then the process environment or `.env`."""
    try:
        value = st.secrets[name]
    except Exception:
        value = ""
    text = str(value or "").strip()
    return text or env(name).strip()


def _query_key(*parts: object) -> str:
    return "|".join(str(part) for part in parts)


if __name__ == "__main__":
    main()