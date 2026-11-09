"""Streamlit dashboard for the SOCMINT narrative tracker."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import cluster_posts, volume_frame
from config import env
from coordination import detect_coordination, repeat_authors
from demo_data import generate_demo_posts
from export_results import (
    alerts_csv,
    build_payload,
    clusters_csv,
    payload_to_json,
    posts_csv,
    save_run,
)
from models import NarrativeCluster, Post
from scrapers import CollectorError, MissingApiKeyError, collect_posts

st.set_page_config(
    page_title="SOCMINT Narrative Tracker",
    page_icon="S",
    layout="wide",
)


def main() -> None:
    st.title("SOCMINT Narrative Tracker")
    st.caption(
        "Prototype for collecting public Reddit and YouTube posts, grouping "
        "similar wording into narratives, and flagging short-window copy or "
        "link patterns. Clusters are not fact-checks. Alerts are not proof of "
        "coordination."
    )

    (
        keyword,
        start,
        end,
        use_demo,
        platforms,
        limit,
        distance,
        window,
        text_threshold,
        youtube_key,
        reddit_client_id,
        reddit_client_secret,
        reddit_user_agent,
    ) = _sidebar()

    if st.button("Run collection", type="primary", use_container_width=True):
        with st.spinner("Collecting posts and running the heuristics..."):
            try:
                result = _run(
                    keyword=keyword,
                    start=start,
                    end=end,
                    use_demo=use_demo,
                    platforms=platforms,
                    limit=limit,
                    distance=distance,
                    window=window,
                    text_threshold=text_threshold,
                    youtube_key=youtube_key,
                    reddit_client_id=reddit_client_id,
                    reddit_client_secret=reddit_client_secret,
                    reddit_user_agent=reddit_user_agent,
                )
            except MissingApiKeyError as exc:
                st.error(str(exc))
                return
            except CollectorError as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"The run failed before producing results: {exc}")
                return
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if not result:
        st.info(
            "Demo mode is on, so the first run needs no API keys. Leave the "
            "sample keyword, choose a date range, and press Run collection. "
            "Switch off demo mode when you are ready to query live public posts."
        )
        _method_notes()
        return

    if result["query_key"] != _query_key(
        keyword, start, end, use_demo, platforms, limit, distance, window, text_threshold
    ):
        st.warning(
            "Sidebar settings differ from the results below. Press Run collection "
            "to refresh."
        )

    _render_dashboard(result)
    _method_notes()


def _sidebar() -> tuple:
    today = date.today()
    with st.sidebar:
        st.header("Query")
        keyword = st.text_input("Keyword or topic", value="Acme Bank").strip()
        start_date = st.date_input("Start date", value=today - timedelta(days=14))
        end_date = st.date_input("End date", value=today)
        use_demo = st.toggle(
            "Demo data",
            value=True,
            help="Synthetic posts for this keyword and date range. Live mode is off while this is on.",
        )
        platforms = st.multiselect(
            "Platforms",
            options=["reddit", "youtube"],
            default=["reddit", "youtube"],
            disabled=use_demo,
        )
        limit = st.slider("Max posts per platform", min_value=10, max_value=100, value=40)
        
        # --- API CREDENTIALS SECTION ---
        st.markdown("---")
        st.subheader("🔑 API Credentials")
        youtube_key = st.text_input(
            "YouTube API key",
            value=env("YOUTUBE_API_KEY"),
            type="password",
            help="Required for live YouTube searches.",
        )
        reddit_client_id = st.text_input(
            "Reddit Client ID",
            value=env("REDDIT_CLIENT_ID"),
            type="password",
            help="Optional: PRAW Client ID for authenticated Reddit requests.",
        )
        reddit_client_secret = st.text_input(
            "Reddit Client Secret",
            value=env("REDDIT_CLIENT_SECRET"),
            type="password",
            help="Optional: PRAW Client Secret.",
        )
        reddit_user_agent = st.text_input(
            "Reddit User Agent",
            value=env("REDDIT_USER_AGENT") or "SOCMINT_Analyzer/1.0",
            help="PRAW User Agent string.",
        )

        st.header("Heuristics")
        distance = st.slider(
            "Cluster distance",
            min_value=0.35,
            max_value=0.8,
            value=0.62,
            step=0.01,
            help="Higher values merge posts more readily. Cosine distance, not similarity.",
        )
        window = st.slider("Coordination window (hours)", min_value=1, max_value=48, value=6)
        text_threshold = st.slider(
            "Near-copy similarity",
            min_value=0.7,
            max_value=0.98,
            value=0.86,
            step=0.01,
        )
        if not use_demo:
            st.caption(
                "Reddit uses public search JSON unless Reddit script-app credentials "
                "are set. YouTube needs a Data API key."
            )
            
    start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
    end = datetime.combine(end_date, time.max.replace(microsecond=0), tzinfo=timezone.utc)
    if end < start:
        start, end = end, start
    return (
        keyword,
        start,
        end,
        use_demo,
        platforms,
        limit,
        distance,
        float(window),
        float(text_threshold),
        youtube_key,
        reddit_client_id,
        reddit_client_secret,
        reddit_user_agent,
    )


def _run(
    *,
    keyword: str,
    start: datetime,
    end: datetime,
    use_demo: bool,
    platforms: list[str],
    limit: int,
    distance: float,
    window: float,
    text_threshold: float,
    youtube_key: str,
    reddit_client_id: str,
    reddit_client_secret: str,
    reddit_user_agent: str,
) -> dict:
    if not keyword:
        raise CollectorError("Enter a keyword before running.")
    if not use_demo and not platforms:
        raise CollectorError("Select at least one platform, or turn demo data on.")

    if use_demo:
        posts = generate_demo_posts(keyword, start, end)
        mode = "demo"
        selected = ["reddit", "youtube"]
    else:
        posts = collect_posts(
            keyword,
            start,
            end,
            platforms=platforms,
            limit_per_platform=limit,
            youtube_api_key=youtube_key,
            reddit_client_id=reddit_client_id,
            reddit_client_secret=reddit_client_secret,
            reddit_user_agent=reddit_user_agent,
        )
        mode = "live"
        selected = platforms

    clusters, similarity = cluster_posts(
        posts, keyword=keyword, distance_threshold=distance
    )
    alerts = detect_coordination(
        posts,
        clusters,
        similarity,
        window_hours=window,
        text_similarity_threshold=text_threshold,
    )
    repeats = repeat_authors(alerts)
    payload = build_payload(
        keyword=keyword,
        start=start,
        end=end,
        mode=mode,
        platforms=selected,
        posts=posts,
        clusters=clusters,
        alerts=alerts,
        repeats=repeats,
    )
    return {
        "query_key": _query_key(
            keyword, start, end, use_demo, selected, limit, distance, window, text_threshold
        ),
        "posts": posts,
        "clusters": clusters,
        "frame": volume_frame(posts, clusters),
        "payload": payload,
    }


def _render_dashboard(result: dict) -> None:
    payload = result["payload"]
    posts: list[Post] = result["posts"]
    clusters: list[NarrativeCluster] = result["clusters"]
    frame: pd.DataFrame = result["frame"]
    query = payload["query"]
    summary = payload["summary"]
    alerts = payload["alerts"]

    if query["mode"] == "demo":
        st.info(
            "These posts are synthetic. They were generated for the keyword and "
            "dates in the sidebar so the workflow can be reviewed without API keys. "
            "Links point at example.com and are not real threads."
        )

    st.subheader(f"Results for “{query['keyword']}”")
    st.caption(
        f"{query['mode']} collection · {query['start'][:10]} to {query['end'][:10]} UTC"
    )

    metrics = st.columns(5)
    metrics[0].metric("Posts", summary["posts"])
    metrics[1].metric("Narrative clusters", summary["clusters"])
    metrics[2].metric("Multi-post themes", summary["multi_post_clusters"])
    metrics[3].metric("Coordination alerts", summary["alerts"])
    platform_counts = frame["platform"].value_counts().to_dict() if not frame.empty else {}
    metrics[4].metric(
        "Reddit / YouTube",
        f"{platform_counts.get('reddit', 0)} / {platform_counts.get('youtube', 0)}",
    )

    if not posts:
        st.warning(
            "No posts came back for that keyword and date range. Widen the dates, "
            "check the spelling, or switch to demo data."
        )
        return

    _charts(frame)
    _cluster_section(clusters)
    _alert_section(alerts, payload.get("repeat_authors", []))
    _export_section(payload)


def _charts(frame: pd.DataFrame) -> None:
    st.subheader("Volume over time")
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
    st.subheader("Narrative clusters")
    st.caption(
        "Earliest post in each cluster is the patient zero of this sample only. "
        "Posts that were never retrieved, or that used different words, are invisible here."
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
                "Reddit": cluster.platform_counts().get("reddit", 0),
                "YouTube": cluster.platform_counts().get("youtube", 0),
                "First seen (UTC)": cluster.patient_zero().created_at.strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "Patient zero": cluster.patient_zero().author,
                "Started on": cluster.patient_zero().platform,
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
        st.markdown("**Earliest post in this sample**")
        st.write(f"{origin.platform} · {origin.community} · {origin.author}")
        st.write(origin.created_at.strftime("%Y-%m-%d %H:%M UTC"))
        st.write(origin.full_text())
        if origin.url:
            st.markdown(f"[Open post]({origin.url})")
    with right:
        st.markdown("**First time the wording shows up on each platform**")
        for step, post in enumerate(cluster.first_on_each_platform(), start=1):
            hours = (post.created_at - origin.created_at).total_seconds() / 3600.0
            st.write(
                f"{step}. {post.platform} · {post.author} · "
                f"{post.created_at.strftime('%Y-%m-%d %H:%M UTC')} "
                f"({hours:.1f} h after the earliest post)"
            )

    detail = pd.DataFrame(
        [
            {
                "Time (UTC)": post.created_at.strftime("%Y-%m-%d %H:%M"),
                "Platform": post.platform,
                "Author": post.author,
                "Community": post.community,
                "Title": post.title,
                "Ups": post.ups,
                "Views": post.views if post.views is not None else "",
                "Comments": post.comments,
                "URL": post.url,
            }
            for post in cluster.sorted_posts()
        ]
    )
    path = px.scatter(
        detail,
        x="Time (UTC)",
        y="Platform",
        color="Platform",
        hover_data=["Author", "Title", "Ups", "Comments"],
    )
    path.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
    st.plotly_chart(path, use_container_width=True)
    st.dataframe(detail, use_container_width=True, hide_index=True)


def _alert_section(alerts: list[dict], repeats: list[dict]) -> None:
    st.subheader("Coordinated activity alerts")
    st.caption(
        "Each alert is a pattern in public text and timestamps. Read the limitation "
        "on the card before treating it as suspicious."
    )
    if repeats:
        names = ", ".join(
            f"{item['author']} ({item['alert_count']} alerts)" for item in repeats
        )
        st.write(f"Accounts that appear in more than one alert: {names}")
    if not alerts:
        st.success(
            "No near-copy or shared-link pattern met the current window and "
            "similarity settings."
        )
        return
    for alert in alerts:
        title = f"{alert['alert_id']} · {alert['severity']} · {alert['indicator']}"
        with st.expander(title, expanded=alert["severity"] == "high"):
            st.write(alert["summary"])
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
    columns = st.columns(4)
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
    if st.button("Save copies into results/"):
        paths = save_run(payload)
        st.success("Saved " + ", ".join(paths.values()))


def _method_notes() -> None:
    with st.expander("How the heuristics work"):
        st.markdown(
            """
**Collection.** Demo mode builds labeled synthetic posts. Live Reddit search
uses the public JSON endpoint, or PRAW when script-app credentials are present.
Live YouTube search uses Data API v3 (`search.list` plus `videos.list` for
views, likes, and comments).

**Narratives.** Post text is converted to TF-IDF vectors (words and word pairs).
Average-linkage clustering merges posts whose cosine distance is below the
sidebar threshold. The keyword itself is removed first so it does not glue
every post together. The earliest timestamp in a cluster is that cluster's
patient zero *inside the sample*.

**Coordination.** Different accounts whose normalized text is identical, or
whose sequence similarity clears the slider, and who posted inside the time
window, become a near-copy alert. The same alert is marked cross-platform when
both Reddit and YouTube are present. Three or more accounts sharing one
outbound URL inside the window become a shared-link alert.

**What this cannot say.** The tool does not score whether a claim is true. A
copied news headline is a true pattern and a weak accusation. Paraphrases,
images, and posts outside the search results are missed.
            """
        )


def _query_key(*parts: object) -> str:
    return "|".join(str(part) for part in parts)


if __name__ == "__main__":
    main()