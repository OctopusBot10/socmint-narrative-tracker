# SOCMINT Narrative Tracker

One module for a topic, a date range, and two public platforms: Telegram and X. It collects posts, groups what is being said, flags accounts that move together, and shows where each narrative started inside the sample. Results are on a Streamlit dashboard and can be downloaded as JSON, CSV, or HTML.

The status badges rank how check-worthy the wording is. They are not a fact-check. An account group is a pattern in the collected posts, not proof of a shared operator.

## Example input

| Field | Demo value |
| --- | --- |
| Topic | `Acme Bank` |
| Date range | the sidebar dates, default the last 14 days |
| Platforms | Twitter (X) and Telegram |

Leave **Demo data** on and press **Run collection**. The sample includes a withdrawal rumor that starts on X and is repeated on Telegram, plus a near-copy account group.

## What the run shows

- **Posts.** Text, photo, video, time, views, likes, reposts, replies, and a link. Open **Posts and volume over time**.
- **Narratives.** Cards group similar wording. The line under each summary names the first account and the platforms in that group.
- **Volume.** The same expander charts posts over time and splits them by platform.
- **Account groups.** Each card lists why the group was formed: the same or near-identical text, posting within minutes, accounts created around the same time, a reused profile photo, or the same accounts amplifying each other.
- **Origin and spread.** Open **Details**, then **Correlation**, for the first post and the later posts on each platform.
- **Export.** JSON, posts CSV, clusters CSV, alerts CSV, and an HTML report. **Save copies into results/** writes the same files locally.

## Setup

Python 3.10 or newer.

```bash
cd /Users/kshitijnegi/Desktop/socmint-detection-tool
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Streamlit opens a local page, usually `http://localhost:8501`. This project has also been run at `http://127.0.0.1:8503`. Stop it with `Ctrl+C`.

No API key is required for the demo or for public collection. `CLAIMBUSTER_API_KEY` is optional. `NITTER_INSTANCES` is optional.

### Live Telegram

Turn demo mode off and enter public channel handles, one per line. The collector reads `https://t.me/s/{channel}` only. It does not log in. At most four preview pages are requested, with a random wait of 2–4 seconds between pages and channels. A second HTTP 429 stops that channel. New message ids append to a Google Sheet when `GOOGLE_SHEETS_ID` is set, otherwise to `results/telegram_public.csv`.

```bash
python collector_telegram.py channel_username
```

### Live X

Turn demo mode off. The collector requests a public Nitter search page for the quoted keyword and `since:` / `until:` dates. It does not log in to X and does not use a paid search API. If one instance fails, it tries the next base in `NITTER_INSTANCES`.

## Methodology

### Why these two platforms

Telegram channel previews and Nitter search pages are public HTML. TikTok, Instagram, Snapchat, and Facebook need an account for a useful search, so they are out of scope. YouTube has a public search page, but this demo stays on Telegram and X so the walkthrough matches the two collectors that are wired into the dashboard.

### Collection

Each post is one record: `id`, `platform`, `author`, `text`, `timestamp`, `url`, `photos`, `videos`, and `engagement` (`likes`, `views`, `shares`, and reposts or replies when the page shows them). Empty text is kept. A timestamp that cannot be parsed does not stop the run.

A multi-word topic is quoted (`"Acme Bank"`). Dates are appended as `since:YYYY-MM-DD until:YYYY-MM-DD`. X treats `until:` as exclusive, so posts on the end date can fall outside the search even when the sidebar includes that day.

### Narratives

Post text is converted to TF-IDF vectors (words and word pairs, English stop words). The query keyword is removed so the topic itself does not glue unrelated posts together. DBSCAN groups posts inside the sidebar cosine distance (default 0.62). A post with no neighbor stays unclustered. The earliest post in a group is where that wording started **in this sample**.

Volume uses hours when the sample is two days or shorter, and days otherwise. Counts are split by platform.

### Coordination

These are the rules behind the reason labels:

- Same or near-identical text: different authors, at least 40 characters, similarity at or above the sidebar threshold (default 0.86), inside the time window (default 6 hours).
- Posting within minutes: any pair in that group is 15 minutes apart or less.
- Accounts created around the same time: any pair of account-creation times is within 48 hours.
- Reused profile photos: the same non-default profile image.
- The same accounts amplifying each other: a reply, repost, or quote among the accounts in the group.

A copied headline or many people reacting to the same event produces the same signals. The limitation text on each group says that.

### Check-worthy wording

If `CLAIMBUSTER_API_KEY` is set, each cluster summary is sent once to `https://idir.uta.edu/api/v2/score/text/` and the 0–1 score is shown as a percent. With no key, a local wording heuristic is used and labeled `local`. At 70% or above the card says FALSE, from 40% to 69% UNVERIFIED, and under 40% VERIFIED. Those words describe the wording, not a verdict.

## Tools

Python, Streamlit, pandas, scikit-learn, plotly, requests, and BeautifulSoup. ClaimBuster is optional. Google Sheets is optional for the Telegram archive. No X or Telegram account is used.

## How AI was used

Cursor drafted the collectors, the clustering pipeline, and the dashboard. The output was checked by running the demo on `Acme Bank`, by parsing saved Telegram and Nitter HTML fixtures, and by clicking through the dashboard: collection, narrative cards, account-group reasons, the Correlation tab, and the export buttons. Generated field mappings were compared with the public HTML, not taken as correct on their own.

## Rate limits, blocking, and restrictions

- Telegram: at most 20 channels, four preview pages each, 2–4 seconds between requests, stop after a second HTTP 429. No proxy and no extra accounts. HTTP 403 or an empty preview is shown as an error.
- X: one public HTML search per Nitter base, 25 second timeout, then the next base. No cookies and no login. A dead instance is skipped. The search is whatever that public page returns, which can be a subset of X.
- Do not shorten the waits to collect faster.

## Accuracy

There is no labeled benchmark in this repo. Treat precision and recall as unknown until a person reviews a sample.

False splits happen when people paraphrase, use another language, or put the claim in an image. False merges happen when two stories share rare words or the distance slider is high.

Coordination false positives include a newswire sentence pasted by two accounts, a copied call to action, and a popular link during a real event. The demo market-style denial is outside the rumor cluster on purpose. Coordination false negatives include a rewrite below the similarity cutoff, a caption-less video, a burst slower than the window, and posts the public page never returned. The earliest post can then be later than the real origin.

## Limitations

- Two platforms, and only the public pages those platforms expose.
- English stop words. No image, video, or audio understanding.
- Nitter instances go offline. A live X search can return nothing on a day when the demo still works.
- Demo posts are synthetic. They show the method. They are not a finding about a real bank.
- Usernames are not resolved to people. Do not publish an account group as an accusation.
