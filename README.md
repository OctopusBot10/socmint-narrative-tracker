# SOCMINT Narrative Tracker

Prototype for a social-media narrative review. It collects public Reddit posts and YouTube videos for a keyword and date range, groups similar wording into narrative clusters, marks the earliest post in each cluster, and flags short-window copy-and-link patterns. Results export to JSON and CSV. The interface is a Streamlit dashboard.

This tool groups and times public claims. It does not decide whether a claim is true or false, and a coordination alert is not evidence that accounts share an operator.

## What you get

| File | Role |
| --- | --- |
| `app.py` | Streamlit dashboard |
| `scrapers.py` | Reddit (public JSON or PRAW) and YouTube Data API |
| `analysis.py` | TF-IDF clustering, volume table, patient-zero ordering |
| `coordination.py` | Near-copy and shared-link alerts |
| `demo_data.py` | Synthetic posts so the app runs with no API keys |
| `export_results.py` | JSON and CSV export |
| `models.py` | Shared post and cluster records |
| `config.py` | `.env` loading and request limits |
| `requirements.txt` | Python packages |

Demo posts are labeled `is_demo: true` and use `example.com` links. They are not real people or real threads.

## Setup

Use Python 3.10 or newer.

```bash
cd /Users/kshitijnegi/Desktop/socmint-detection-tool
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Streamlit opens a local page, usually at `http://localhost:8501`. Leave **Demo data** on, keep the keyword `Acme Bank`, and press **Run collection**.

To stop the app, return to the terminal and press `Ctrl+C`.

### Live Reddit

Leave `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` empty. The app calls the public search endpoint:

`https://www.reddit.com/search.json?q=...&sort=new&type=link`

Each request sends a descriptive `User-Agent` and waits about 1.1 seconds between pages. At most four pages are requested per run. Adult-rated posts are skipped. Dates are filtered locally, because Reddit's `t=` parameter only supports coarse windows such as week or month.

Optional PRAW path: create a "script" app at [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps), put the id and secret in `.env`, and install from `requirements.txt` (PRAW is already listed). If both variables are set, PRAW is used instead of public JSON.

Reddit sometimes answers non-browser clients with HTTP 403 or a non-JSON block page. The dashboard shows that error. Use demo mode, try again later, or switch to PRAW credentials. Do not add proxies or extra accounts to get around a block.

### Live YouTube

1. In Google Cloud Console, create a project and enable **YouTube Data API v3**.
2. Create an API key.
3. Put it in `.env` as `YOUTUBE_API_KEY=...`, or paste it into the sidebar when demo mode is off. The sidebar field is masked. The key is not written into exports.

The collector calls `search.list` (`type=video`, `order=date`, `publishedAfter`, `publishedBefore`) and then `videos.list` for view, like, and comment counts. One search is capped at 50 videos so a single click stays near the 100-unit search cost. The default daily quota on a new Google project is 10,000 units. A run that searches once and then requests statistics is usually about 100 units plus a small statistics cost. Quota exhaustion is reported as an error, not retried in a loop.

There is no YouTube HTML scraper and no unofficial InnerTube client in this project.

## How to read a demo run

The generator plants patterns inside whatever date range you pick:

- A withdrawal rumor starts as a Reddit post worded as a branch conversation. Several later Reddit accounts, then two YouTube descriptions, repeat a near-copy of a "freeze withdrawals" message inside a few hours.
- Fee complaints and mobile-app login reports are spread across the range with related but not copied wording.
- Three accounts share one `example.com` link while using different sentences.
- Two accounts post the same market-wire sentence a few minutes apart. That alert is a realistic false positive: copied news text, not a hidden network.

The earliest post in a cluster is labeled patient zero **for the collected sample**. In the rumor cluster that should be the earlier branch post, with YouTube appearing later on the propagation list.

## Methodology

### Collection

Each item is stored as one record: platform, author, community (subreddit or channel), title, text, URL, UTC timestamp, media links, outbound URLs, and engagement. Reddit engagement is score and comment count. Reddit does not provide view counts, so `views` is empty. YouTube engagement is like count, view count, and comment count.

Live search only returns what the platform search ranking returns. It is not a full archive of every public post containing the keyword.

### Narrative grouping

1. Combine title and body.
2. Remove the query keyword, URLs, and punctuation.
3. Build a TF-IDF matrix of unigrams and bigrams, with English stop-words and sublinear term frequency. Terms that appear in more than 85% of posts are dropped.
4. Convert vectors to cosine distance and cluster with average-linkage agglomerative clustering. The sidebar distance is the merge cutoff. A higher cutoff produces fewer, broader themes.
5. Label a cluster with its highest-weight distinctive terms.

TF-IDF is the default because it installs with scikit-learn and runs offline. Sentence-transformer embeddings are a reasonable upgrade if you later need paraphrases to land in one cluster. They are not required here, and they pull in a large model download. A swap-in would replace the TF-IDF matrix with embedding cosine distance and keep the same clustering and dashboard code.

Volume charts bucket by day when the sample spans more than two days, and by hour otherwise. Counts are split by platform and by cluster label.

### Patient zero and propagation

For each cluster, patient zero is the post with the minimum timestamp. Propagation is the first post on each platform, in time order, plus the full post list. "Hours after the earliest post" is arithmetic on those timestamps.

This is the earliest item **in the sample**, not the origin of a claim on the internet. Search ranking, deleted posts, other platforms, images, and different wording can all hide an earlier source.

### Coordination indicators

Text is lowercased, URLs and punctuation are removed, and whitespace is collapsed.

**Near-duplicate burst.** Different authors, each with at least 40 characters of normalized text, are linked when either the normalized text is identical or `difflib.SequenceMatcher` ratio is at least the sidebar threshold (default 0.86). A TF-IDF cosine of at least 0.90 is also accepted. Pairs must fall inside the time window (default 6 hours). Linked posts form a group. Groups whose first and last post exceed the window are discarded, so a slow chain of loose matches does not become one alert. A group that includes both Reddit and YouTube is labeled `cross_platform_near_duplicate`. Otherwise it is `near_duplicate_burst`.

**Shared-link burst.** The same outbound URL (ignoring query strings) posted by at least three different authors inside the window. The post's own Reddit or YouTube permalink is not an outbound URL. Severity is capped at medium because a popular article link is a weak signal.

**Repeat presence.** A username that occurs in two or more alerts is listed. Usernames are not resolved to real identities. `[deleted]`, `[removed]`, and AutoModerator are ignored so anonymous rows are not treated as one account or as a conspirator.

Severity uses the author count, how close the posts are inside the window, text similarity, and whether both platforms appear. High means four or more authors and similarity of at least 0.80. The score is a rank inside this tool, not a probability.

## Rate limits and polite use

- Reddit: about one request per second, at most four pages, and at most one retry sequence after HTTP 429. The app honors a short `Retry-After` and then stops with an error. It does not rotate identities, proxies, or user agents.
- YouTube: at most one search page (50 videos) per run. Stay inside the Google quota. Cache in the interface is the Streamlit session only; rerunning a live query hits the APIs again.
- Do not lower the sleep interval to "go faster." If a source refuses, stop.

## Accuracy, false positives, and false negatives

No labeled benchmark ships with this prototype. Treat precision and recall as unknown until a human reviews a sample against a written definition of "same narrative" and "coordinated."

### Clustering

False splits (one story becomes several clusters) happen when people paraphrase, write in another language, or put the claim in an image or a video while the title stays vague. False merges happen when two stories share distinctive words, or when the distance slider is high. The keyword is stripped to reduce one specific false merge. Labels are weighted terms, not editor-written headlines, so a label can be awkward and still be the right group.

Small samples make IDF unstable. A handful of posts can look more similar than they would inside a large corpus.

### Coordination false positives

- A newswire sentence pasted by two accounts minutes apart. The demo market-wire pair is this case on purpose.
- A quote, a petition text, or a call-to-action that sympathetic people copy.
- "Is the app down for anyone else?" templates during a real outage, if the wording is close enough and the window is wide.
- A popular link during breaking news. Three people sharing one article is the shared-link rule doing what it was built to do, which is often not coordination.
- The same person using two public usernames. The tool cannot see that, and it also cannot prove the opposite.

Widening the time window or lowering the similarity slider increases these hits.

### Coordination false negatives

- The same claim rewritten below the similarity cutoff. This is the main miss, and it is intentional: the detector looks for copies, not for agreement.
- Images, memes, and YouTube speech that is not in the description.
- Coordination slower than the window, or a burst that falls partly outside the selected dates.
- Posts the search endpoint never returned. Patient zero then shifts later, and a real burst can disappear.
- Deleted authors. Those posts can still sit in a narrative cluster, but they do not enter account-level alerts.
- One account reposting many times. That is repeat activity by the same name, and this version does not alert on it, because the requirement is accounts acting together.

### Patient zero errors

The earliest retrieved post can be a quote of something older, a YouTube reupload, or a late comment thread that search ranked first. Cross-platform "movement" only means which collected platform has the earlier timestamp. It does not show who saw whose post.

## Suggested review path

1. Run demo mode on `Acme Bank`.
2. In the volume chart, confirm posts exist on both platforms.
3. Open the withdrawal cluster and check that the earliest post is before the YouTube rows.
4. Open the high cross-platform alert and the shared-link alert. Read the limitation text out loud if you are presenting.
5. Point at the market-wire alert and describe it as an expected false positive.
6. Export JSON and the three CSVs, or save them into `results/`.
7. Turn demo mode off only if keys and quota are ready. A live run can legitimately return zero posts.

## Ethics and scope

Use this on public posts for a clearly described topic, keep exported files private when they contain real usernames, and do not publish an alert as an accusation. The prototype does not access private accounts, messages, or credentials, and it does not try to identify the person behind a username.

## Limitations to mention in a write-up

- Two platforms, keyword search, English stop-word list.
- No image, video, or audio understanding.
- No claim verification and no bot score from a platform.
- Reddit public JSON is unofficial in the sense that it can change or block clients without notice. PRAW is the stable alternative when you have API credentials.
- YouTube results depend on quota, API key restrictions, and `safeSearch=moderate`.
- Demo data will always "work." It is for the interface and the method, not for a finding about any real institution.
