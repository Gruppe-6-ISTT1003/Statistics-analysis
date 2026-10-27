# Statistics-analysis

Statistical analysis, statistical learning, and machine learning project for **ISTT1003 / ISTx1003** (Statistikk, statistisk læring og data science, NTNU).

The project analyzes the **Spotify Top 2500 Most Streamed Tracks** dataset ([`data/kworb_top2500_spotify.csv`](data/kworb_top2500_spotify.csv)), scraped from [Kworb](https://kworb.net/spotify/songs.html), because Spotify does not provide an official top streams list. The list was then enriched with musical features via the old Spotify Web API through [Exportify](https://exportify.net).

---

## Dataset Summary

* **File location:** [`data/kworb_top2500_spotify.csv`](data/kworb_top2500_spotify.csv)
* **Observations (rows):** 2,500 tracks
* **Variables (columns):** 22 attributes
* **Missing values:** All columns are complete except `Genres` (934 missing/blank values where genre tags were unavailable).

---

## Data Dictionary / Variable Reference

The dataset includes track metadata, commercial indicators, and acoustic/musical features calculated by Spotify's audio analysis algorithms.

### 1. Metadata & Identifiers

| Column Name | Data Type | Possible Values / Format | Description | Dataset Example |
| :--- | :--- | :--- | :--- | :--- |
| `Track URI` | String (Categorical / ID) | `spotify:track:<id>` | Spotify's unique resource identifier for the track. | `spotify:track:0sf12qNH5qcw8qpgymFOqD` |
| `Track Name` | String (Text) | Free text | Title of the track / song. | `Blinding Lights` |
| `Album Name` | String (Text) | Free text | Name of the album or EP/single release containing the track. | `After Hours` |
| `Artist Name(s)` | String (Text) | Semicolon-separated text | Primary and featured artist(s) credited on the track (separated by `;` if multiple). | `The Weeknd;Daft Punk` |
| `Release Date` | String (Date / Ordinal) | `YYYY-MM-DD` or `YYYY` | Release date of the album/single. Range in dataset spans from 1957 to 2026. | `2019-11-29` |
| `Record Label` | String (Categorical) | Free text | Record label, distributor, or copyright owner that released the track. | `Republic Records` |

---

### 2. Commercial & Platform Attributes

| Column Name | Data Type | Range / Values | Description | Dataset Example |
| :--- | :--- | :--- | :--- | :--- |
| `Popularity` | Integer (Discrete / Scale) | `0` to `100` | Current popularity index calculated by Spotify's algorithm. Heavily based on total play counts and recency of streams (higher = currently more streamed). | `92` |
| `Explicit` | Boolean (Binary) | `true` / `false` | Indicates whether the track contains explicit language or themes (profanity, mature content). 704 tracks are explicit (~28%), 1796 are clean (~72%). | `false` |
| `Genres` | String (Categorical / List) | Comma-separated text | Genre classifications associated with the artist/track. Note: 934 entries are empty. | `rap,afrobeats,pop` |

---

### 3. Audio Features & Musical Properties

Spotify extracts these digital signal processing (DSP) metrics directly from the audio file using neural networks and acoustic modeling.

| Column Name | Data Type | Range / Unit | Description & Musical Interpretation | Dataset Example |
| :--- | :--- | :--- | :--- | :--- |
| `Duration (ms)` | Integer (Continuous) | `80,927` – `613,026` ms (1.3 to 10.2 min) | Track duration in milliseconds ($1\text{ min} = 60\,000\text{ ms}$). Median duration is $\approx 3.5$ minutes. | `201573` (3m 21s) |
| `Danceability` | Float (Continuous) | `0.0` – `1.0` | Describes how suitable a track is for dancing based on tempo, rhythm stability, beat strength, and overall regularity. `1.0` is most danceable. | `0.512` |
| `Energy` | Float (Continuous) | `0.0` – `1.0` | Perceptual measure of intensity, dynamic range, loudness, and entropy. Fast, loud, and noisy tracks (e.g. death metal, EDM) score close to `1.0`; acoustic ballads score lower. | `0.796` |
| `Key` | Integer (Pitch Class) | `0` – `11` (or `-1` if undetected) | The estimated overall musical key of the track using standard Pitch Class notation: `0 = C`, `1 = C♯/D♭`, `2 = D`, `3 = D♯/E♭`, `4 = E`, `5 = F`, `6 = F♯/G♭`, `7 = G`, `8 = G♯/A♭`, `9 = A`, `10 = A♯/B♭`, `11 = B`. | `1` (C♯ / D♭) |
| `Loudness` | Float (Continuous) | `-46.11` to `-0.51` dB | Overall loudness of the track in decibels (dB), averaged across the entire song. Values are relative to digital full scale (values closer to `0 dB` are louder). | `-4.075` |
| `Mode` | Integer (Binary) | `0` (Minor) or `1` (Major) | Indicates the modality (musical scale) of the track: `1 = Major` (dur), `0 = Minor` (moll). In this dataset, ~60% are major and ~40% are minor. | `1` (Major) |
| `Speechiness` | Float (Continuous) | `0.0` – `1.0` | Detects spoken words: values $> 0.66$ indicate talk-shows/spoken word; $0.33 - 0.66$ describe tracks containing both music and speech (e.g. rap); $< 0.33$ represent typical music and singing. | `0.0645` |
| `Acousticness` | Float (Continuous) | `0.0` – `1.0` | Confidence score for whether the track is acoustic (unplugged/natural acoustic instruments). `1.0` indicates high certainty of acoustic recording. | `0.00147` |
| `Instrumentalness` | Float (Continuous) | `0.0` – `1.0` | Predicts whether a track contains no vocals. Rap and vocal music score near `0.0`; tracks above `0.5` are typically instrumental (classical, lo-fi beats, ambient). | `0.000209` |
| `Liveness` | Float (Continuous) | `0.0` – `1.0` | Detects the presence of an audience in the recording. Values $> 0.8$ provide strong evidence that the song was recorded live (e.g. concert recording). | `0.0938` |
| `Valence` | Float (Continuous) | `0.0` – `1.0` | Musical positiveness/mood conveyed by the track. Tracks with high valence sound happy, cheerful, or euphoric; tracks with low valence sound sad, depressed, or angry. | `0.344` |
| `Tempo` | Float (Continuous) | `48.7` – `214.0` BPM | Overall estimated tempo in beats per minute (BPM). | `171.014` |
| `Time Signature` | Integer (Discrete) | `1` – `5` (meter) | Estimated time signature (beats per measure). Most popular music is in `4` ($4/4$ common time) or `3` ($3/4$ waltz time). | `4` ($4/4$ time) |

---

## Statistical Summary of Audio Features

| Feature | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Duration (min)` | 3.68 | 0.81 | 1.35 | 3.16 | 3.56 | 4.07 | 10.22 |
| `Popularity` | 71.49 | 24.36 | 0.00 | 66.00 | 80.00 | 88.00 | 99.00 |
| `Danceability` | 0.646 | 0.141 | 0.142 | 0.551 | 0.659 | 0.751 | 0.970 |
| `Energy` | 0.656 | 0.163 | 0.003 | 0.548 | 0.671 | 0.778 | 0.989 |
| `Loudness (dB)` | -6.61 | 2.56 | -46.11 | -7.77 | -6.16 | -4.95 | -0.51 |
| `Speechiness` | 0.086 | 0.080 | 0.023 | 0.036 | 0.053 | 0.098 | 0.591 |
| `Acousticness` | 0.217 | 0.237 | 0.000 | 0.033 | 0.138 | 0.334 | 0.995 |
| `Instrumentalness` | 0.024 | 0.108 | 0.000 | 0.000 | 0.000 | 0.001 | 0.995 |
| `Liveness` | 0.170 | 0.126 | 0.021 | 0.095 | 0.119 | 0.208 | 0.890 |
| `Valence` | 0.516 | 0.221 | 0.000 | 0.344 | 0.512 | 0.686 | 0.981 |
| `Tempo (BPM)` | 120.87 | 27.60 | 48.72 | 98.05 | 120.04 | 139.98 | 214.02 |
