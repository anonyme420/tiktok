# 🎬 Multi-Agent Video Pipeline

Generate vertical (1080×1920) short-form story videos automatically — Reddit-style **AITA** stories, AI voiceover, synced subtitles, burned onto your own background footage.

Built with **LangGraph** (multi-agent orchestration) + **DeepSeek** (story writing) + **Kokoro TTS** (local voice) + **FFmpeg** (rendering).

One run produces **multiple videos** (default 3), each from a different story.

---

## How it works

Four agents pass a shared state through a LangGraph graph:

| Agent | Job |
|-------|-----|
| **Supervisor** | Routes state between agents and retries failed stages |
| **Scout** | Asks DeepSeek for a pool of AITA stories, scores them, picks the best |
| **Narrator** | Cleans the text, runs Kokoro TTS locally, builds a synced `.srt` |
| **Editor** | Picks a footage clip and runs FFmpeg: scale → crop → burn subtitles → mix narration |

For a multi-video run, the pipeline generates and ranks **one pool of stories**, then renders the top *N* as separate videos (the Kokoro model loads only once for the whole batch).

---

## 1. Prerequisites

You need three things installed **before** the Python packages:

### Python 3.10+
Check with `python --version`.

### FFmpeg (on your PATH)
- **Windows:** download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or [BtbN builds](https://github.com/BtbN/FFmpeg-Builds/releases), unzip, and add the `bin` folder to your PATH. Verify with `ffmpeg -version`.
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`

### eSpeak-NG (required by Kokoro TTS for phonemization)
- **Windows:** install from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases) (run the `.msi`).
- **macOS:** `brew install espeak-ng`
- **Linux:** `sudo apt-get install espeak-ng`

---

## 2. Install Python dependencies

From the project folder (`tiktok/`):

```powershell
pip install -r requirements.txt
```

> The first run will also **download the Kokoro-82M model (~330 MB)** from Hugging Face automatically. This is a one-time download.

---

## 3. Configure

### DeepSeek API key (required)
Get a key at [platform.deepseek.com](https://platform.deepseek.com). Provide it either way:

**Option A — environment variable (recommended):**
```powershell
$env:DEEPSEEK_API_KEY = "sk-your-key-here"
```

**Option B — edit the file:** set `DEEPSEEK_API_KEY` near the top of [`pipeline.py`](pipeline.py).

### Background footage (required)
Drop one or more `.mp4` clips into the `footage/` folder. The Editor picks one at random per video.

> No clips? The pipeline falls back to a plain black background. For gameplay/“satisfying” footage, grab free clips from [pixabay.com/videos](https://pixabay.com/videos) or record your screen.

### Optional settings (top of `pipeline.py`)
| Setting | Default | What it does |
|---------|---------|--------------|
| `NUM_VIDEOS` | `3` | How many videos to produce per run (also via env var) |
| `KOKORO_VOICE` | `af_heart` | Voice. Others: `af_bella`, `af_nova`, `af_sarah`, `am_adam`, `bf_emma` |
| `FOOTAGE_DIR` | `./footage` | Where background clips live |
| `OUTPUT_DIR` | `./output` | Where results are written |

---

## 4. Run

**Default (3 videos):**
```powershell
python pipeline.py
```

**Choose how many videos:**
```powershell
$env:NUM_VIDEOS = 5; python pipeline.py
```

> On macOS/Linux use `NUM_VIDEOS=5 python pipeline.py` instead.

You’ll see per-video progress and a final summary:

```
Final summary — 3 video(s):
  [OK]   output\video_50e17169.mp4  ←  AITA for not letting my ex-husband visit our dying dog?
  [OK]   output\video_1b9036d9.mp4  ←  ...
  [OK]   output\video_9f2c0a7d.mp4  ←  ...

Done! 3/3 succeeded.
```

---

## 5. Output

Each video produces three files in `output/`, keyed by a random run id:

| File | Description |
|------|-------------|
| `audio_<id>.wav` | Kokoro narration (24 kHz mono) |
| `subtitles_<id>.srt` | Subtitles, timed to the real audio length |
| `video_<id>.mp4` | **Final 1080×1920 video** — narration + burned-in captions |

The `video_<id>.mp4` files are your finished, ready-to-post shorts.

---

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| `ffmpeg: command not found` / FFmpeg errors | FFmpeg isn’t on your PATH — see Prerequisites. |
| Kokoro / phonemizer errors, or robotic/empty audio | eSpeak-NG isn’t installed — see Prerequisites. |
| Video has the footage’s original audio, not the voice | You’re on an old version — the fixed command maps narration explicitly (`-map 0:v:0 -map 1:a:0 -shortest`). |
| API / authentication errors | Check `DEEPSEEK_API_KEY` is set and valid. |
| First run is slow | One-time Kokoro model download (~330 MB) + model load. Subsequent runs are faster. |
| `[FAIL] ...` in the summary | The Editor retries up to 3× then reports failure for that video; the run continues with the others. |

---

## Project layout

```
tiktok/
├── pipeline.py        # the whole pipeline (agents + graph + main)
├── requirements.txt   # Python dependencies
├── footage/           # drop your background .mp4 clips here
│   └── 1.mp4
└── output/            # generated audio, subtitles, and videos
```
