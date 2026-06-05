"""
Multi-Agent Video Pipeline — LangGraph + DeepSeek V3 + Kokoro TTS
==================================================================
Agents:
  Supervisor  — routes state, retries on failure
  Scout       — generates Reddit-style AITA stories & scores them
  Narrator    — cleans text, runs Kokoro TTS locally, builds .srt
  Editor      — picks a footage clip, runs FFmpeg to render final MP4

Setup (one-time):
  # Linux
  apt-get install espeak-ng ffmpeg
  pip install langgraph langchain-openai kokoro>=0.9.4 soundfile numpy torch

  # macOS
  brew install espeak ffmpeg
  pip install langgraph langchain-openai kokoro>=0.9.4 soundfile numpy torch

  # Windows
  # Install eSpeak-NG from https://github.com/espeak-ng/espeak-ng/releases
  pip install langgraph langchain-openai kokoro>=0.9.4 soundfile numpy torch

Keys — edit the two lines below:
  DEEPSEEK_API_KEY  →  your DeepSeek key (https://platform.deepseek.com)

Run:
  python pipeline.py
"""

import os, re, uuid, json, random, subprocess, textwrap, sys
from pathlib import Path
from typing import TypedDict, Literal, Optional

# Fix Unicode output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ── CONFIG — SET YOUR KEY HERE (or export DEEPSEEK_API_KEY) ───
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-15c3ea9616d244398f5d783538be280e")

KOKORO_VOICE     = "af_heart"   # Grade-A female voice (warm, natural)
# Other great voices: af_bella, af_nova, af_sarah, am_adam, bf_emma
# Full list: https://huggingface.co/hexgrad/Kokoro-82M

# How many videos to produce in one run. Scout generates a pool of stories,
# ranks them, and the top NUM_VIDEOS become separate videos.
NUM_VIDEOS       = int(os.getenv("NUM_VIDEOS", "3"))

FOOTAGE_DIR      = Path("./footage")   # drop .mp4 files here
OUTPUT_DIR       = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── LLM (DeepSeek V4 Pro via OpenAI-compatible API) ──────────
llm = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
    temperature=0.7,
)

# ── PIPELINE STATE ────────────────────────────────────────────
class PipelineState(TypedDict):
    run_id:  str
    story:   Optional[dict]   # {title, body, score, upvotes}
    audio:   Optional[str]    # path to .wav
    srt:     Optional[str]    # path to .srt
    video:   Optional[str]    # path to .mp4
    error:   Optional[str]
    retries: int
    next:    str

# ── HELPERS ───────────────────────────────────────────────────
def log(agent: str, msg: str):
    print(f"[{agent.upper():<10}] {msg}")

def llm_call(system: str, user: str) -> str:
    from langchain_core.messages import SystemMessage, HumanMessage
    resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return resp.content.strip()

_KOKORO = None
def _get_kokoro():
    """Load the Kokoro TTS pipeline once and reuse it — the model is slow to
    initialize, so we don't want to reload it for every video in a batch."""
    global _KOKORO
    if _KOKORO is None:
        from kokoro import KPipeline
        _KOKORO = KPipeline(lang_code="a", device="cpu")   # 'a' = American English
    return _KOKORO

# ── AGENT: SCOUT ──────────────────────────────────────────────
def brainstorm_stories(n_candidates: int = 5) -> list[dict]:
    """Generate `n_candidates` AITA stories, score them, and return the full
    list sorted best-first (each story gets a "score" field). Shared by the
    single-video scout agent and the multi-video batch selection in main()."""
    n_candidates = max(n_candidates, 1)
    log("scout", f"Asking DeepSeek to generate {n_candidates} Reddit-style AITA stories…")

    raw = llm_call(
        system=textwrap.dedent(f"""\
            You are a creative writer who invents realistic Reddit AITA stories.
            Respond ONLY with valid JSON — no markdown, no commentary, no code fences.
            Return a JSON array of {n_candidates} objects, each with:
              "title":   string  (post title, starts with AITA or WIBTA)
              "body":    string  (2-4 paragraphs, first-person, ~200 words)
              "upvotes": integer (between 5000 and 50000)
        """),
        user=f"Generate {n_candidates} original AITA stories with distinct, "
             f"emotionally compelling moral dilemmas. Make them varied — no two alike."
    )
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        candidates = json.loads(raw)
        assert isinstance(candidates, list) and candidates
    except (json.JSONDecodeError, AssertionError) as e:
        log("scout", f"JSON parse error: {e} — using fallback story")
        candidates = [{
            "title": "AITA for keeping my lottery winnings secret from my family?",
            "body": ("I won $50,000 in the lottery six months ago and told no one. "
                     "My family has a history of asking for money and never paying it back. "
                     "Last week my brother found out and called me selfish. My parents are upset too. "
                     "I just wanted to feel financially secure for once. AITA?"),
            "upvotes": 23451
        }]

    log("scout", f"Scoring {len(candidates)} candidates for video potential…")
    scores_raw = llm_call(
        system=textwrap.dedent("""\
            You are a content strategist scoring Reddit stories for short-form video.
            Respond ONLY with valid JSON — no markdown, no code fences.
            Return a JSON array of objects: {"index": 0, "score": 8}
            Score 1-10 on: emotional hook, clear conflict, relatable dilemma, satisfying arc.
        """),
        user=f"Score these:\n{json.dumps([{'index': i, 'title': c['title']} for i, c in enumerate(candidates)])}"
    )
    scores_raw = re.sub(r"```json|```", "", scores_raw).strip()
    try:
        # Keep only indices that actually point at a candidate, so a stray
        # index from the model can't crash the lookup below.
        scores = {s["index"]: s["score"] for s in json.loads(scores_raw)
                  if 0 <= s.get("index", -1) < len(candidates)}
    except Exception:
        scores = {}

    for i, c in enumerate(candidates):
        c["score"] = scores.get(i, random.randint(6, 9))
    return sorted(candidates, key=lambda c: c["score"], reverse=True)


def scout_agent(state: PipelineState) -> PipelineState:
    best = brainstorm_stories(5)[0]
    log("scout", f"[OK] Selected: \"{best['title']}\" (score {best['score']}/10, {best['upvotes']:,} upvotes)")
    return {**state, "story": best, "next": "narrator"}

# ── AGENT: NARRATOR ───────────────────────────────────────────
def narrator_agent(state: PipelineState) -> PipelineState:
    story  = state["story"]
    run_id = state["run_id"]

    # 1. Clean text with DeepSeek
    log("narrator", "Cleaning story text for TTS narration…")
    clean_text = llm_call(
        system=("Clean Reddit post text for text-to-speech narration. "
                "Remove markdown, emojis, usernames, subreddit references. "
                "Fix punctuation and spelling. Return plain text only — no commentary."),
        user=f"{story['title']}\n\n{story['body']}"
    )
    log("narrator", f"Clean text ready: {len(clean_text)} chars")

    # 2. Kokoro TTS — runs 100% locally, no API key needed
    audio_path = OUTPUT_DIR / f"audio_{run_id}.wav"
    audio_duration = None   # real spoken length, used to sync subtitles below
    log("narrator", f"Running Kokoro TTS locally (voice: {KOKORO_VOICE})…")
    try:
        import soundfile as sf
        import numpy as np

        tts = _get_kokoro()   # 'a' = American English; loaded once and reused

        # Kokoro returns a generator of (graphemes, phonemes, audio_array) chunks
        chunks = []
        for _, _, audio in tts(clean_text, voice=KOKORO_VOICE, speed=1.0, split_pattern=r"\n+"):
            chunks.append(audio)

        if chunks:
            full_audio = np.concatenate(chunks)
            sf.write(str(audio_path), full_audio, 24000)
            duration_s = len(full_audio) / 24000
            audio_duration = duration_s
            log("narrator", f"[OK] Audio: {duration_s/60:.1f}m {duration_s%60:.0f}s -> {audio_path}")
        else:
            log("narrator", "⚠ Kokoro returned no audio — writing empty file")
            audio_path.write_bytes(b"")

    except ImportError:
        log("narrator", "⚠ Kokoro not installed — run: pip install kokoro soundfile torch")
        log("narrator", "  Writing silent placeholder and continuing…")
        audio_path.write_bytes(b"")
    except Exception as e:
        log("narrator", f"⚠ Kokoro error: {e}")
        audio_path.write_bytes(b"")

    # 3. Build .srt synced to the ACTUAL audio length.
    # Each caption's on-screen time is proportional to its word count, and the
    # whole thing is stretched to fit the real spoken duration — so subtitles
    # can't drift past the audio. If TTS failed, fall back to a words-per-second
    # estimate (~2.5 wps).
    log("narrator", "Building word-synced SRT subtitles…")
    srt_path   = OUTPUT_DIR / f"subtitles_{run_id}.srt"
    words      = clean_text.split()
    chunk_size = 8
    chunks_txt = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    total_words = max(len(words), 1)
    if audio_duration and audio_duration > 0:
        per_word = audio_duration / total_words
    else:
        per_word = 1 / 2.5   # fallback estimate when no audio was produced

    def srt_time(s: float) -> str:
        h, r   = divmod(int(s), 3600)
        m, sec = divmod(r, 60)
        ms     = int((s - int(s)) * 1000)
        return f"{h:02}:{m:02}:{sec:02},{ms:03}"

    lines  = []
    cursor = 0.0
    for idx, chunk in enumerate(chunks_txt):
        start = cursor
        end   = cursor + max(len(chunk.split()), 1) * per_word
        cursor = end
        lines.append(f"{idx+1}\n{srt_time(start)} --> {srt_time(end)}\n{chunk}\n")
    srt_path.write_text("\n".join(lines), encoding="utf-8")
    log("narrator", f"[OK] SRT: {len(chunks_txt)} segments -> {srt_path}")

    return {**state, "audio": str(audio_path), "srt": str(srt_path), "next": "editor"}

# ── AGENT: EDITOR ─────────────────────────────────────────────
def editor_agent(state: PipelineState) -> PipelineState:
    run_id     = state["run_id"]
    audio_path = state["audio"]
    srt_path   = state["srt"]
    video_out  = OUTPUT_DIR / f"video_{run_id}.mp4"

    # Pick or generate footage
    clips = list(FOOTAGE_DIR.glob("*.mp4")) if FOOTAGE_DIR.exists() else []
    if clips:
        clip = random.choice(clips)
        log("editor", f"Using footage: {clip.name}")
    else:
        log("editor", "No .mp4 clips found in ./footage/ — using black placeholder")
        log("editor", "  -> For Minecraft footage, download free clips from pixabay.com/videos")
        log("editor", "    or record your screen, then drop any .mp4 into ./footage/")
        clip = OUTPUT_DIR / "placeholder.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=black:s=1080x1920:r=30",
            "-t", "180", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(clip)
        ], capture_output=True)

    # FFmpeg: scale -> crop -> burn subtitles -> mix audio
    log("editor", "FFmpeg: 1080x1920 vertical, burning subtitles, mixing audio...")

    # Windows fix: copy srt to output dir root so path has no drive colon issues
    # Copy SRT to CWD as a plain filename — avoids drive-letter colons that
    # break FFmpeg's subtitles filter parser on Windows.
    import shutil
    safe_srt = Path("subs_tmp.srt")          # relative, no colon
    shutil.copy(str(srt_path), str(safe_srt))

    # A bare relative filename has no special characters, so no escaping needed.
    srt_path_esc = safe_srt.as_posix()       # → "subs_tmp.srt"

    # Wrap the srt path in single quotes inside the filter so FFmpeg's
    # filter parser treats it as a literal string, not option tokens.
    sub_filter = (
        f"subtitles='{srt_path_esc}'"
        ":force_style='FontSize=18,Alignment=2,MarginV=40,Bold=1,"
        "PrimaryColour=16777215,OutlineColour=0,Outline=2,Shadow=1'"
    )

    cmd = ["ffmpeg", "-y"]
    if Path(audio_path).stat().st_size > 0:
        # Loop the footage so it always covers the narration, then explicitly
        # map the footage VIDEO + narration AUDIO. Without -map, FFmpeg's default
        # stream selection keeps the footage's own audio and drops the narration;
        # worse, with -stream_loop -1 the only mapped stream loops forever, so
        # -shortest has no finite anchor and the encode runs away → "Conversion
        # failed!". Mapping the finite WAV gives -shortest a real stopping point.
        cmd += ["-stream_loop", "-1", "-i", str(clip), "-i", audio_path]
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    else:
        # No narration audio: play the footage once (no infinite loop to bound).
        cmd += ["-i", str(clip), "-map", "0:v:0"]
    cmd += [
        "-vf", f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,{sub_filter}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        str(video_out),
    ]

    log("editor", f"FFmpeg cmd: {chr(32).join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        safe_srt.unlink(missing_ok=True)   # don't leave subs_tmp.srt behind

    if result.returncode != 0:
        log("editor", "FFmpeg FAILED. Error:")
        # Surface the actual error lines, not just the encoder summary stats
        # (libx264 prints kb/s/etc. on close, which used to bury the real cause
        # when we only showed the last 800 chars).
        err_lines = [ln for ln in result.stderr.splitlines()
                     if re.search(r"error|invalid|failed|no such|cannot|unable",
                                  ln, re.IGNORECASE)]
        log("editor", "\n".join(err_lines[-15:]) or result.stderr[-1200:])
        # Report failure instead of writing an empty file and claiming success —
        # the supervisor can then retry or stop, rather than reporting a broken
        # 0-byte video as "complete".
        return {**state, "video": None, "error": "FFmpeg render failed"}

    size_mb = video_out.stat().st_size / 1_048_576
    log("editor", f"[OK] Video: {video_out} ({size_mb:.1f} MB)")
    return {**state, "video": str(video_out), "error": None, "next": "done"}

# ── AGENT: SUPERVISOR ─────────────────────────────────────────
def supervisor_agent(state: PipelineState) -> PipelineState:
    if state.get("error"):
        if state["retries"] >= 3:
            # Give up rather than re-routing to a stage that keeps failing
            # (which would loop forever, since the failed stage leaves its
            # output unset).
            log("supervisor", f"[FAIL] Giving up after {state['retries']} retries: {state['error']}")
            return {**state, "next": "done"}
        log("supervisor", f"Error — retry #{state['retries']+1}: {state['error']}")
        return {**state, "retries": state["retries"] + 1, "error": None,
                "next": "scout" if not state.get("story") else
                        "narrator" if not state.get("audio") else "editor"}

    if not state.get("story"):
        log("supervisor", "No story yet → Scout")
        return {**state, "next": "scout"}
    if not state.get("audio"):
        log("supervisor", "Story ready, no audio → Narrator")
        return {**state, "next": "narrator"}
    if not state.get("video"):
        log("supervisor", "Audio ready, no video → Editor")
        return {**state, "next": "editor"}

    log("supervisor", f"[OK] Pipeline complete - {state['video']}")
    return {**state, "next": "done"}

# ── ROUTING ───────────────────────────────────────────────────
def route(state: PipelineState) -> Literal["scout", "narrator", "editor", "__end__"]:
    return "__end__" if state.get("next") == "done" else state.get("next", "scout")

# ── BUILD GRAPH ───────────────────────────────────────────────
def build_graph():
    g = StateGraph(PipelineState)
    g.add_node("supervisor", supervisor_agent)
    g.add_node("scout",      scout_agent)
    g.add_node("narrator",   narrator_agent)
    g.add_node("editor",     editor_agent)
    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route)
    for agent in ["scout", "narrator", "editor"]:
        g.add_edge(agent, "supervisor")
    return g.compile()

# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Multi-Agent Video Pipeline  (DeepSeek V3 + Kokoro TTS + LangGraph)")
    print("─" * 60)
    graph = build_graph()

    n_videos = max(NUM_VIDEOS, 1)
    print(f"Target: {n_videos} video(s) this run\n")

    # Generate and rank a story pool ONCE, then make one video per top story.
    # (Generating a few extra candidates than needed gives the ranker some room.)
    pool     = brainstorm_stories(max(5, n_videos + 2))
    selected = pool[:n_videos]
    if len(selected) < n_videos:
        log("scout", f"Only {len(selected)} distinct stories available — making that many.")

    results = []
    for n, story in enumerate(selected, start=1):
        print("\n" + "═" * 60)
        print(f"VIDEO {n}/{len(selected)} — {story['title']}")
        print("═" * 60)
        # Pre-seed the story so the supervisor skips Scout and goes straight to
        # Narrator → Editor for this specific story.
        state: PipelineState = {
            "run_id":  uuid.uuid4().hex[:8],
            "story":   story,
            "audio":   None,
            "srt":     None,
            "video":   None,
            "error":   None,
            "retries": 0,
            "next":    "supervisor",
        }
        results.append(graph.invoke(state))

    print("\n" + "─" * 60)
    print(f"Final summary — {len(results)} video(s):")
    ok = 0
    for r in results:
        title = r["story"]["title"] if r.get("story") else "—"
        if r.get("video") and not r.get("error"):
            ok += 1
            print(f"  [OK]   {r['video']}  ←  {title}")
        else:
            print(f"  [FAIL] {r.get('error') or 'unknown error'}  ←  {title}")
    print(f"\nDone! {ok}/{len(results)} succeeded.")