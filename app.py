import os
import socket
import tempfile
import ipaddress
from urllib.parse import urlparse

import numpy as np
import requests
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from pydub import AudioSegment
from pydub.utils import which as pydub_which

import torch
import torch.nn.functional as F
from transformers import ClapModel, ClapProcessor

# ============================
# Config
# ============================
TARGET_SR = 48_000
MAX_SECONDS = 20
MAX_UPLOAD_MB = 30
MAX_URL_MB = 35

ALLOWED_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# ============================
# pydub ffmpeg binding
# ============================
AudioSegment.converter = pydub_which("ffmpeg") or AudioSegment.converter
AudioSegment.ffprobe = pydub_which("ffprobe") or AudioSegment.ffprobe

# ============================
# Labels (internal, from files)
# ============================
def load_labels(path: str, fallback: list[str]) -> list[str]:
    try:
        out, seen = [], set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                x = line.strip()
                if not x or x.startswith("#"):
                    continue
                k = x.lower()
                if k not in seen:
                    seen.add(k)
                    out.append(x)
        return out if out else fallback
    except Exception:
        return fallback

# IMPORTANT: ці списки потрібні моделі для ранжування,
# але ми їх не показуємо в UI і не даємо вводити користувачу.
GENRE_LABELS = load_labels("data/genres.txt", ["rock", "pop", "classical"])
MOOD_LABELS  = load_labels("data/moods.txt",  ["party", "study", "relaxing"])

TOP_K = 5  # завжди топ-5

# ============================
# CLAP model + caching
# ============================
_clap_model = None
_clap_processor = None
_TEXT_EMB_CACHE = {}  # key: ("genre"/"mood", tuple(labels)) -> torch.Tensor [N,D]

def get_clap():
    global _clap_model, _clap_processor
    if _clap_model is None:
        _clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        _clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        _clap_model.eval()
    return _clap_model, _clap_processor

def get_text_embeds(category: str, labels: list[str]) -> torch.Tensor:
    key = (category, tuple(labels))
    if key in _TEXT_EMB_CACHE:
        return _TEXT_EMB_CACHE[key]

    model, proc = get_clap()
    all_vecs = []
    batch = 128
    with torch.no_grad():
        for i in range(0, len(labels), batch):
            chunk = labels[i:i+batch]
            inputs = proc(text=chunk, return_tensors="pt", padding=True)
            t = model.get_text_features(**inputs)
            t = F.normalize(t, dim=-1)
            all_vecs.append(t)
    t = torch.cat(all_vecs, dim=0)
    _TEXT_EMB_CACHE[key] = t
    return t

def get_audio_embed(samples: np.ndarray) -> torch.Tensor:
    model, proc = get_clap()
    inputs = proc(audios=samples, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        a = model.get_audio_features(**inputs)
        a = F.normalize(a, dim=-1)
    return a  # [1, D]

def score_to_pct(cos_sim: float) -> float:
    # cosine similarity roughly in [-1..1]
    pct = (cos_sim + 1.0) * 50.0
    return float(max(0.0, min(100.0, pct)))

def rank_labels(samples: np.ndarray, labels: list[str], category: str, top_k: int = 5):
    if not labels:
        return []

    a = get_audio_embed(samples)           # [1,D]
    t = get_text_embeds(category, labels)  # [N,D]
    sims = (a @ t.T).squeeze(0)            # [N]

    k = min(top_k, len(labels))
    vals, idx = torch.topk(sims, k=k)

    results = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        results.append({
            "label": labels[i],
            "score": float(v),
            "pct": score_to_pct(float(v))
        })
    return results

# ============================
# Helpers
# ============================
def _ext_ok(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT

def _is_private_host(hostname: str) -> bool:
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip)
        return (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
        )
    except Exception:
        return True

def load_audio_as_array(filepath: str) -> np.ndarray:
    try:
        audio = AudioSegment.from_file(filepath)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg не знайдено. Встанови ffmpeg і додай у PATH.")
    except Exception as e:
        raise RuntimeError(f"Не вдалося декодувати аудіо: {e}")

    audio = audio.set_channels(1).set_frame_rate(TARGET_SR)
    audio = audio[: MAX_SECONDS * 1000]

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    denom = float(1 << (8 * audio.sample_width - 1))
    if denom > 0:
        samples = samples / denom
    return samples

def looks_like_audio(first_bytes: bytes, content_type: str) -> bool:
    ct = (content_type or "").lower()
    if "audio" in ct:
        return True
    if first_bytes.startswith(b"ID3"):
        return True
    if first_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return True
    if first_bytes[:4] == b"RIFF":
        return True
    if first_bytes.startswith(b"fLaC"):
        return True
    if first_bytes.startswith(b"OggS"):
        return True
    return False

def download_audio(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError("URL має починатися з http або https.")
    if not parsed.netloc or _is_private_host(parsed.hostname or ""):
        raise RuntimeError("Небезпечний або некоректний хост у URL.")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "*/*",
    }

    r = requests.get(url, stream=True, timeout=25, headers=headers, allow_redirects=True)

    # Пояснюємо причину текстом (без JSON)
    if r.status_code == 403:
        raise RuntimeError("403 Forbidden: сайт блокує скачування. Дай прямий mp3/ogg URL або завантаж файл.")
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP помилка: {r.status_code}")

    ctype = r.headers.get("Content-Type", "")
    it = r.iter_content(chunk_size=256 * 1024)
    first = next(it, b"")

    if not looks_like_audio(first, ctype):
        raise RuntimeError("URL не є прямим аудіофайлом (схоже, це сторінка/HTML). Потрібен прямий .mp3/.ogg/.wav...")

    suffix = os.path.splitext(parsed.path)[1].lower()
    if not suffix or suffix not in ALLOWED_EXT:
        suffix = ".audio"

    fd, path = tempfile.mkstemp(suffix=suffix)
    total = 0
    with os.fdopen(fd, "wb") as f:
        f.write(first)
        total += len(first)
        for chunk in it:
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_URL_MB * 1024 * 1024:
                raise RuntimeError("Файл завеликий (URL).")
            f.write(chunk)
    return path

def recommend_from_mood(top_mood: str) -> str:
    m = (top_mood or "").lower()
    if any(x in m for x in ["party", "energetic", "upbeat", "dance", "club"]):
        return "🎉 Підійде для **вечірки**"
    if any(x in m for x in ["study", "focus", "concentration", "lofi", "instrumental"]):
        return "📚 Підійде для **навчання / фокусу**"
    if any(x in m for x in ["calm", "relax", "sleep", "ambient", "chill"]):
        return "🌿 Підійде для **відпочинку / релаксу**"
    return "✅ Можливий **змішаний настрій**"

# ============================
# Routes
# ============================
@app.get("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    tmp_path = None
    try:
        # 1) URL OR file
        url = (request.form.get("url") or "").strip()

        if url:
            tmp_path = download_audio(url)
        else:
            if "file" not in request.files:
                return ("Немає файлу або URL.", 200, {"Content-Type": "text/plain; charset=utf-8"})

            f = request.files["file"]
            if not f or not f.filename:
                return ("Файл не вибрано.", 200, {"Content-Type": "text/plain; charset=utf-8"})

            filename = secure_filename(f.filename)
            if not _ext_ok(filename):
                return ("Непідтримуваний формат. Спробуй mp3/wav/flac/ogg/m4a/aac.", 200,
                        {"Content-Type": "text/plain; charset=utf-8"})

            _, ext = os.path.splitext(filename.lower())
            fd, tmp_path = tempfile.mkstemp(suffix=ext)
            with os.fdopen(fd, "wb") as out:
                out.write(f.read())

        # 2) audio -> samples
        samples = load_audio_as_array(tmp_path)

        # 3) predict
        genre_scores = rank_labels(samples, GENRE_LABELS, "genre", top_k=TOP_K)
        mood_scores  = rank_labels(samples, MOOD_LABELS,  "mood",  top_k=TOP_K)

        top_genre = genre_scores[0]["label"] if genre_scores else "unknown"
        top_mood  = mood_scores[0]["label"] if mood_scores else "unknown"

        out = {
            "top_genre": top_genre,
            "top_mood": top_mood,
            "recommendation": recommend_from_mood(top_mood),
            "genres": genre_scores,
            "moods": mood_scores,
        }
        return jsonify(out)

    except Exception as e:
        # ВАЖЛИВО: на помилку повертаємо ЛИШЕ ТЕКСТ причини
        return (str(e), 200, {"Content-Type": "text/plain; charset=utf-8"})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=True)
