"""
Focus Quest — ADHD Attention Trainer
• Eye-tracking engagement monitoring via MediaPipe iris landmarks
• Adaptive difficulty: responds to blink rate, gaze stability, saccade patterns
• Session data exported to JSON for parent dashboard
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import random
import math
import json
import os
from datetime import datetime
from collections import deque
from PIL import Image, ImageFont, ImageDraw

# ─────────────────────────────────────────────
#  MediaPipe Setup
# ─────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720
FULLSCREEN    = False

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_data")
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  Iris / Face landmarks
# ─────────────────────────────────────────────
LEFT_IRIS   = [474, 475, 476, 477]
RIGHT_IRIS  = [469, 470, 471, 472]
LEFT_EYE_V  = [159, 145]   # top / bottom for blink
RIGHT_EYE_V = [386, 374]
NOSE_TIP    = 1
LEFT_FACE   = 234
RIGHT_FACE  = 454


# ═══════════════════════════════════════════════════════════════════
#  EMOJI RENDERING  (PIL → numpy → OpenCV alpha composite)
#  OpenCV cannot render Unicode emoji — we use NotoColorEmoji via PIL
# ═══════════════════════════════════════════════════════════════════

def _find_emoji_font() -> str | None:
    """Auto-detect a colour emoji font on Windows, macOS, or Linux."""
    import sys, os, platform

    candidates = []

    if platform.system() == "Windows":
        windir = os.environ.get("WINDIR", "C:\\Windows")
        candidates += [
            os.path.join(windir, "Fonts", "seguiemj.ttf"),   # Segoe UI Emoji (Win10/11)
            os.path.join(windir, "Fonts", "NotoColorEmoji.ttf"),
        ]
        # Also check Noto font if user installed it
        for drive in ["C:", "D:", "E:"]:
            candidates += [
                f"{drive}\\Noto\\NotoColorEmoji.ttf",
                f"{drive}\\fonts\\NotoColorEmoji.ttf",
            ]

    elif platform.system() == "Darwin":  # macOS
        candidates += [
            "/System/Library/Fonts/Apple Color Emoji.ttc",
            "/Library/Fonts/NotoColorEmoji.ttf",
        ]

    else:  # Linux
        candidates += [
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
            "/usr/share/fonts/noto/NotoColorEmoji.ttf",
            "/usr/local/share/fonts/NotoColorEmoji.ttf",
        ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    # Last resort: search common font directories
    search_dirs = []
    if platform.system() == "Windows":
        search_dirs = [os.path.join(os.environ.get("WINDIR","C:\\Windows"), "Fonts")]
    elif platform.system() == "Darwin":
        search_dirs = ["/System/Library/Fonts", "/Library/Fonts",
                       os.path.expanduser("~/Library/Fonts")]
    else:
        search_dirs = ["/usr/share/fonts", "/usr/local/share/fonts",
                       os.path.expanduser("~/.fonts")]

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if "emoji" in f.lower() and f.endswith((".ttf", ".ttc", ".otf")):
                    return os.path.join(root, f)
    return None

_EMOJI_FONT_PATH = _find_emoji_font()
if _EMOJI_FONT_PATH:
    print(f"✅ Emoji font found: {_EMOJI_FONT_PATH}")
else:
    print("⚠️  No colour emoji font found — animal icons will be text labels.")
    print("   Download NotoColorEmoji.ttf from fonts.google.com and place it")
    print("   in the same folder as this script, then restart.")

_emoji_font_cache: dict = {}   # size → ImageFont
_emoji_img_cache: dict  = {}   # (char, size) → numpy RGBA array

# NotoColorEmoji is a bitmap font with fixed internal sizes.
# Segoe UI Emoji (Windows) is a vector font and works with any size.
# We always request size 109 for NotoColorEmoji; Segoe works at any size.
_EMOJI_RENDER_SIZE = 109

def _get_emoji_font(size: int):
    if _EMOJI_FONT_PATH is None:
        return None
    if size not in _emoji_font_cache:
        try:
            _emoji_font_cache[size] = ImageFont.truetype(_EMOJI_FONT_PATH, _EMOJI_RENDER_SIZE)
        except OSError:
            # Segoe UI Emoji is a vector font — request target size directly
            try:
                _emoji_font_cache[size] = ImageFont.truetype(_EMOJI_FONT_PATH, max(size, 40))
            except OSError:
                _emoji_font_cache[size] = None
    return _emoji_font_cache[size]

def _render_emoji_img(emoji_char: str, target_size: int) -> np.ndarray | None:
    """Return an RGBA numpy array (target_size × target_size) for the emoji."""
    key = (emoji_char, target_size)
    if key in _emoji_img_cache:
        return _emoji_img_cache[key]
    font = _get_emoji_font(target_size)
    if font is None:
        return None
    try:
        canvas_size = max(220, target_size * 2)
        canvas = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), emoji_char, font=font, embedded_color=True)
        bbox = canvas.getbbox()
        if not bbox:
            return None
        cropped = canvas.crop(bbox)
        resized = cropped.resize((target_size, target_size), Image.LANCZOS)
        arr = np.array(resized)          # H×W×4  RGBA
        _emoji_img_cache[key] = arr
        return arr
    except Exception:
        return None

def draw_emoji(frame: np.ndarray, emoji_char: str, cx: int, cy: int, size: int) -> None:
    """Draw emoji centred at pixel (cx, cy) with given square size onto a BGR frame.
    Falls back to a coloured circle + letter if no emoji font is available."""
    arr = _render_emoji_img(emoji_char, size)
    if arr is None:
        # Fallback: draw a simple coloured circle so the game still looks ok
        fallback_colors = {'🐳':(200,160,60),'🐉':(60,60,200),'🐸':(60,180,60),
                           '🐝':(60,180,220),'🦉':(160,120,80),'⭐':(60,200,220),
                           '🧠':(180,100,200),'⏱️':(150,150,220),'🧘':(100,200,180)}
        col = fallback_colors.get(emoji_char, (140,140,180))
        cv2.circle(frame, (cx, cy), size//2, col, -1)
        cv2.circle(frame, (cx, cy), size//2, (255,255,255), 1)
        return
    fh, fw = frame.shape[:2]
    x1, y1 = cx - size // 2, cy - size // 2
    # Source slice (handle clipping)
    sx  = max(0, -x1);   sy  = max(0, -y1)
    ex  = min(size, fw - x1); ey  = min(size, fh - y1)
    fx1 = max(0, x1);    fy1 = max(0, y1)
    fx2 = fx1 + (ex - sx); fy2 = fy1 + (ey - sy)
    if fx2 <= fx1 or fy2 <= fy1 or ex <= sx or ey <= sy:
        return
    src   = arr[sy:ey, sx:ex]                    # RGBA crop
    alpha = src[:, :, 3:4].astype(np.float32) / 255.0
    rgb   = src[:, :, :3][:, :, ::-1].astype(np.float32)   # RGB→BGR
    bg    = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    frame[fy1:fy2, fx1:fx2] = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)



# ═══════════════════════════════════════════════════════════════════
#  EYE-TRACKING ENGAGEMENT ENGINE
# ═══════════════════════════════════════════════════════════════════
class EngagementTracker:
    """
    Monitors real-time engagement from iris / eyelid data.

    Signals tracked
    ───────────────
    • Gaze stability       — variance of iris x-position over ~1 s
    • Blink rate           — blinks/min; high rate → fatigue
    • Saccade velocity     — rapid eye jumps → distraction / exploration
    • Fixation duration    — cumulative on-target dwell time
    • Off-target ratio     — fraction of frames where gaze left the target
    • Pupillary oscillation — proxy for cognitive load (iris size variance)
    • Head-drift           — head turning, distinct from eye movement
    """

    WINDOW = 90          # frames (~3 s at 30 fps)
    BLINK_THRESH = 0.18  # eye-aspect-ratio below this = closed

    def __init__(self):
        self.iris_x_hist   = deque(maxlen=self.WINDOW)
        self.iris_y_hist   = deque(maxlen=self.WINDOW)
        self.head_x_hist   = deque(maxlen=self.WINDOW)
        self.ear_hist      = deque(maxlen=self.WINDOW)       # eye-aspect-ratio
        self.on_target_hist= deque(maxlen=self.WINDOW)       # bool

        self.blink_count   = 0
        self.blink_cooldown= 0
        self.in_blink      = False
        self.last_iris_x   = None
        self.saccade_count = 0          # large rapid jumps
        self.session_blinks= 0

        # Per-second snapshots for timeline export
        self.timeline: list[dict] = []
        self._last_snapshot = time.time()

        # Derived scores (0–1, updated each frame)
        self.gaze_stability    = 1.0
        self.blink_rate_score  = 1.0    # 1 = healthy, 0 = too high (fatigue)
        self.saccade_score     = 1.0    # 1 = calm, 0 = very saccadic
        self.on_target_score   = 1.0
        self.engagement_score  = 1.0    # composite
        self.fatigue_flag      = False
        self.distraction_flag  = False

    # ── helpers ──────────────────────────────────────────────────
    def _ear(self, landmarks, top_idx, bot_idx):
        top = landmarks[top_idx]
        bot = landmarks[bot_idx]
        return abs(top.y - bot.y)

    def _iris_pos(self, landmarks, indices):
        xs = [landmarks[i].x for i in indices]
        ys = [landmarks[i].y for i in indices]
        return sum(xs)/len(xs), sum(ys)/len(ys)

    # ── main update (call every frame) ───────────────────────────
    def update(self, landmarks, on_target: bool):
        lx, ly = self._iris_pos(landmarks, LEFT_IRIS)
        rx, ry = self._iris_pos(landmarks, RIGHT_IRIS)
        ix = (lx + rx) / 2
        iy = (ly + ry) / 2

        l_ear = self._ear(landmarks, LEFT_EYE_V[0],  LEFT_EYE_V[1])
        r_ear = self._ear(landmarks, RIGHT_EYE_V[0], RIGHT_EYE_V[1])
        avg_ear = (l_ear + r_ear) / 2

        hx = (landmarks[LEFT_FACE].x + landmarks[RIGHT_FACE].x) / 2

        # ── blink detection ──
        if avg_ear < self.BLINK_THRESH and not self.in_blink and self.blink_cooldown <= 0:
            self.in_blink = True
            self.blink_count += 1
            self.session_blinks += 1
            self.blink_cooldown = 8
        elif avg_ear >= self.BLINK_THRESH:
            self.in_blink = False
        if self.blink_cooldown > 0:
            self.blink_cooldown -= 1

        # ── saccade detection ──
        if self.last_iris_x is not None:
            jump = abs(ix - self.last_iris_x)
            if jump > 0.025:          # threshold in normalised coords
                self.saccade_count += 1
        self.last_iris_x = ix

        # ── store history ──
        self.iris_x_hist.append(ix)
        self.iris_y_hist.append(iy)
        self.head_x_hist.append(hx)
        self.ear_hist.append(avg_ear)
        self.on_target_hist.append(float(on_target))

        # ── derive scores (only when enough data) ──
        if len(self.iris_x_hist) >= 20:
            self._compute_scores()

        # ── per-second timeline snapshot ──
        now = time.time()
        if now - self._last_snapshot >= 1.0:
            self._snapshot(now)
            self._last_snapshot = now
            self.blink_count  = 0     # reset per-second counter
            self.saccade_count = 0

    def _compute_scores(self):
        arr_x = np.array(self.iris_x_hist)
        arr_h = np.array(self.head_x_hist)

        # Gaze stability (low variance = high stability)
        var = float(np.var(arr_x))
        self.gaze_stability = max(0.0, 1.0 - var * 60)

        # Blink rate score (target: 15–25 blinks/min → ~0.5–0.8 blinks/3 s)
        recent_blinks = sum(1 for e in list(self.ear_hist)[-90:]
                            if e < self.BLINK_THRESH)
        blink_rate_3s = recent_blinks
        if blink_rate_3s > 6:           # >120 bpm = high fatigue
            self.blink_rate_score = max(0.0, 1.0 - (blink_rate_3s - 6) / 6)
            self.fatigue_flag = True
        else:
            self.blink_rate_score = 1.0
            self.fatigue_flag = False

        # Saccade score (few rapid jumps = calm)
        saccades_in_window = sum(
            1 for i in range(1, len(arr_x))
            if abs(float(arr_x[i]) - float(arr_x[i-1])) > 0.025
        )
        self.saccade_score = max(0.0, 1.0 - saccades_in_window / 20)
        self.distraction_flag = saccades_in_window > 12

        # On-target score
        self.on_target_score = float(np.mean(list(self.on_target_hist)))

        # Composite engagement
        self.engagement_score = (
            self.gaze_stability   * 0.35 +
            self.blink_rate_score * 0.15 +
            self.saccade_score    * 0.25 +
            self.on_target_score  * 0.25
        )

    def _snapshot(self, ts: float):
        self.timeline.append({
            "ts":           round(ts, 2),
            "engagement":   round(self.engagement_score, 3),
            "stability":    round(self.gaze_stability, 3),
            "on_target":    round(self.on_target_score, 3),
            "saccade_score":round(self.saccade_score, 3),
            "blink_score":  round(self.blink_rate_score, 3),
            "fatigue":      self.fatigue_flag,
            "distraction":  self.distraction_flag,
        })

    def get_state_label(self):
        if self.fatigue_flag:
            return "TIRED", (100, 150, 255)
        if self.distraction_flag:
            return "DISTRACTED", (80, 180, 255)
        if self.engagement_score > 0.75:
            return "FOCUSED", (100, 255, 150)
        if self.engagement_score > 0.5:
            return "TRYING", (200, 230, 150)
        return "DRIFTING", (180, 180, 255)


# ═══════════════════════════════════════════════════════════════════
#  ADAPTIVE DIFFICULTY ENGINE
# ═══════════════════════════════════════════════════════════════════
class AdaptiveEngine:
    """
    Watches engagement + performance and nudges the game in real-time.
    Rules are designed to PREVENT frustration, not just track it.
    """
    COOLDOWN = 8.0   # minimum seconds between adaptations

    def __init__(self):
        self.last_adapt_time = 0.0
        self.consecutive_frustrated = 0
        self.consecutive_focused    = 0
        self.adaptations: list[dict] = []

    def evaluate(self, eng: EngagementTracker, game) -> str | None:
        """
        Returns an action string or None.
        Possible actions:
          'increase_time_tolerance'  – give more fixation time
          'decrease_time_tolerance'  – challenge more
          'activate_help'            – flash visual guides
          'deactivate_help'          – remove training wheels
          'suggest_break'            – fatigue detected
          'add_pulse'                – add pulsing cue
          'remove_pulse'             – reduce stimulation
        """
        now = time.time()
        if now - self.last_adapt_time < self.COOLDOWN:
            return None

        score = eng.engagement_score
        perf  = game.performance.get_success_rate()

        # ── frustration path ──────────────────────────────────
        if eng.fatigue_flag:
            self.last_adapt_time = now
            action = 'suggest_break'
            self._log(action, score, perf)
            return action

        if score < 0.35 or (perf < 0.3 and len(game.performance.recent_attempts) >= 5):
            self.consecutive_frustrated += 1
            self.consecutive_focused    = 0
            if self.consecutive_frustrated >= 2:
                self.last_adapt_time = now
                action = 'activate_help' if not game.help_mode_active else 'increase_time_tolerance'
                self._log(action, score, perf)
                return action

        # ── flow / mastery path ───────────────────────────────
        elif score > 0.78 and perf > 0.72:
            self.consecutive_focused    += 1
            self.consecutive_frustrated = 0
            if self.consecutive_focused >= 3:
                self.last_adapt_time = now
                action = 'deactivate_help' if game.help_mode_active else 'decrease_time_tolerance'
                self._log(action, score, perf)
                return action

        # ── distraction path ──────────────────────────────────
        elif eng.distraction_flag and not game.help_mode_active:
            self.last_adapt_time = now
            action = 'add_pulse'
            self._log(action, score, perf)
            return action

        elif not eng.distraction_flag and score > 0.65:
            self.consecutive_frustrated = 0

        return None

    def _log(self, action, eng_score, perf):
        self.adaptations.append({
            "ts":     round(time.time(), 2),
            "action": action,
            "eng":    round(eng_score, 3),
            "perf":   round(perf, 3),
        })


# ═══════════════════════════════════════════════════════════════════
#  SESSION DATA MANAGER  (saves JSON for the dashboard)
# ═══════════════════════════════════════════════════════════════════
class SessionManager:
    def __init__(self, child_name="Player"):
        self.child_name  = child_name
        self.session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time  = time.time()
        self.start_dt    = datetime.now().isoformat()

    def save(self, game, eng: EngagementTracker, adaptive: AdaptiveEngine):
        perf = game.performance
        duration = time.time() - self.start_time
        total    = len(perf.recent_attempts) + session_data["total_attempts"]
        success  = session_data["successful_attempts"]

        record = {
            "session_id":   self.session_id,
            "child_name":   self.child_name,
            "date":         self.start_dt,
            "duration_sec": round(duration, 1),

            # ── performance ──────────────────────────────────
            "performance": {
                "total_attempts":      session_data["total_attempts"],
                "successful_attempts": session_data["successful_attempts"],
                "success_rate_pct":    round(success / max(1, session_data["total_attempts"]) * 100, 1),
                "stars_collected":     game.total_gems,
                "best_combo":          game.best_combo,
                "final_level":         game.level,
                "avg_fixation_sec":    round(perf.get_average_fixation_time(), 2),
                "impulse_breaks":      game.impulse_breaks_total,
                "impulse_control_pct": round(perf.get_impulse_control_score() * 100, 1),
            },

            # ── engagement summary ────────────────────────────
            "engagement": {
                "avg_engagement":    round(float(np.mean([s["engagement"] for s in eng.timeline])) if eng.timeline else 0, 3),
                "avg_stability":     round(float(np.mean([s["stability"]  for s in eng.timeline])) if eng.timeline else 0, 3),
                "avg_on_target":     round(float(np.mean([s["on_target"]  for s in eng.timeline])) if eng.timeline else 0, 3),
                "total_blinks":      eng.session_blinks,
                "fatigue_events":    sum(1 for s in eng.timeline if s["fatigue"]),
                "distraction_events":sum(1 for s in eng.timeline if s["distraction"]),
                "timeline":          eng.timeline,
            },

            # ── adaptive events ───────────────────────────────
            "adaptations": adaptive.adaptations,

            # ── therapist insights ────────────────────────────
            "insights": self._generate_insights(game, eng, adaptive),
        }

        path = os.path.join(DATA_DIR, f"session_{self.session_id}.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

        # Also update a cumulative "all_sessions.json" for the dashboard
        all_path = os.path.join(DATA_DIR, "all_sessions.json")
        all_sessions = []
        if os.path.exists(all_path):
            try:
                with open(all_path) as f:
                    all_sessions = json.load(f)
            except Exception:
                all_sessions = []
        all_sessions.append(record)
        with open(all_path, "w") as f:
            json.dump(all_sessions, f, indent=2)

        print(f"\n✅ Session saved → {path}")
        print(f"📊 Dashboard data updated → {all_path}")
        return path

    def _generate_insights(self, game, eng: EngagementTracker, adaptive: AdaptiveEngine):
        insights = []
        perf = game.performance
        success_rate = session_data["successful_attempts"] / max(1, session_data["total_attempts"])
        avg_eng = float(np.mean([s["engagement"] for s in eng.timeline])) if eng.timeline else 0

        if avg_eng > 0.72:
            insights.append({"type": "positive", "text": "Excellent sustained engagement throughout the session."})
        elif avg_eng > 0.5:
            insights.append({"type": "neutral", "text": "Moderate engagement — some periods of drift but generally attentive."})
        else:
            insights.append({"type": "attention", "text": "Low engagement detected. Consider shorter sessions or more breaks."})

        fatigue_events = sum(1 for s in eng.timeline if s["fatigue"])
        if fatigue_events > 5:
            insights.append({"type": "warning", "text": f"Fatigue signs detected ({fatigue_events} events). Reduce session length."})

        if perf.get_impulse_control_score() > 0.7:
            insights.append({"type": "positive", "text": "Strong impulse control — child waited patiently for rewards."})
        elif perf.get_impulse_control_score() < 0.4:
            insights.append({"type": "attention", "text": "Impulse control needs support. Use Help Mode (H) next session."})

        if game.level >= 5:
            insights.append({"type": "positive", "text": f"Reached Level {game.level} — great focus progression!"})

        if len(adaptive.adaptations) > 0:
            actions = [a["action"] for a in adaptive.adaptations]
            if actions.count("suggest_break") > 1:
                insights.append({"type": "warning", "text": "Multiple fatigue signals — next session limit to 10 min."})
            if actions.count("activate_help") > 2:
                insights.append({"type": "neutral", "text": "Help Mode activated frequently — child may benefit from level 1-2 practice."})

        return insights


# ═══════════════════════════════════════════════════════════════════
#  PERFORMANCE TRACKER  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════
class PerformanceTracker:
    def __init__(self):
        self.recent_attempts = []
        self.max_history = 15
        self.impulse_breaks = 0

    def add_attempt(self, success, fixation_time, accuracy_score, was_impulse_break=False):
        self.recent_attempts.append({
            'success': success, 'fixation_time': fixation_time,
            'accuracy_score': accuracy_score, 'timestamp': time.time(),
            'was_impulse_break': was_impulse_break
        })
        if was_impulse_break:
            self.impulse_breaks += 1
        if len(self.recent_attempts) > self.max_history:
            removed = self.recent_attempts.pop(0)
            if removed.get('was_impulse_break'):
                self.impulse_breaks = max(0, self.impulse_breaks - 1)

    def get_success_rate(self):
        if not self.recent_attempts: return 0
        return sum(1 for a in self.recent_attempts if a['success']) / len(self.recent_attempts)

    def get_average_fixation_time(self):
        good = [a['fixation_time'] for a in self.recent_attempts if a['success'] and a['fixation_time'] > 0.3]
        return sum(good) / len(good) if good else 0

    def get_impulse_control_score(self):
        if len(self.recent_attempts) < 5: return 0.5
        return max(0.1, 1 - self.impulse_breaks / len(self.recent_attempts))

    def should_increase_difficulty(self):
        if len(self.recent_attempts) < 8: return False
        return (self.get_success_rate() >= 0.75 and
                self.get_impulse_control_score() >= 0.6 and
                self.get_average_fixation_time() > 0.8)

    def should_decrease_difficulty(self):
        if len(self.recent_attempts) < 6: return False
        return self.get_success_rate() < 0.4 or self.get_impulse_control_score() < 0.3


# ═══════════════════════════════════════════════════════════════════
#  GAME LEVEL
# ═══════════════════════════════════════════════════════════════════
class GameLevel:
    def __init__(self):
        self.level = 1
        self.target_fix_time = 1.5
        self.num_colors = 2
        self.use_words = False
        self.stroop_mode = False
        self.score = 0
        self.total_gems = 0
        self.combo = 0
        self.best_combo = 0
        self.attempts_at_level = 0
        self.session_start_time = time.time()
        self.help_mode_active = False
        self.force_pulse = False        # set by adaptive engine
        self.themes = [
            {"name": "BLUE WHALE",  "color": (255, 180, 80),  "emoji": "🐳"},
            {"name": "RED DRAGON",  "color": (80,  80,  255), "emoji": "🐉"},
            {"name": "GREEN FROG",  "color": (100, 220, 100), "emoji": "🐸"},
            {"name": "SUNNY BEE",   "color": (80,  200, 255), "emoji": "🐝"},
        ]
        self.performance = PerformanceTracker()
        self.impulse_breaks_total = 0

    def get_level_description(self):
        descriptions = {
            1: {"name": "👶 BABY DIVER",        "therapeutic_goal": "Build confidence with easy wins",       "visual_aids": "Big friendly guides!", "challenge": "Find the whale or dragon",         "skills": ["Gentle focus", "Starting calm"]},
            2: {"name": "🐸 FROG JUMPER",        "therapeutic_goal": "Practice choice without pressure",      "visual_aids": "Helpful guides",        "challenge": "Pick the right animal friend",     "skills": ["Decision making", "Ignoring distractions"]},
            3: {"name": "🐢 SLOW TURTLE",        "therapeutic_goal": "Build attention stamina gently",        "visual_aids": "Subtle helpers",         "challenge": "Stay a little longer",             "skills": ["Patience", "Calm focus"]},
            4: {"name": "🐝 BEE BUDDY",          "therapeutic_goal": "Connect words with focus",              "visual_aids": "Light guides",           "challenge": "Find the animal by name!",         "skills": ["Word recognition", "Visual search"]},
            5: {"name": "🌈 RAINBOW CHAMELEON",  "therapeutic_goal": "Strengthen impulse control",            "visual_aids": "Faint guides",           "challenge": "Look at COLOR, not the word!",     "skills": ["Ignoring tricks", "Mental flexibility"]},
            6: {"name": "🦉 WISE OWL",           "therapeutic_goal": "Build internal focus strength",         "visual_aids": "Tiny hints",             "challenge": "Focus with almost no help",         "skills": ["Independent focus", "Self-reliance"]},
            7: {"name": "✨ STAR GAZER",          "therapeutic_goal": "Celebrate focus independence",          "visual_aids": "You are the guide!",     "challenge": "No helpers — just your focus!",    "skills": ["Mastery", "Confidence", "Calm mind"]},
        }
        level = min(self.level, 7)
        if self.level > 7:
            d = descriptions[7].copy()
            d["name"] = f"✨ STAR LEVEL {self.level}"
            return d
        return descriptions[level]

    def get_visual_assistance_config(self):
        config = {
            'show_pulse': True, 'pulse_intensity': 1.0,
            'show_border': True, 'border_opacity': 1.0,
            'show_arrow': True,  'arrow_opacity': 1.0,
            'target_size_boost': 1.0,
        }
        if self.help_mode_active or self.force_pulse:
            config.update({'pulse_intensity': 1.3, 'arrow_opacity': 1.0,
                           'border_opacity': 1.0, 'target_size_boost': 1.1})
            return config
        lvl = self.level
        if lvl == 1:   config.update({'pulse_intensity': 1.2, 'border_opacity': 1.0,  'arrow_opacity': 1.0,  'target_size_boost': 1.05})
        elif lvl == 2: config.update({'pulse_intensity': 1.0, 'border_opacity': 0.9,  'arrow_opacity': 0.85, 'target_size_boost': 1.0})
        elif lvl == 3: config.update({'pulse_intensity': 0.85,'border_opacity': 0.75, 'arrow_opacity': 0.65, 'target_size_boost': 1.0})
        elif lvl == 4: config.update({'pulse_intensity': 0.7, 'border_opacity': 0.6,  'arrow_opacity': 0.45, 'target_size_boost': 0.95})
        elif lvl == 5: config.update({'pulse_intensity': 0.55,'border_opacity': 0.45, 'arrow_opacity': 0.25, 'target_size_boost': 0.95})
        elif lvl == 6: config.update({'pulse_intensity': 0.35,'border_opacity': 0.3,  'arrow_opacity': 0.1,  'target_size_boost': 0.9})
        else:          config.update({'show_pulse': False, 'pulse_intensity': 0.0, 'border_opacity': 0.15,
                                      'show_arrow': False, 'arrow_opacity': 0.0,  'target_size_boost': 0.9})
        return config


# ═══════════════════════════════════════════════════════════════════
#  PARTICLE SYSTEM
# ═══════════════════════════════════════════════════════════════════
class Particle:
    def __init__(self, x, y, color, is_star=False):
        self.x, self.y = x, y
        self.vx = random.uniform(-4, 4)
        self.vy = random.uniform(-7, -2)
        self.color = color
        self.life = random.randint(30, 50)
        self.size = random.randint(3, 8)
        self.is_star = is_star
        self.angle = random.uniform(0, 2*math.pi)
        self.rot   = random.uniform(-0.1, 0.1)

    def update(self):
        self.x += self.vx; self.y += self.vy
        self.vy += 0.25; self.life -= 1
        self.size = max(1, self.size - 0.2)
        self.angle += self.rot
        return self.life > 0

    def draw(self, frame):
        if self.is_star:
            pts = []
            for i in range(5):
                a = self.angle + i * 2*math.pi/5
                pts.append((int(self.x + math.cos(a)*self.size*2),
                             int(self.y + math.sin(a)*self.size*2)))
                b = a + math.pi/5
                pts.append((int(self.x + math.cos(b)*self.size),
                             int(self.y + math.sin(b)*self.size)))
            cv2.fillPoly(frame, [np.array(pts, np.int32)], self.color)
        else:
            cv2.circle(frame, (int(self.x), int(self.y)), int(self.size), self.color, -1)


# ═══════════════════════════════════════════════════════════════════
#  GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════
game     = GameLevel()
eng      = EngagementTracker()
adaptive = AdaptiveEngine()
session_mgr = SessionManager()

session_data = {"start_time": time.time(), "total_attempts": 0, "successful_attempts": 0}

fixation_time       = 0.0
fixation_start_time = 0.0
target_zone         = 1
gaze_history        = deque(maxlen=5)
head_pos_history    = deque(maxlen=20)
gaze_stability_scores = deque(maxlen=25)
message_display     = {"text": "", "time": 0, "color": (255,255,255)}
particles: list[Particle] = []
tutorial_mode  = True
tutorial_step  = 0
last_help_press = 0.0
show_break_reminder = False
tracking_status = "CALM"
eye_variance    = 0.0
head_variance   = 0.0

success_messages = [
    "🐳 YES! Happy whale!","🐉 ROAR! Dragon power!","🐸 BOING! Frog jump!",
    "🐝 BUZZ! Bee happy!","✨ MAGIC!","🌈 RAINBOW SUCCESS!","⭐ YOU DID IT!",
    "🎯 PERFECT!","💫 SHINY FOCUS!","🌟 SUPER STAR!"
]
effort_messages = [
    "💙 Almost there!","💚 Keep trying!","💛 You can do it!",
    "🧡 Try again!","💜 So close!","💗 I believe in you!",
    "🤍 Breathe and try","🤎 Gentle focus…","❤️ You're learning!"
]
combo_messages = [
    "💫 TWINKLE x2!","✨ SPARKLE x3!","🌟 SUPER SPARKLE x4!",
    "🌈 RAINBOW x5!","🌠 GALAXY x6+!"
]


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def show_message(text, color=(100,255,100), duration=1.8):
    message_display["text"] = text
    message_display["time"] = time.time() + duration
    message_display["color"] = color

def create_celebration(x, y, color):
    for _ in range(25):
        particles.append(Particle(x, y, color, is_star=random.random()>0.6))

def create_encouragement(x, y):
    for _ in range(12):
        c = (random.randint(150,255), random.randint(150,255), random.randint(200,255))
        particles.append(Particle(x, y, c))

def calculate_accuracy_score():
    if len(gaze_stability_scores) < 3: return 0.5
    return max(0.2, min(1.0, sum(gaze_stability_scores)/len(gaze_stability_scores)))

def new_target():
    global target_zone
    target_zone = random.randint(0, game.num_colors - 1)

def draw_rounded_rect(img, pt1, pt2, color, thickness, r=15):
    x1,y1 = pt1; x2,y2 = pt2
    cv2.line(img,(x1+r,y1),(x2-r,y1),color,thickness)
    cv2.line(img,(x1+r,y2),(x2-r,y2),color,thickness)
    cv2.line(img,(x1,y1+r),(x1,y2-r),color,thickness)
    cv2.line(img,(x2,y1+r),(x2,y2-r),color,thickness)
    cv2.ellipse(img,(x1+r,y1+r),(r,r),180,0,90,color,thickness)
    cv2.ellipse(img,(x2-r,y1+r),(r,r),270,0,90,color,thickness)
    cv2.ellipse(img,(x1+r,y2-r),(r,r), 90,0,90,color,thickness)
    cv2.ellipse(img,(x2-r,y2-r),(r,r),  0,0,90,color,thickness)

def get_gaze_zone(landmarks):
    lx,_ = eng._iris_pos(landmarks, LEFT_IRIS)
    rx,_ = eng._iris_pos(landmarks, RIGHT_IRIS)
    avg  = (lx + rx) / 2
    hx   = (landmarks[LEFT_FACE].x + landmarks[RIGHT_FACE].x) / 2
    head_pos_history.append(hx)
    combined = avg*0.6 + hx*0.4
    gaze_history.append(combined)
    smoothed = sum(gaze_history)/len(gaze_history)
    if smoothed < 0.40:   return 0
    elif smoothed > 0.60: return 2 if game.num_colors > 2 else 1
    else:                 return 1

def get_tracking_status(landmarks):
    lx,_ = eng._iris_pos(landmarks, LEFT_IRIS)
    rx,_ = eng._iris_pos(landmarks, RIGHT_IRIS)
    ix   = (lx + rx) / 2
    hx   = (landmarks[LEFT_FACE].x + landmarks[RIGHT_FACE].x) / 2
    gaze_history.append(ix)
    head_pos_history.append(hx)
    if len(gaze_history) >= 8:
        ev = float(np.var(list(gaze_history)[-8:]))
        hv = float(np.var(list(head_pos_history)[-8:]))
        if hv > ev*1.5:   return "HEAD_MOVING", ev, hv
        elif ev > 0.008:  return "EYES_MOVING", ev, hv
    return "CALM", 0.0, 0.0

def calculate_gaze_stability(zone, tgt):
    if zone == tgt and len(gaze_history) >= 4:
        var = float(np.var(list(gaze_history)[-4:]))
        s   = max(0.1, 1 - var*40)
        gaze_stability_scores.append(s)
        return s
    return 0.1

def update_difficulty():
    game.attempts_at_level += 1
    session_data["total_attempts"] += 1
    if game.performance.should_increase_difficulty():
        game.level += 1; game.attempts_at_level = 0
        game.help_mode_active = False; game.score = 0
        ld = game.get_level_description()
        if game.level == 2:  game.num_colors = 3;       show_message(f"🎉 {ld['name']}\nNew friend joined!", (100,255,255), 3.0)
        elif game.level == 3: game.target_fix_time = 2.0; show_message(f"🐢 {ld['name']}\nHold a little longer!",(100,200,100),3.0)
        elif game.level == 4: game.use_words = True;      show_message(f"🐝 {ld['name']}\nNow with word friends!",(255,220,100),3.0)
        elif game.level == 5: game.stroop_mode = True;    show_message(f"🌈 {ld['name']}\nColor trick challenge!",(200,100,255),3.0)
        elif game.level == 6: game.target_fix_time = 2.5; show_message(f"🦉 {ld['name']}\nQuiet focus time",(150,150,255),3.0)
        elif game.level == 7: show_message(f"✨ {ld['name']}\nYou're a focus master!",(255,215,0),4.0)
        else:
            game.target_fix_time = min(3.5, 2.5+(game.level-7)*0.15)
            show_message(f"✨ STAR LEVEL {game.level}!",(255,215,0),3.0)
    elif game.performance.should_decrease_difficulty():
        if game.level > 1:
            game.level -= 1; game.help_mode_active = True
            show_message("🤗 Let's make it easier!\nYou've got this!",(100,255,255),3.0)
            if game.level <= 2: game.target_fix_time = 1.5
            if game.level == 1: game.num_colors = 2
            if game.level < 4:  game.use_words = False
            if game.level < 5:  game.stroop_mode = False


# ─────────────────────────────────────────────
#  Draw functions
# ─────────────────────────────────────────────
def draw_engagement_hud(frame, h, w):
    """Tiny coloured dot in the corner — visible to parent/therapist,
    ignored by the child. Green=focused, Yellow=drifting, Red=tired."""
    _, lcol = eng.get_state_label()
    cx, cy, r = 22, 22, 9
    cv2.circle(frame, (cx, cy), r + 3, (15, 20, 35), -1)
    cv2.circle(frame, (cx, cy), r,     lcol,          -1)
    if eng.fatigue_flag:
        cv2.circle(frame, (cx, cy), r + 4, (80, 80, 220), 2)


def draw_level_info_panel(frame, h, w):
    pass  # removed — too much text for children


def draw_tracking_status(frame, h, w, status, ev, hv):
    pass  # removed — too distracting for children


def draw_hud(frame, h, w):
    ph=145; x1,y1=15,h-ph-15; x2,y2=340,h-15
    ov=frame.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),(30,35,50),-1)
    cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
    draw_rounded_rect(frame,(x1,y1),(x2,y2),(120,160,220),2,r=16)
    lv_em = ["","👶","🐸","🐢","🐝","🌈","🦉","✨"]
    em = lv_em[min(game.level,7)]
    draw_emoji(frame, em, x1+42, y1+38, 36)
    cv2.putText(frame,f"LEVEL {game.level}",(x1+68,y1+52),cv2.FONT_HERSHEY_DUPLEX,1.3,(220,240,255),2)
    draw_emoji(frame, "⭐", x1+38, y1+88, 28)
    cv2.putText(frame,f"STARS: {game.total_gems}",(x1+60,y1+100),cv2.FONT_HERSHEY_DUPLEX,1.05,(180,180,255),2)

    x1,y1=w-355,h-ph-15; x2,y2=w-15,h-15
    ov=frame.copy()
    bgc=(40,30,50) if show_break_reminder else (35,30,50)
    cv2.rectangle(ov,(x1,y1),(x2,y2),bgc,-1)
    cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
    bdc=(255,150,150) if show_break_reminder else (150,150,220)
    draw_rounded_rect(frame,(x1,y1),(x2,y2),bdc,2,r=16)
    st=int(time.time()-game.session_start_time)
    if show_break_reminder:
        draw_emoji(frame, "🧘", x1+40, y1+40, 28)
        cv2.putText(frame,"BREAK TIME!",(x1+68,y1+55),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,220,150),2)
    else:
        draw_emoji(frame, "⏱️", x1+40, y1+40, 28)
        cv2.putText(frame,f"{st//60:02d}:{st%60:02d}",(x1+65,y1+55),cv2.FONT_HERSHEY_DUPLEX,1.05,(220,220,255),2)

    if time.time() < message_display["time"]:
        txt=message_display["text"]; col=message_display["color"]
        lines=txt.split('\n')
        mw = max(cv2.getTextSize(l,cv2.FONT_HERSHEY_DUPLEX,1.7,4)[0][0] for l in lines)
        bx1=(w-mw)//2-28; by1=58; bx2=bx1+mw+56; by2=by1+len(lines)*62+18
        ov=frame.copy()
        cv2.rectangle(ov,(bx1,by1),(bx2,by2),(20,25,40),-1)
        cv2.addWeighted(ov,0.9,frame,0.1,0,frame)
        draw_rounded_rect(frame,(bx1,by1),(bx2,by2),(100,150,200),3,r=22)
        for idx,line in enumerate(lines):
            tw=cv2.getTextSize(line,cv2.FONT_HERSHEY_DUPLEX,1.7,4)[0][0]
            tx=(w-tw)//2; ty=98+idx*62
            cv2.putText(frame,line,(tx+3,ty+3),cv2.FONT_HERSHEY_DUPLEX,1.7,(15,20,30),5)
            cv2.putText(frame,line,(tx,ty),cv2.FONT_HERSHEY_DUPLEX,1.7,col,4)


def draw_instruction_hint(frame, h, w):
    theme=game.themes[target_zone%len(game.themes)]; em=theme["emoji"]
    # For stroop mode only keep the colour hint — all other levels just show the big emoji
    col=(220,220,255)
    # Draw a subtle pill behind the target emoji only
    draw_emoji(frame, em, w//2, 44, 52)
    if game.stroop_mode:
        hint_text="Look at the COLOR!"
        tw=cv2.getTextSize(hint_text,cv2.FONT_HERSHEY_DUPLEX,0.9,2)[0][0]
        cv2.putText(frame,hint_text,(w//2-tw//2,84),cv2.FONT_HERSHEY_DUPLEX,0.9,col,2)


def draw_focus_mascot(frame, h, w, prog):
    """Clean mascot: glowing circle that grows with progress + emoji on top."""
    mx, my = w//2, 105
    mascot = "🐳" if game.level<=2 else "🐸" if game.level<=4 else "🦉"
    # Glow ring — colour shifts from blue to green as progress grows
    radius = 62 + int(prog * 28)
    b = int(220 * (1 - prog)); g = int(220 * prog); r = 60
    cv2.circle(frame, (mx, my), radius + 6, (b//3, g//3, r//3), -1)
    cv2.circle(frame, (mx, my), radius + 6, (b, g, r), 3)
    draw_emoji(frame, mascot, mx, my, radius)


def draw_target_zones(frame, h, w, zone, vcfg):
    zw=int(w//game.num_colors*vcfg['target_size_boost'])
    bz=w//game.num_colors
    ov=frame.copy()
    for i in range(game.num_colors):
        bx1=i*bz; bx2=(i+1)*bz
        cx=(bx1+bx2)//2; x1=cx-zw//2; x2=cx+zw//2
        theme=game.themes[i%len(game.themes)]
        col=theme["color"]; em=theme["emoji"]; nm=theme["name"].split()[0]
        is_tgt=(i==target_zone)
        if is_tgt:
            ps=int(15*abs(math.sin(time.time()*6))*vcfg['pulse_intensity']) if vcfg['show_pulse'] else 0
            y1_z,y2_z=160-ps,340+ps
            cv2.rectangle(ov,(x1+20,y1_z),(x2-20,y2_z),col,-1)
            if vcfg['show_border']:
                bo=vcfg['border_opacity']
                bc=(int(255*bo),int(255*bo),int(255*bo))
                cv2.rectangle(frame,(x1+20,y1_z),(x2-20,y2_z),bc,max(2,int(8*bo)))
            if vcfg['show_arrow'] and vcfg['arrow_opacity']>0:
                ao=vcfg['arrow_opacity']; ax=(x1+x2)//2
                ac=(int(255*ao),int(255*ao),int(150*ao))
                cv2.arrowedLine(frame,(ax,90),(ax,120),ac,max(2,int(6*ao)),tipLength=0.4)
            cv2.putText(frame,em,(x1+(x2-x1)//2-45,260),cv2.FONT_HERSHEY_DUPLEX,4,(255,255,255),3)
        else:
            cv2.rectangle(ov,(x1+25,160),(x2-25,340),
                          (max(30,col[0]//2),max(30,col[1]//2),max(40,col[2]//2)),-1)
            cv2.putText(frame,em,(x1+(x2-x1)//2-35,250),cv2.FONT_HERSHEY_DUPLEX,3,(220,220,220),2)
        if game.use_words:
            tc=col
            if game.stroop_mode and is_tgt:
                tc=random.choice([t for j,t in enumerate(game.themes) if j!=i])["color"]
            tsz=cv2.getTextSize(nm,cv2.FONT_HERSHEY_DUPLEX,1.9,3)[0]
            cv2.putText(frame,nm,(x1+(x2-x1-tsz[0])//2,390),cv2.FONT_HERSHEY_DUPLEX,1.9,tc,3)
    cv2.addWeighted(ov,0.45,frame,0.55,0,frame)
    if zone==target_zone and fixation_time>0:
        bw=bz-80; bx=target_zone*bz+40; by=420
        cv2.rectangle(frame,(bx,by),(bx+bw,by+28),(60,70,90),-1)
        prog=min(1.0,fixation_time/game.target_fix_time)
        fw=int(bw*prog)
        for i in range(fw):
            r=i/fw if fw>0 else 0
            cv2.line(frame,(bx+i,by+4),(bx+i,by+24),(int(255*(1-r)),int(255*r),int(200*(1-r/2))),2)
        cv2.rectangle(frame,(bx,by),(bx+bw,by+28),(180,220,255),2)


def draw_tutorial(frame, h, w):
    ov=frame.copy()
    cv2.rectangle(ov,(0,0),(w,h),(15,25,40),-1)
    cv2.addWeighted(ov,0.88,frame,0.12,0,frame)
    # Emoji prefix + clean text pairs for tutorial lines
    lines_data=[
        ("✨", "Welcome, Focus Friend!"),
        ("🐳", "This game helps your brain get stronger at paying attention"),
        ("🌈", "Look at the animal friends to collect stars!"),
        ("⏱️", "Keep looking until the happy face fills up"),
        ("💖", "It's OK to move your head — we track eyes AND head!"),
        ("🌟", "Every try makes your focus superpower stronger!"),
        ("🧘", "Remember: Gentle focus > perfect focus"),
        ("🚀", "Press SPACE to start your adventure!"),
    ]
    ys=h//2-175
    for i,(em,txt) in enumerate(lines_data):
        sz=1.02 if i==tutorial_step else 0.78
        col=(100,255,255) if i==tutorial_step else (200,220,255)
        tw=cv2.getTextSize(txt,cv2.FONT_HERSHEY_DUPLEX,sz,2)[0][0]
        em_size = 34 if i==tutorial_step else 26
        total = tw + em_size + 10
        tx = w//2 - total//2
        draw_emoji(frame, em, tx + em_size//2, ys+i*52 - em_size//2 + 8, em_size)
        cv2.putText(frame,txt,(tx+em_size+8,ys+i*52),cv2.FONT_HERSHEY_DUPLEX,sz,col,2)
    if tutorial_step==len(lines_data)-1 and int(time.time()*2)%2==0:
        t2="PRESS SPACE TO BEGIN"
        tw=cv2.getTextSize(t2,cv2.FONT_HERSHEY_DUPLEX,1.55,3)[0][0]
        cv2.putText(frame,t2,(w//2-tw//2,h-85),cv2.FONT_HERSHEY_DUPLEX,1.55,(100,255,180),3)


# ─────────────────────────────────────────────
#  Apply adaptive engine action
# ─────────────────────────────────────────────
def apply_adaptive_action(action: str):
    global last_help_press
    if action == 'activate_help':
        game.help_mode_active = True
        last_help_press = time.time()
        show_message("🤗 Auto-Help ON!\nI noticed you need support", (100,255,200), 3.0)
    elif action == 'deactivate_help':
        game.help_mode_active = False
        game.force_pulse = False
        show_message("✨ Flying solo!\nYou're doing amazing!", (180,220,255), 2.5)
    elif action == 'increase_time_tolerance':
        game.target_fix_time = min(game.target_fix_time + 0.3, 4.0)
        show_message("💙 More time given!\nTake it steady…", (150,200,255), 2.5)
    elif action == 'decrease_time_tolerance':
        game.target_fix_time = max(game.target_fix_time - 0.2, 1.0)
        show_message("⚡ Feeling sharp!\nChallenge increased!", (255,220,100), 2.0)
    elif action == 'suggest_break':
        show_message("😴 Your eyes look tired!\nPress SPACE for a break", (180,160,255), 4.0)
    elif action == 'add_pulse':
        game.force_pulse = True
        show_message("✨ Look here!", (255,255,150), 1.5)
    elif action == 'remove_pulse':
        game.force_pulse = False


# ─────────────────────────────────────────────
#  Camera + main loop
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

new_target()

print("✨ FOCUS QUEST — Eye-Tracking ADHD Trainer ✨")
print("=" * 60)
print(f"📂 Session data will be saved to: {DATA_DIR}")
print("\n🎮 Controls:")
print("   SPACE = Start / Skip tutorial / Continue after break")
print("   H     = Toggle Help Mode")
print("   R     = Restart")
print("   F     = Toggle fullscreen")
print("   ESC   = Quit & save session")
print("=" * 60)

window_name = "✨ Focus Quest — Adaptive Attention Trainer ✨"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
if FULLSCREEN:
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    h, w, _ = frame.shape

    session_timer = time.time() - game.session_start_time
    show_break_reminder = 300 <= session_timer < 310

    # ── forced break window ──────────────────────────────────
    if 300 <= session_timer < 330:
        ov=frame.copy()
        cv2.rectangle(ov,(0,0),(w,h),(20,25,40),-1)
        cv2.addWeighted(ov,0.9,frame,0.1,0,frame)
        draw_emoji(frame, "🧘", w//2-340, h//2-50, 48)
        cv2.putText(frame,"BRAIN BREAK TIME!",(w//2-280,h//2-38),cv2.FONT_HERSHEY_DUPLEX,1.95,(150,220,255),3)
        draw_emoji(frame, "🧘", w//2+300, h//2-50, 48)
        cv2.putText(frame,"Rest your eyes for 30 seconds",(w//2-310,h//2+30),cv2.FONT_HERSHEY_DUPLEX,1.15,(220,240,255),2)
        cv2.putText(frame,"Press SPACE to continue playing",(w//2-340,h//2+95),cv2.FONT_HERSHEY_SIMPLEX,0.88,(180,220,255),2)
        cv2.imshow(window_name, frame)
        key=cv2.waitKey(1)&0xFF
        if key==32: game.session_start_time=time.time()-60
        if key==27: break
        continue

    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=face_mesh.process(rgb)

    # ── tutorial ─────────────────────────────────────────────
    if tutorial_mode:
        draw_tutorial(frame,h,w)
        cv2.imshow(window_name,frame)
        key=cv2.waitKey(1)&0xFF
        if key==32:
            tutorial_step+=1
            if tutorial_step>=8:
                tutorial_mode=False
                show_message("🚀 Let's go!",(100,255,180),2.5)
                game.session_start_time=time.time()
        if key==27: break
        continue

    # ── face tracking ─────────────────────────────────────────
    zone=None
    if results.multi_face_landmarks:
        lm=results.multi_face_landmarks[0].landmark
        on_tgt=(get_gaze_zone(lm)==target_zone)
        zone=get_gaze_zone(lm)
        tracking_status,eye_variance,head_variance=get_tracking_status(lm)
        eng.update(lm, on_tgt)

        # ── run adaptive engine every frame (it has internal cooldown) ──
        action=adaptive.evaluate(eng,game)
        if action:
            apply_adaptive_action(action)
    else:
        tracking_status="CALM"

    # ── help mode auto-timeout ────────────────────────────────
    if game.help_mode_active and time.time()-last_help_press>10:
        game.help_mode_active=False
        show_message("🤗 Help mode off\nYou're doing great on your own!",(180,220,255),2.0)

    # ── draw world ────────────────────────────────────────────
    vcfg=game.get_visual_assistance_config()
    draw_target_zones(frame,h,w,zone,vcfg)
    prog=min(1.0,fixation_time/game.target_fix_time) if fixation_time>0 else 0
    draw_focus_mascot(frame,h,w,prog)

    # ── fixation logic ────────────────────────────────────────
    if zone==target_zone:
        if fixation_time==0: fixation_start_time=time.time()
        fixation_time+=0.04
        calculate_gaze_stability(zone,target_zone)
        if fixation_time>=game.target_fix_time:
            acc=calculate_accuracy_score()
            game.performance.add_attempt(True,time.time()-fixation_start_time,acc)
            session_data["successful_attempts"]+=1
            game.total_gems+=1; game.combo+=1
            game.best_combo=max(game.best_combo,game.combo)
            zw=w//game.num_colors; cx=target_zone*zw+zw//2
            create_celebration(cx,250,game.themes[target_zone]["color"])
            if game.combo>=6:   show_message(combo_messages[4],(255,200,100))
            elif game.combo>=2: show_message(combo_messages[min(game.combo-2,3)],(180,220,255))
            else:               show_message(random.choice(success_messages),(100,255,180))
            update_difficulty(); new_target()
            fixation_time=0; fixation_start_time=0
    else:
        if fixation_time>0.3:
            imp=fixation_time<game.target_fix_time*0.4
            if imp: game.impulse_breaks_total+=1
            game.performance.add_attempt(False,0,calculate_accuracy_score(),imp)
            if random.random()>0.3:
                zw=w//game.num_colors; cx=target_zone*zw+zw//2
                create_encouragement(cx,250)
                show_message(random.choice(effort_messages),(180,220,255),1.2)
        if fixation_time>0.2:
            fixation_time=max(0,fixation_time-0.08)
            if fixation_time==0 and game.combo>0:
                if game.attempts_at_level>3:
                    game.combo=0
                    if random.random()>0.5:
                        show_message("💙 Gentle focus…\nYou'll get it next time!",(180,200,255),1.5)
                else:
                    show_message("💛 Almost! Keep trying!",(220,220,150),1.2)
        else:
            fixation_time=0

    # ── particles ─────────────────────────────────────────────
    particles[:] = [p for p in particles if p.update()]
    for p in particles: p.draw(frame)

    # ── overlays ──────────────────────────────────────────────
    draw_instruction_hint(frame,h,w)
    draw_tracking_status(frame,h,w,tracking_status,eye_variance,head_variance)
    draw_level_info_panel(frame,h,w)
    draw_engagement_hud(frame,h,w)   # ← new eye-tracking panel
    draw_hud(frame,h,w)

    if not results.multi_face_landmarks:
        ov=frame.copy()
        cv2.rectangle(ov,(w//2-400,h//2-55),(w//2+400,h//2+55),(30,40,60),-1)
        cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
        draw_emoji(frame, "👀", w//2-340, h//2-12, 30)
        cv2.putText(frame,"Please bring your face closer to the camera",(w//2-310,h//2-8),cv2.FONT_HERSHEY_DUPLEX,1.05,(150,200,255),2)
        cv2.putText(frame,"We need to see your eyes to play!",(w//2-290,h//2+35),cv2.FONT_HERSHEY_SIMPLEX,0.88,(180,220,255),2)

    cv2.imshow(window_name,frame)

    key=cv2.waitKey(1)&0xFF
    if key==27: break
    elif key==ord('r'):
        game=GameLevel(); new_target()
        gaze_history.clear(); head_pos_history.clear()
        gaze_stability_scores.clear(); particles.clear()
        eng=EngagementTracker(); adaptive=AdaptiveEngine()
        show_message("🔄 New adventure!\nFresh focus journey!",(150,220,255))
        game.session_start_time=time.time()
    elif key==ord('h'):
        now=time.time()
        if now-last_help_press>2:
            game.help_mode_active=not game.help_mode_active
            last_help_press=now
            if game.help_mode_active: show_message("🤗 HELP MODE ON!\nExtra guides for 10 s",(100,255,200),3.0)
            else:                     show_message("✨ Help mode off — you're great!",(180,220,255),2.0)
    elif key==ord('f'):
        FULLSCREEN=not FULLSCREEN
        prop=cv2.WINDOW_FULLSCREEN if FULLSCREEN else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,prop)
        if not FULLSCREEN: cv2.resizeWindow(window_name,WINDOW_WIDTH,WINDOW_HEIGHT)
    elif key==32 and show_break_reminder:
        game.session_start_time=time.time()-60

cap.release()
cv2.destroyAllWindows()

# ── Save session data ─────────────────────────────────────────────
saved_path = session_mgr.save(game, eng, adaptive)

# ── Console report ────────────────────────────────────────────────
dur=time.time()-session_data["start_time"]
ta=session_data["total_attempts"]; sa=session_data["successful_attempts"]
sr=(sa/max(1,ta))*100
ic=game.performance.get_impulse_control_score()*100
af=game.performance.get_average_fixation_time()
avg_eng=float(np.mean([s["engagement"] for s in eng.timeline])) if eng.timeline else 0

print("\n"+"="*70)
print("✨ FOCUS QUEST — SESSION REPORT ✨")
print("="*70)
print(f"⏰ Duration : {int(dur//60)} min {int(dur%60)} sec")
print(f"⭐ Stars    : {game.total_gems}   |   🌈 Best Streak: {game.best_combo}")
print(f"📊 ATTENTION METRICS")
print(f"   Focus Success Rate : {sr:.1f}%")
print(f"   Impulse Control    : {ic:.1f}%")
print(f"   Avg Fixation Time  : {af:.2f} s")
print(f"   Avg Engagement     : {avg_eng*100:.1f}%")
print(f"   Total Blinks       : {eng.session_blinks}")
print(f"   Fatigue Events     : {sum(1 for s in eng.timeline if s['fatigue'])}")
print(f"   Distraction Events : {sum(1 for s in eng.timeline if s['distraction'])}")
print(f"   Adaptive Actions   : {len(adaptive.adaptations)}")
print(f"\n📁 Data saved → {saved_path}")
print("="*70)
print("🌟 Every moment of focus is a victory for an ADHD brain! 💙")