"""Central configuration. Everything here is overridable by environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BASELINES = ROOT / "baselines"
RESULTS = ROOT / "results"

# Load .env if present, so the API key lives in a gitignored file rather than in
# your shell profile. Optional dependency: everything still works without it as
# long as the environment is set some other way.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

# --- models ----------------------------------------------------------------
# The agent under test runs on Haiku 4.5: it is the cheap, fast tier and this is
# the model we call hundreds of times per eval run. The judge runs on Opus 5 --
# the grader must be at least as capable as the thing it grades, or the scores
# are noise. Override either with AGENT_MODEL / JUDGE_MODEL.
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-haiku-4-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-opus-5")

# Replies go out over SMS/WhatsApp, so they are deliberately short.
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "1024"))
JUDGE_MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "2000"))

# NOTE: we deliberately never pass `temperature`. Sampling params are rejected
# (400) on Opus 5 / Sonnet 5 and the 4.6+ family, so omitting them keeps
# AGENT_MODEL swappable to any current model without code changes.

# --- run shape -------------------------------------------------------------
# A single run of a non-deterministic system proves nothing. Every case is run
# REPETITIONS times and scored on pass *rate*.
REPETITIONS = int(os.getenv("EVAL_REPETITIONS", "3"))

# A case passes overall when at least this fraction of its runs pass.
PASS_RATE_THRESHOLD = float(os.getenv("PASS_RATE_THRESHOLD", "1.0"))

# Judge scores are 1-5 per dimension; this is the minimum acceptable.
JUDGE_MIN_SCORE = int(os.getenv("JUDGE_MIN_SCORE", "4"))

# How far the suite pass rate may fall below baseline before CI fails.
BASELINE_TOLERANCE = float(os.getenv("BASELINE_TOLERANCE", "0.02"))

# --- offline mode ----------------------------------------------------------
# EVAL_FAKE=1 runs the agent from recorded fixtures instead of the API. This is
# what CI uses on every push: the deterministic graders are exercised for free
# and offline. The judged run is a separate, scheduled job.
FAKE = os.getenv("EVAL_FAKE", "0") == "1"

MAX_REPLY_CHARS = int(os.getenv("MAX_REPLY_CHARS", "480"))
