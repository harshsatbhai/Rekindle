"""Central configuration. Everything is overridable through environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("EDA_DATA_DIR", ROOT / "data"))
PLATFORM_DB = DATA_DIR / "platform.db"      # learner activity (the "LMS"); agent has read-only access
MEMORY_DB = DATA_DIR / "agent_memory.db"    # agent state, persists across runs/sessions
OUTBOX_DIR = DATA_DIR / "outbox"            # emails captured by the local SMTP sink
CALENDAR_DIR = DATA_DIR / "calendar"        # .ics invites produced by schedule_session
RESOURCES_FILE = ROOT / "eda" / "resources.json"

def _load_dotenv(path):
    """Minimal .env loader (KEY=VALUE lines); real environment variables take precedence."""
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

SMTP_HOST = os.getenv("EDA_SMTP_HOST", "127.0.0.1")
SMTP_PORT = int(os.getenv("EDA_SMTP_PORT", "8025"))
SLACK_WEBHOOK_URL = os.getenv("EDA_SLACK_WEBHOOK_URL")  # optional: notify counsellors
SENDER = os.getenv("EDA_SENDER", "learner-success@eda-academy.example")

# Real delivery (e.g. Gmail SMTP with an app password), only for allowlisted recipients.
# Everyone else, including all simulated learners, still goes to the local sink only.
RELAY_HOST = os.getenv("EDA_RELAY_HOST")                     # e.g. smtp.gmail.com
RELAY_PORT = int(os.getenv("EDA_RELAY_PORT", "587"))        # STARTTLS
RELAY_USER = os.getenv("EDA_RELAY_USER")                     # e.g. you@gmail.com
RELAY_PASSWORD = os.getenv("EDA_RELAY_PASSWORD")             # Gmail app password, never your real password
REAL_RECIPIENTS = {e.strip().lower() for e in os.getenv("EDA_REAL_RECIPIENTS", "").split(",") if e.strip()}

LLM_MODEL = os.getenv("EDA_LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
LLM_BASE_URL = os.getenv("EDA_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_API_KEY = os.getenv("NVIDIA_API_KEY")
LLM_TIMEOUT = float(os.getenv("EDA_LLM_TIMEOUT", "45"))
LLM_THINKING = os.getenv("EDA_LLM_THINKING", "off") == "on"  # reasoning adds latency; not needed for short rewrites
LLM_WORKERS = int(os.getenv("EDA_LLM_WORKERS", "4"))
LLM_RPM = float(os.getenv("EDA_LLM_RPM", "35"))           # client-side cap; NVIDIA's free tier allows ~40 requests/min      # learners processed in parallel when the LLM is on
USE_LLM = os.getenv("EDA_USE_LLM", "auto")  # auto | off

DETECTION_DAY = 28          # "today" for the first detection cycle
RISK_THRESHOLD = 0.21
MIN_HISTORY_DAYS = 10       # below this the agent abstains instead of guessing
FOLLOWUP_DAYS = 10          # how long to wait before measuring an intervention
CONTACT_COOLDOWN_DAYS = 7
MAX_ATTEMPTS = 3            # after this many failed interventions -> human escalation
