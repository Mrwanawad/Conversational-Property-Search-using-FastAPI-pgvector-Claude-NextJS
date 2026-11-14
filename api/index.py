"""Vercel serverless entry point. All paths are rewritten here (vercel.json); FastAPI routes on /api/*.

The database is seeded once ahead of time (`python -m propsearch.seed` against DATABASE_URL), so cold
starts only open a connection and build the matcher.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from propsearch.api import create_app  # noqa: E402

app = create_app(seed=False)
