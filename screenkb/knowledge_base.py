"""Knowledge Base — SQLite storage for patterns, rules, and corrections."""

from __future__ import annotations

import json
import os
import platform
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# ---------------------------------------------------------------------------
# DB location
# ---------------------------------------------------------------------------

def _default_db_dir() -> Path:
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("APPDATA", str(Path.home()))
    else:
        base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    p = Path(base) / "screenkb"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_db_path() -> Path:
    override = os.environ.get("SCREENKB_DB")
    if override:
        return Path(override)
    return _default_db_dir() / "screenkb.db"


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    screen_type TEXT NOT NULL,
    feature_fingerprint TEXT NOT NULL,   -- JSON array of features, sorted
    confidence_boost REAL DEFAULT 0.1,
    source TEXT DEFAULT 'rule',          -- 'rule' | 'learn' | 'llm'
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    screen_type TEXT NOT NULL,
    required_features TEXT NOT NULL,     -- JSON array — ALL must be present
    bonus_features TEXT DEFAULT '[]',    -- JSON array — each adds to confidence
    base_confidence REAL DEFAULT 0.5,
    description TEXT DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_hash TEXT NOT NULL,
    screen_type TEXT NOT NULL,
    feature_fingerprint TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    screen_type TEXT NOT NULL,
    ocr_snippet TEXT DEFAULT '',
    features TEXT NOT NULL,              -- JSON array
    confidence REAL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_patterns_screen_type ON patterns(screen_type);
CREATE INDEX IF NOT EXISTS idx_rules_screen_type ON rules(screen_type);
CREATE INDEX IF NOT EXISTS idx_corrections_hash ON corrections(image_hash);
"""

_CURRENT_VERSION = 2


def init_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (or create) the KB database and apply schema migrations."""
    path = db_path or get_db_path()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _apply_migrations(conn)
    return conn


def _apply_migrations(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_SQL)
    row = conn.execute("SELECT MAX(version) as v FROM schema_version").fetchone()
    current = row["v"] if row["v"] is not None else 0

    if current < 1:
        _seed_default_rules(conn)
        conn.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
            (1, _now()),
        )
        conn.commit()

    if current < 2:
        _seed_v2_rules(conn)
        conn.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
            (2, _now()),
        )
        conn.commit()


def _now() -> str:
    return datetime.utcnow().isoformat()


# ---------------------------------------------------------------------------
# Default rules seed
# ---------------------------------------------------------------------------

def _seed_default_rules(conn: sqlite3.Connection) -> None:
    """Populate with default IF-THEN rules."""
    default_rules = [
        # Python errors
        {
            "screen_type": "python_error",
            "required": ["traceback", "python_error"],
            "bonus": ["terminal", "editor"],
            "base": 0.85,
            "description": "Python exception in terminal or editor",
        },
        # Generic traceback
        {
            "screen_type": "stack_trace",
            "required": ["stack_trace"],
            "bonus": ["terminal", "editor", "javascript_error"],
            "base": 0.60,
            "description": "Any language stack trace",
        },
        # JavaScript error
        {
            "screen_type": "javascript_error",
            "required": ["javascript_error"],
            "bonus": ["browser", "terminal"],
            "base": 0.75,
            "description": "JavaScript runtime error",
        },
        # HTTP error page
        {
            "screen_type": "http_error",
            "required": ["http_error"],
            "bonus": ["browser"],
            "base": 0.70,
            "description": "HTTP error response (4xx/5xx)",
        },
        # Login form
        {
            "screen_type": "login_form",
            "required": ["login_form"],
            "bonus": ["form", "browser"],
            "base": 0.80,
            "description": "Authentication / login screen",
        },
        # Dashboard
        {
            "screen_type": "dashboard",
            "required": ["navigation", "sidebar"],
            "bonus": ["table", "content"],
            "base": 0.55,
            "description": "Application dashboard with navigation",
        },
        # IDE / code editor
        {
            "screen_type": "code_editor",
            "required": ["editor"],
            "bonus": ["sidebar", "terminal", "toolbar"],
            "base": 0.65,
            "description": "Code editor view",
        },
        # VS Code
        {
            "screen_type": "vscode",
            "required": ["vscode"],
            "bonus": ["editor", "terminal", "sidebar"],
            "base": 0.85,
            "description": "Visual Studio Code",
        },
        # Terminal / CLI
        {
            "screen_type": "terminal",
            "required": ["terminal"],
            "bonus": ["git"],
            "base": 0.70,
            "description": "Terminal or CLI window",
        },
        # API response / Postman
        {
            "screen_type": "api_response",
            "required": ["api_response"],
            "bonus": ["network_log"],
            "base": 0.72,
            "description": "API JSON response viewer",
        },
        # Test runner
        {
            "screen_type": "test_runner",
            "required": ["test_runner"],
            "bonus": ["terminal", "editor"],
            "base": 0.78,
            "description": "Test suite output",
        },
        # Form
        {
            "screen_type": "form",
            "required": ["form"],
            "bonus": ["modal", "browser"],
            "base": 0.60,
            "description": "Generic web form",
        },
        # Browser
        {
            "screen_type": "browser",
            "required": ["browser"],
            "bonus": ["navigation", "content"],
            "base": 0.50,
            "description": "Web browser window",
        },
    ]

    now = _now()
    for r in default_rules:
        conn.execute(
            """
            INSERT INTO rules(screen_type, required_features, bonus_features,
                              base_confidence, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                r["screen_type"],
                json.dumps(r["required"]),
                json.dumps(r["bonus"]),
                r["base"],
                r["description"],
                now,
            ),
        )


# ---------------------------------------------------------------------------
# V2 seed — additional deterministic rules
# ---------------------------------------------------------------------------

def _seed_v2_rules(conn: sqlite3.Connection) -> None:
    """V2: higher-specificity rules for French UIs and common patterns."""
    v2_rules = [
        # French login form — specific to French-language admin/webapp UIs
        {
            "screen_type": "login_form",
            "required": ["login_form", "sidebar"],
            "bonus": ["form", "browser"],
            "base": 0.92,
            "description": "French admin/webapp login (email + mot de passe)",
        },
        # Kanban / board view
        {
            "screen_type": "kanban",
            "required": ["navigation", "content"],
            "bonus": ["sidebar"],
            "base": 0.55,
            "description": "Kanban board or card-based task view",
        },
        # Stats / analytics page
        {
            "screen_type": "statistics",
            "required": ["navigation", "sidebar"],
            "bonus": ["content"],
            "base": 0.50,
            "description": "Statistics or analytics dashboard",
        },
    ]
    now = _now()
    for r in v2_rules:
        conn.execute(
            """
            INSERT INTO rules(screen_type, required_features, bonus_features,
                              base_confidence, description, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                r["screen_type"],
                json.dumps(r["required"]),
                json.dumps(r["bonus"]),
                r["base"],
                r["description"],
                now,
            ),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_rules(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute("SELECT * FROM rules ORDER BY base_confidence DESC").fetchall()
    result = []
    for row in rows:
        result.append({
            "id": row["id"],
            "screen_type": row["screen_type"],
            "required_features": json.loads(row["required_features"]),
            "bonus_features": json.loads(row["bonus_features"]),
            "base_confidence": row["base_confidence"],
            "description": row["description"],
        })
    return result


def save_correction(
    conn: sqlite3.Connection,
    image_hash: str,
    screen_type: str,
    features: List[str],
) -> None:
    fingerprint = json.dumps(sorted(features))
    conn.execute(
        """
        INSERT OR REPLACE INTO corrections(image_hash, screen_type, feature_fingerprint, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (image_hash, screen_type, fingerprint, _now()),
    )
    # Also add as a pattern for future matching
    conn.execute(
        """
        INSERT INTO patterns(screen_type, feature_fingerprint, confidence_boost, source, created_at)
        VALUES (?, ?, 0.15, 'learn', ?)
        """,
        (screen_type, fingerprint, _now()),
    )
    conn.commit()


def lookup_correction(conn: sqlite3.Connection, image_hash: str) -> Optional[str]:
    row = conn.execute(
        "SELECT screen_type FROM corrections WHERE image_hash = ?", (image_hash,)
    ).fetchone()
    return row["screen_type"] if row else None


def find_matching_patterns(
    conn: sqlite3.Connection, features: List[str]
) -> List[tuple[str, float]]:
    """Return (screen_type, boost) pairs whose fingerprint is a subset of features."""
    feature_set = set(features)
    rows = conn.execute(
        "SELECT screen_type, feature_fingerprint, confidence_boost FROM patterns"
    ).fetchall()
    matches = []
    for row in rows:
        pattern_features = set(json.loads(row["feature_fingerprint"]))
        if pattern_features.issubset(feature_set):
            matches.append((row["screen_type"], row["confidence_boost"]))
    return matches


def save_example(
    conn: sqlite3.Connection,
    screen_type: str,
    ocr_snippet: str,
    features: List[str],
    confidence: float,
) -> None:
    conn.execute(
        """
        INSERT INTO examples(screen_type, ocr_snippet, features, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (screen_type, ocr_snippet[:500], json.dumps(features), confidence, _now()),
    )
    conn.commit()
