import sqlite3
import uuid
import datetime
import json
from rich.console import Console

import os

console = Console()
DB_FILE = os.path.join("logs", "htmlminer_logs.db")

def init_db():
    """
    Initializes the SQLite database and creates the logs table if it doesn't exist.
    """
    try:
        os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                component TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                data TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                engine TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                content TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extractions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                url TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                data TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS page_relevance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                base_url TEXT NOT NULL,
                page_url TEXT NOT NULL,
                feature TEXT NOT NULL,
                relevance_score INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS step_timings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                url TEXT,
                duration_seconds REAL NOT NULL,
                details TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        console.print(f"[red]Failed to initialize database: {e}[/red]")

def create_session() -> str:
    """
    Creates a unique session ID.
    """
    return str(uuid.uuid4())


def log_event(session_id: str, component: str, level: str, message: str, data: dict = None):
    """
    Logs an event to the database.
    """
    try:
        data_json = json.dumps(data) if data else None
        with sqlite3.connect(DB_FILE, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO logs (session_id, component, level, message, data)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, component, level, message, data_json))
            # conn.commit() is called automatically on successful exit
    except Exception as e:
        # Silently fail or simple print to avoid interrupting flow
        print(f"Log error: {e}")


def log_step_timing(
    session_id: str,
    step_name: str,
    duration_seconds: float,
    url: str = None,
    details: dict = None,
):
    """
    Logs step timing for performance debugging.
    
    Args:
        session_id: The current session ID
        step_name: Name of the step (e.g., 'page_selection', 'extraction', 'synthesis')
        duration_seconds: How long the step took
        url: Optional URL being processed
        details: Optional dict with additional info (feature name, counts, etc.)
    """
    try:
        details_json = json.dumps(details) if details else None
        with sqlite3.connect(DB_FILE, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO step_timings (session_id, step_name, url, duration_seconds, details)
                VALUES (?, ?, ?, ?, ?)
            ''', (session_id, step_name, url, duration_seconds, details_json))
    except Exception as e:
        print(f"Step timing log error: {e}")


def save_snapshot(url: str, engine: str, content: str):
    """
    Saves a snapshot to the database.
    """
    try:
        with sqlite3.connect(DB_FILE, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO snapshots (url, engine, content)
                VALUES (?, ?, ?)
            ''', (url, engine, content))
    except Exception as e:
        console.print(f"[red]Failed to save snapshot: {e}[/red]")

def get_latest_snapshot(url: str, engine: str):
    """
    Retrieves the latest snapshot for a URL and engine.
    Returns tuple (content, timestamp) or None.
    """
    try:
        with sqlite3.connect(DB_FILE, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT content, timestamp FROM snapshots
                WHERE url = ? AND engine = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (url, engine))
            row = cursor.fetchone()
        return row
    except Exception as e:
        console.print(f"[red]Failed to get snapshot: {e}[/red]")
        return None

def save_extractions(extractions: list, session_id: str = None):
    """
    Saves a list of extraction dictionaries to the database.
    """
    try:
        with sqlite3.connect(DB_FILE, timeout=30.0) as conn:
            cursor = conn.cursor()
            
            for extraction in extractions:
                url = extraction.get("URL", "unknown")
                data_json = json.dumps(extraction)
                cursor.execute('''
                    INSERT INTO extractions (session_id, url, data)
                    VALUES (?, ?, ?)
                ''', (session_id, url, data_json))
        
        console.print(f"[green]Saved {len(extractions)} extractions to {DB_FILE}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to save extractions to DB: {e}[/red]")


def save_page_relevance(
    session_id: str,
    base_url: str,
    relevance_scores: dict,
):
    """
    Saves page relevance scores to the database.
    
    Args:
        session_id: The current session ID
        base_url: The base URL being analyzed
        relevance_scores: Dict mapping page_url -> {feature: score}
    """
    if not relevance_scores:
        return
        
    try:
        with sqlite3.connect(DB_FILE, timeout=30.0) as conn:
            cursor = conn.cursor()
            
            for page_url, feature_scores in relevance_scores.items():
                for feature, score in feature_scores.items():
                    cursor.execute('''
                        INSERT INTO page_relevance (session_id, base_url, page_url, feature, relevance_score)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (session_id, base_url, page_url, feature, score))
    except Exception as e:
        console.print(f"[red]Failed to save page relevance: {e}[/red]")



def get_token_usage_report(session_id: str) -> dict:
    """
    Aggregate token usage for a session from step_timings.

    Prefers per-call token logs; falls back to per-step totals if needed.
    """
    if not session_id or not os.path.exists(DB_FILE):
        return {"steps": [], "totals": {}, "source": None}

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT step_name, duration_seconds, details
            FROM step_timings
            WHERE session_id = ?
            """,
            (session_id,),
        )
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        console.print(f"[red]Failed to read token usage: {e}[/red]")
        return {"steps": [], "totals": {}, "source": None}

    def _as_int(value) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _as_float(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    direct_rows = []
    nested_rows = []

    for step_name, duration_seconds, details_json in rows:
        if not details_json:
            continue
        try:
            details = json.loads(details_json)
        except Exception:
            continue
        if not isinstance(details, dict):
            continue

        if "prompt_tokens" in details or "completion_tokens" in details:
            direct_rows.append(
                (
                    step_name,
                    _as_float(duration_seconds),
                    _as_int(details.get("prompt_tokens")),
                    _as_int(details.get("completion_tokens")),
                    1,
                )
            )
            continue

        tokens = details.get("tokens")
        if isinstance(tokens, dict):
            calls = tokens.get("total_calls")
            nested_rows.append(
                (
                    step_name,
                    _as_float(duration_seconds),
                    _as_int(tokens.get("prompt_tokens")),
                    _as_int(tokens.get("completion_tokens")),
                    _as_int(calls) if calls is not None else 1,
                )
            )

    rows_to_use = direct_rows if direct_rows else nested_rows
    if not rows_to_use:
        return {"steps": [], "totals": {}, "source": None}

    by_step = {}
    for step_name, duration, prompt, completion, calls in rows_to_use:
        entry = by_step.setdefault(
            step_name,
            {
                "step_name": step_name,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
                "duration_seconds": 0.0,
            },
        )
        entry["prompt_tokens"] += prompt
        entry["completion_tokens"] += completion
        entry["total_tokens"] = entry["prompt_tokens"] + entry["completion_tokens"]
        entry["calls"] += calls
        entry["duration_seconds"] += duration

    steps = sorted(by_step.values(), key=lambda item: item["total_tokens"], reverse=True)

    totals = {
        "prompt_tokens": sum(step["prompt_tokens"] for step in steps),
        "completion_tokens": sum(step["completion_tokens"] for step in steps),
        "total_tokens": sum(step["total_tokens"] for step in steps),
        "calls": sum(step["calls"] for step in steps),
        "duration_seconds": sum(step["duration_seconds"] for step in steps),
    }

    source = "per_call" if direct_rows else "step_totals"
    return {"steps": steps, "totals": totals, "source": source}
