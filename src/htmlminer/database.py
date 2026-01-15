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
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        data_json = json.dumps(data) if data else None
        
        cursor.execute('''
            INSERT INTO logs (session_id, component, level, message, data)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, component, level, message, data_json))
        
        conn.commit()
        conn.close()
    except Exception as e:
        # Silently fail or simple print to avoid interrupting flow
        print(f"Log error: {e}")

def save_snapshot(url: str, engine: str, content: str):
    """
    Saves a snapshot to the database.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO snapshots (url, engine, content)
            VALUES (?, ?, ?)
        ''', (url, engine, content))
        conn.commit()
        conn.close()
    except Exception as e:
        console.print(f"[red]Failed to save snapshot: {e}[/red]")

def get_latest_snapshot(url: str, engine: str):
    """
    Retrieves the latest snapshot for a URL and engine.
    Returns tuple (content, timestamp) or None.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT content, timestamp FROM snapshots
            WHERE url = ? AND engine = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (url, engine))
        row = cursor.fetchone()
        conn.close()
        return row
    except Exception as e:
        console.print(f"[red]Failed to get snapshot: {e}[/red]")
        return None

def save_extractions(extractions: list, session_id: str = None):
    """
    Saves a list of extraction dictionaries to the database.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        for extraction in extractions:
            url = extraction.get("URL", "unknown")
            data_json = json.dumps(extraction)
            cursor.execute('''
                INSERT INTO extractions (session_id, url, data)
                VALUES (?, ?, ?)
            ''', (session_id, url, data_json))
            
        conn.commit()
        conn.close()
        console.print(f"[green]Saved {len(extractions)} extractions to {DB_FILE}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to save extractions to DB: {e}[/red]")
