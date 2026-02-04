"""Version information for HTMLMiner."""

import subprocess
from pathlib import Path

# Base semantic version - bump this for releases
__version__ = "0.1.0"


def get_git_info() -> dict:
    """Get git commit hash and other info."""
    try:
        # Get the directory where this file is located
        pkg_dir = Path(__file__).parent

        # Get short commit hash
        commit_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=pkg_dir,
            timeout=5,
        )

        # Check if there are uncommitted changes
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=pkg_dir,
            timeout=5,
        )

        # Get commit count
        commit_count = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            capture_output=True,
            text=True,
            cwd=pkg_dir,
            timeout=5,
        )

        return {
            "hash": commit_hash.stdout.strip() if commit_hash.returncode == 0 else None,
            "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else False,
            "count": int(commit_count.stdout.strip()) if commit_count.returncode == 0 else None,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {"hash": None, "dirty": False, "count": None}


def get_version() -> str:
    """Get the full version string including git info."""
    git_info = get_git_info()

    version = __version__

    if git_info["hash"]:
        version = f"{version}+{git_info['hash']}"
        if git_info["dirty"]:
            version = f"{version}.dirty"

    return version


def get_version_info() -> dict:
    """Get detailed version information."""
    git_info = get_git_info()

    return {
        "version": __version__,
        "full_version": get_version(),
        "git_hash": git_info["hash"],
        "git_dirty": git_info["dirty"],
        "commit_count": git_info["count"],
    }


# For convenience
VERSION = get_version()
VERSION_INFO = get_version_info()
