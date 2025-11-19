from pathlib import Path

def get_project_root() -> Path:
    """
    Detect and return the project root directory dynamically.
    Uses the presence of the `.git` folder as anchor.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent

    # Fallback: return folder containing this file
    return current.parent
