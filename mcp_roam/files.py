"""File system access for org-roam directory."""

import tempfile
from pathlib import Path

from mcp_roam.interfaces import FileAccess


class RoamFileAccess(FileAccess):
    """File system operations for the org-roam directory."""

    def __init__(self, roam_dir: str | Path):
        self.roam_dir = Path(roam_dir).resolve()

    def resolve_path(self, filename: str) -> Path:
        """Resolve a filename to an absolute path in the roam directory."""
        if filename.startswith('/'):
            return Path(filename)
        return self.roam_dir / filename

    def read_file(self, path: Path | str) -> str:
        """Read file content."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return p.read_text(encoding='utf-8', errors='replace')

    def write_file(self, path: Path | str, content: str) -> None:
        """Write file content (atomic write via temp + rename)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp = tempfile.mkstemp(
            dir=str(p.parent),
            prefix='.mcp-roam-',
            suffix='.org'
        )
        try:
            with open(fd, 'w', encoding='utf-8') as f:
                f.write(content)
            Path(tmp).rename(p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def append_file(self, path: Path | str, content: str) -> None:
        """Append content to a file."""
        p = Path(path)
        if not p.exists():
            self.write_file(p, content)
            return
        with open(p, 'a', encoding='utf-8') as f:
            f.write(content)

    def daily_path(self, date: str) -> Path:
        """Get path for a daily note file (YYYY-MM-DD.org)."""
        return self.roam_dir / 'daily' / f'{date}.org'

    def exists(self, path: Path | str) -> bool:
        """Check if a file exists."""
        return Path(path).exists()

    def list_files(
        self,
        directory: Path | str,
        pattern: str = '*'
    ) -> list[Path]:
        """List files in a directory matching a glob pattern."""
        d = Path(directory)
        if not d.exists():
            return []
        return sorted(d.glob(pattern))