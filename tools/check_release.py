"""Reject data, binaries and accidental credentials from the tracked release tree."""

from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {"README.md", "LICENSE", "pyproject.toml", ".gitignore", ".env.example"}
SOURCE_DIRS = {"src", "examples", "tests", "tools"}
PATTERNS = {
    "access token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9_-]{25,})"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "embedded image": re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]{40,}"),
    "private IPv4": re.compile(r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"),
    "dataset subject/study identifier": re.compile(r"\b[ps][0-9]{8}\b"),
    "Windows data path": re.compile(r"[A-Za-z]:[\\/](?:spl|Users|data)[\\/]", re.I),
}


def allowed_file(name):
    path = PurePosixPath(name)
    if name in TOP_LEVEL:
        return True
    if path.parts[0] in SOURCE_DIRS and path.suffix == ".py":
        return True
    if path.parts[0] == "docs" and len(path.parts) == 2 and path.suffix == ".md":
        return True
    return path.parts[:2] == (".github", "workflows") and len(path.parts) == 3 and path.suffix == ".yml"


def check(paths):
    errors = []
    for name in paths:
        if not allowed_file(name):
            errors.append(f"{name}: not an approved source/documentation file")
            continue
        file = ROOT / name
        if file.is_symlink():
            errors.append(f"{name}: symlinks are excluded")
            continue
        raw = file.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"{name}: binary content")
            continue
        if b"\0" in raw or len(raw) > 150_000:
            errors.append(f"{name}: binary/oversize content")
        for reason, pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{name}: possible {reason}")
    return errors


def main():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
    paths = [name for name in result.stdout.decode().split("\0") if name]
    if not paths:
        print("No tracked files; stage the reviewed release first", file=sys.stderr)
        return 1
    errors = check(paths)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Release check passed: {len(paths)} text source/documentation files; no bundled data/binaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
