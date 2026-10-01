"""
Initialize Git repository and prepare for GitHub push using Dulwich (pure-python Git).
"""
import os
import sys
from pathlib import Path
from dulwich import porcelain
from dulwich.repo import Repo

project_dir = Path(r"c:\Users\tahab\OneDrive\Desktop\Secret Project")

print(f"Initializing Git repository in {project_dir}...")
git_dir = project_dir / ".git"

if not git_dir.exists():
    repo = porcelain.init(str(project_dir))
    print("  -> Initialized new Git repository (.git created)")
else:
    repo = Repo(str(project_dir))
    print("  -> Existing Git repository loaded")

# Add all files respecting .gitignore
print("Staging project files...")
porcelain.add(str(project_dir), paths=None)

# Create commit
status = porcelain.status(str(project_dir))
staged_count = len(status.staged["add"]) + len(status.staged["modify"])
print(f"Staged {staged_count} files for commit.")

try:
    commit_id = porcelain.commit(
        str(project_dir),
        message=b"Initial commit: Multi-Omics Neuroplasticity AI Pipeline & In Silico Simulator",
        committer=b"Taha <taha@users.noreply.github.com>",
        author=b"Taha <taha@users.noreply.github.com>"
    )
    print(f"Committed successfully! Commit ID: {commit_id.decode() if isinstance(commit_id, bytes) else commit_id}")
except Exception as e:
    print(f"Commit note: {e}")

# Set remote origin
remote_url = "https://github.com/Taha0428/Projects.git"
try:
    config = repo.get_config()
    config.set((b"remote", b"origin"), b"url", remote_url.encode("utf-8"))
    config.write_to_path()
    print(f"Remote 'origin' set to: {remote_url}")
except Exception as e:
    print(f"Remote configuration note: {e}")

print("\nGit repository successfully prepared locally!")
