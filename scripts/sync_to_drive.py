"""
Sync project code, assets, and trained AI models to Google Drive.
"""
import os
import sys
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

# Google Drive candidate paths on Windows
DRIVE_CANDIDATES = [
    Path("G:/My Drive"),
    Path("C:/Users/satet/Google Drive"),
    Path("C:/Users/satet/My Drive"),
]

IGNORED_DIRS = {
    "node_modules",
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    ".system_generated",
}

IGNORED_EXTS = {
    ".pyc",
    ".pyo",
    ".tmp",
    ".log",
}


def find_google_drive() -> Path:
    for candidate in DRIVE_CANDIDATES:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def sync_project_to_drive(workspace_dir: Path = None):
    if workspace_dir is None:
        workspace_dir = Path(__file__).resolve().parent.parent

    drive_root = find_google_drive()
    if not drive_root:
        print("[DriveSync] Google Drive path not found. Please ensure Google Drive is mounted on G: or local folder.")
        return False

    print(f"[DriveSync] Target Google Drive root: {drive_root}")

    # Destination 1: Full project mirror under G:\My Drive\training
    dest_project = drive_root / "training"
    dest_project.mkdir(parents=True, exist_ok=True)
    print(f"[DriveSync] Mirroring project files to: {dest_project}")

    copied_files = 0
    copied_bytes = 0

    for root, dirs, files in os.walk(workspace_dir):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        rel_path = Path(root).relative_to(workspace_dir)

        # Do not copy intermediate checkpoints from runs (only best.pt and last.pt)
        if "runs" in rel_path.parts:
            # Only keep config.yaml, metrics.json, best.pt, last.pt
            allowed_run_files = {"config.yaml", "metrics.json", "best.pt", "last.pt", "events.log"}
            files = [f for f in files if f in allowed_run_files or f.endswith(".pt")]

        target_dir = dest_project / rel_path
        target_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            file_ext = Path(file).suffix.lower()
            if file_ext in IGNORED_EXTS:
                continue

            # Skip intermediate checkpoint files like checkpoint_epoch_1.pt
            if file.startswith("checkpoint_epoch_"):
                continue

            src_file = Path(root) / file
            dst_file = target_dir / file

            # Copy if destination doesn't exist or src is newer
            try:
                if not dst_file.exists() or src_file.stat().st_mtime > dst_file.stat().st_mtime:
                    shutil.copy2(src_file, dst_file)
                    copied_files += 1
                    copied_bytes += src_file.stat().st_size
            except Exception as e:
                print(f"[DriveSync] Warning copying {src_file.name}: {e}")

    print(f"[DriveSync] Synced {copied_files} updated/new files ({copied_bytes / (1024*1024):.2f} MB) to {dest_project}")

    # Destination 2: Sync trained models directly to G:\My Drive\TrainM\KDel_models
    trainm_dir = drive_root / "TrainM"
    if trainm_dir.exists():
        kdel_models_dir = trainm_dir / "KDel_models"
        kdel_models_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DriveSync] Exporting model weights to: {kdel_models_dir}")

        runs_dir = workspace_dir / "data" / "runs"
        if runs_dir.exists():
            for run_path in runs_dir.iterdir():
                if run_path.is_dir():
                    ckpt_dir = run_path / "checkpoints"
                    best_pt = ckpt_dir / "best.pt"
                    if best_pt.exists():
                        target_model_file = kdel_models_dir / f"{run_path.name}_best.pt"
                        if not target_model_file.exists() or best_pt.stat().st_mtime > target_model_file.stat().st_mtime:
                            shutil.copy2(best_pt, target_model_file)
                            print(f"[DriveSync] Copied model weights: {target_model_file.name} ({best_pt.stat().st_size / (1024*1024):.2f} MB)")

    # Destination 3: Create a clean latest zip package in G:\My Drive\TrainM\training_backup_latest.zip
    backup_zip_path = (trainm_dir if trainm_dir.exists() else dest_project) / "training_backup_latest.zip"
    print(f"[DriveSync] Packaging clean backup archive to: {backup_zip_path}")

    try:
        with zipfile.ZipFile(backup_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(workspace_dir):
                dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and "runs" not in d]
                rel_root = Path(root).relative_to(workspace_dir)

                for file in files:
                    if Path(file).suffix.lower() in IGNORED_EXTS:
                        continue
                    if file.endswith(".db") or file.endswith(".sqlite"):
                        continue
                    file_path = Path(root) / file
                    zf.write(file_path, arcname=str(rel_root / file))
        print(f"[DriveSync] Successfully created {backup_zip_path.name} ({backup_zip_path.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as ze:
        print(f"[DriveSync] Warning creating backup zip: {ze}")

    # Write a sync status log file
    status_file = dest_project / "SYNC_STATUS.txt"
    with open(status_file, "w", encoding="utf-8") as sf:
        sf.write(f"Last Google Drive Sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        sf.write(f"Source Workspace: {workspace_dir}\n")
        sf.write(f"Google Drive Destination: {dest_project}\n")
        sf.write("Sync Status: OK\n")

    print("[DriveSync] Google Drive synchronization completed successfully!")
    return True


if __name__ == "__main__":
    sync_project_to_drive()
