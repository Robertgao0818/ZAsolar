"""Colab-specific path configuration for SA_Solar.

Handles Google Drive mounting and symlink creation so that existing scripts
work without modification on Colab.

Usage (in Colab notebook):
    from scripts.colab_config import setup_colab
    setup_colab()  # mounts Drive, creates symlinks
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# Default Drive folder where large files (tiles, checkpoints) are stored
DRIVE_DATA_DIR = "SA_Solar_Data"


def is_colab() -> bool:
    """Return True if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def mount_drive(force: bool = False) -> Path:
    """Mount Google Drive and return the mount point."""
    from google.colab import drive  # noqa: F811
    mount_point = "/content/drive"
    if force or not Path(mount_point, "MyDrive").exists():
        drive.mount(mount_point, force_remount=force)
    return Path(mount_point, "MyDrive")


def setup_colab(
    drive_data_dir: str = DRIVE_DATA_DIR,
    mount: bool = True,
) -> dict[str, Path]:
    """Set up Colab environment for SA_Solar.

    1. Adds project root to sys.path
    2. Mounts Google Drive (if mount=True)
    3. Creates symlinks: tiles/ → Drive, checkpoints/ → Drive
    4. Sets environment variables

    Args:
        drive_data_dir: Folder name under MyDrive for large files.
        mount: Whether to mount Google Drive.

    Returns:
        Dict with resolved paths.
    """
    project_root = get_project_root()

    # Ensure project root is on sys.path
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # Set environment variables
    os.environ["MPLCONFIGDIR"] = str(project_root / ".cache" / "matplotlib")
    os.environ["JOBLIB_TEMP_FOLDER"] = str(project_root / ".tmp" / "joblib")
    os.environ["XDG_CACHE_HOME"] = str(project_root / ".cache")

    paths = {"project_root": project_root}

    if mount and is_colab():
        drive_root = mount_drive()
        data_dir = drive_root / drive_data_dir

        # Create Drive data directory structure if needed
        for sub in ["tiles", "checkpoints", "results"]:
            (data_dir / sub).mkdir(parents=True, exist_ok=True)

        # Symlink large-file directories to Drive
        for name in ["tiles", "checkpoints", "results"]:
            local_path = project_root / name
            drive_path = data_dir / name
            if local_path.is_symlink():
                local_path.unlink()
            if not local_path.exists():
                local_path.symlink_to(drive_path)
                print(f"  {name}/ → {drive_path}")

        paths["drive_data"] = data_dir
        os.environ["SOLAR_TILES_ROOT"] = str(data_dir / "tiles")

    return paths
