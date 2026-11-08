"""Copy the three supplied Java worlds from maps.zip; never overwrite a save."""

import argparse
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

WORLDS = {"Map 1": "RL-Map-1", "map 2": "RL-Map-2", "MAP 3": "RL-Map-3"}
ROOT = Path(__file__).resolve().parents[2]


def import_maps(archive_path: Path, saves: Path) -> list[Path]:
    saves = saves.expanduser().resolve()
    with ZipFile(archive_path.expanduser()) as archive:
        # Validate the whole archive before creating any worlds.
        entries = []
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
                raise ValueError(f"Unsafe archive path: {entry.filename}")
            if not path.parts or path.parts[0] not in WORLDS:
                raise ValueError(f"Unexpected world folder: {entry.filename}")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"Symlink in archive: {entry.filename}")
            entries.append((entry, path))
        for world, folder in WORLDS.items():
            if f"{world}/level.dat" not in archive.namelist():
                raise ValueError(f"Missing level.dat for {world}")
            if (saves / folder).exists():
                raise FileExistsError(f"Save already exists: {saves / folder}")
        saves.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".parkour-import-", dir=saves) as temporary:
            staging = Path(temporary)
            for entry, path in entries:
                # A copied save must acquire its own lock when Minecraft opens it.
                if path.name == "session.lock":
                    continue
                target = staging.joinpath(WORLDS[path.parts[0]], *path.parts[1:])
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(entry) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
            for folder in WORLDS.values():
                (staging / folder).rename(saves / folder)
    return [saves / folder for folder in WORLDS.values()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--saves",
        type=Path,
        default=ROOT / ".local" / "worlds",
        help="Minecraft saves folder; default: .local/worlds in this checkout",
    )
    args = parser.parse_args()
    for path in import_maps(args.archive, args.saves):
        print(path)


if __name__ == "__main__":
    main()
