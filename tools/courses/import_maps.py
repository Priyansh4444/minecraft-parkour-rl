"""Import the downloaded Minecraft worlds without overwriting existing saves."""

import argparse
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo

WORLD_FOLDERS = {
    "Map 1": "RL-Map-1",
    "map 2": "RL-Map-2",
    "MAP 3": "RL-Map-3",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ArchiveEntry = tuple[ZipInfo, PurePosixPath]


def read_world_entries(archive: ZipFile) -> list[ArchiveEntry]:
    """Read and validate every file path before copying anything."""
    entries: list[ArchiveEntry] = []

    for file_info in archive.infolist():
        relative_path = PurePosixPath(file_info.filename)

        # reject paths that could escape the destination folder
        if relative_path.is_absolute() or ".." in relative_path.parts or "\\" in file_info.filename:
            raise ValueError(f"Unsafe archive path: {file_info.filename}")

        if not relative_path.parts or relative_path.parts[0] not in WORLD_FOLDERS:
            raise ValueError(f"Unexpected world folder: {file_info.filename}")

        if stat.S_ISLNK(file_info.external_attr >> 16):
            raise ValueError(f"Symlink in archive: {file_info.filename}")

        entries.append((file_info, relative_path))

    for world_name in WORLD_FOLDERS:
        if f"{world_name}/level.dat" not in archive.namelist():
            raise ValueError(f"Missing level.dat for {world_name}")

    return entries


def copy_world_files(archive: ZipFile, entries: list[ArchiveEntry], staging_folder: Path) -> None:
    """Copy validated files into a temporary folder."""
    for file_info, relative_path in entries:
        # each copied world needs a fresh lock when minecraft opens it
        if relative_path.name == "session.lock":
            continue

        world_name, *world_path = relative_path.parts
        destination = staging_folder.joinpath(WORLD_FOLDERS[world_name], *world_path)

        if file_info.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)

        with archive.open(file_info) as source_file, destination.open("wb") as destination_file:
            shutil.copyfileobj(source_file, destination_file)


def import_maps(archive_path: Path, saves_folder: Path) -> list[Path]:
    """Validate the ZIP, copy its worlds, then publish the finished copies."""
    saves_folder = saves_folder.expanduser().resolve()
    destinations = [saves_folder / folder_name for folder_name in WORLD_FOLDERS.values()]

    with ZipFile(archive_path.expanduser()) as archive:
        entries = read_world_entries(archive)

        # check all destinations before creating or copying any worlds
        for destination in destinations:
            if destination.exists():
                raise FileExistsError(f"Save already exists: {destination}")

        saves_folder.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix=".parkour-import-", dir=saves_folder) as temporary:
            staging_folder = Path(temporary)
            copy_world_files(archive, entries, staging_folder)

            # move complete worlds into place only after copying succeeds
            for destination in destinations:
                (staging_folder / destination.name).rename(destination)

    return destinations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--saves", type=Path, default=PROJECT_ROOT / ".local/worlds")
    arguments = parser.parse_args()

    for world_folder in import_maps(arguments.archive, arguments.saves):
        print(world_folder)


if __name__ == "__main__":
    main()
