"""Archive import boundaries and preservation of existing saves."""

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from tools.courses import import_maps as module


class ImportTests(unittest.TestCase):
    def archive(self, directory, extra=None):
        path = Path(directory) / "maps.zip"
        with ZipFile(path, "w") as archive:
            for world in module.WORLDS:
                archive.writestr(f"{world}/level.dat", b"original metadata")
                archive.writestr(f"{world}/session.lock", b"old lock")
                archive.writestr(
                    f"{world}/dimensions/minecraft/overworld/region/r.0.0.mca", b"blocks"
                )
            if extra:
                archive.writestr(extra, b"unexpected")
        return path

    def test_copies_nested_worlds_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.archive(directory)
            original = archive.read_bytes()
            saves = Path(directory) / "saves"
            worlds = module.import_maps(archive, saves)
            self.assertEqual(len(worlds), 3)
            for world in worlds:
                self.assertEqual((world / "level.dat").read_bytes(), b"original metadata")
                self.assertFalse((world / "session.lock").exists())
                self.assertEqual(
                    (world / "dimensions/minecraft/overworld/region/r.0.0.mca").read_bytes(),
                    b"blocks",
                )
            (worlds[0] / "level.dat").write_bytes(b"player progress")
            with self.assertRaises(FileExistsError):
                module.import_maps(archive, saves)
            self.assertEqual((worlds[0] / "level.dat").read_bytes(), b"player progress")
            self.assertEqual(archive.read_bytes(), original)

    def test_rejects_paths_before_creating_any_saves(self):
        for extra in ("Map 1/../../escape", "/absolute", "Map 1\\escape", "Unexpected/level.dat"):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as directory:
                archive = self.archive(directory, extra)
                saves = Path(directory) / "saves"
                with self.assertRaises(ValueError):
                    module.import_maps(archive, saves)
                self.assertFalse(saves.exists())


if __name__ == "__main__":
    unittest.main()
