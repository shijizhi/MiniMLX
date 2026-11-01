import importlib.util
from pathlib import Path

import tempfile
import unittest

spec = importlib.util.spec_from_file_location("release_version", Path(__file__).resolve().parents[1] / "tools/version.py")
release_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_version)


def fixture_root(tmp_path):
    (tmp_path / "Cargo.toml").write_text('[workspace.package]\nedition = "2021"\nversion = "0.1.0"\n')
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\nversion = "0.1.0"\n')
    (tmp_path / "Cargo.lock").write_text('\n'.join(f'[[package]]\nname = "{name}"\nversion = "0.1.0"\n' for name in release_version.PACKAGES))
    runtime = tmp_path / "python/minimlx_runtime"
    runtime.mkdir(parents=True)
    (runtime / "_version.py").write_text('__version__ = "0.1.0"\n')
    return tmp_path


class ReleaseVersionTests(unittest.TestCase):
    def test_version_update_synchronizes_release_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = fixture_root(Path(directory))
            self.assertEqual(release_version.set_version("0.2.1", root), "0.2.1")
            self.assertEqual(release_version.check(root), "0.2.1")

    def test_version_check_rejects_mismatch_and_invalid_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = fixture_root(Path(directory))
            with self.assertRaises(ValueError):
                release_version.set_version("1.2.3-beta", root)
            (root / "pyproject.toml").write_text('[project]\nversion = "0.2.0"\n')
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                release_version.check(root)
