import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import setup_exp57_env as setup


class PersistentEnvironmentTests(unittest.TestCase):
    def test_driver_boundaries_select_official_compatible_builds(self):
        cases = [("11.8", "2.6.0", "cu118"), ("12.0", "2.6.0", "cu118"),
                 ("12.1", "2.5.1", "cu121"), ("12.3", "2.5.1", "cu121"),
                 ("12.4", "2.6.0", "cu124"), ("12.7", "2.6.0", "cu124"),
                 ("12.8", "2.7.1", "cu128"), ("13.0", "2.7.1", "cu128")]
        for cuda, version, build in cases:
            with self.subTest(cuda=cuda):
                selected = setup.select_profile(f"| CUDA Version: {cuda} |", (8, 0))
                self.assertEqual((selected["torch"], selected["build"]), (version, build))

    def test_rejects_unsupported_driver_and_gpu_combinations(self):
        for output, capability in [("CUDA Version: 11.7", (8, 0)),
                                   ("CUDA Version: 12.4", (12, 0)),
                                   ("CUDA Version: 12.8", (7, 5)),
                                   ("CUDA Version: N/A", None)]:
            with self.subTest(output=output, capability=capability):
                with self.assertRaises(ValueError):
                    setup.select_profile(output, capability)

    def test_install_pins_torch_and_reconnect_needs_no_conda_or_installs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prefix = root / "persistent environment"
            conda = root / "conda"
            conda.touch()
            profile = setup.select_profile("CUDA Version: 12.4", (8, 0))
            commands = []

            def run(command, **kwargs):
                commands.append(command)
                if "create" in command:
                    (prefix / "bin").mkdir(parents=True)
                    (prefix / "bin/python").touch()
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")

            def output(command, **kwargs):
                if "freeze" in command:
                    return "torch==2.6.0+cu124\n"
                return '{"torch": "2.6.0+cu124", "flash_attention": "passed"}\n'

            with patch.object(setup.shutil, "which", return_value=str(conda)), \
                 patch.object(setup.subprocess, "run", side_effect=run), \
                 patch.object(setup.subprocess, "check_output", side_effect=output):
                setup.setup(prefix, profile, "0")
                self.assertEqual((prefix / "exp57_torch_constraints.txt").read_text(),
                                 "torch==2.6.0+cu124\n")
                dependency_install = next(c for c in commands if "-r" in c)
                self.assertIn("-c", dependency_install)
                self.assertEqual(json.loads((prefix / "exp57_setup.json").read_text())["state"], "ready")
                commands.clear()
                with patch.object(setup.shutil, "which", side_effect=AssertionError("Conda must not be needed")):
                    setup.setup(prefix, profile, "0")
                self.assertEqual(commands, [[str(prefix / "bin/python"), "-m", "pip", "check"]])

    def test_new_driver_does_not_overwrite_existing_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            prefix = Path(temp)
            (prefix / "exp57_setup.json").write_text(json.dumps(dict(
                state="ready", spec=dict(python="3.12", torch="2.7.1+cu128"))))
            with patch.object(setup.subprocess, "run") as run:
                with self.assertRaisesRegex(ValueError, "new --prefix"):
                    setup.setup(prefix, setup.select_profile("CUDA Version: 12.4"), "0")
                run.assert_not_called()

    def test_unmanaged_prefix_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "unmanaged environment"):
                setup.setup(Path(temp), setup.select_profile("CUDA Version: 12.4"), "0")


if __name__ == "__main__":
    unittest.main()
