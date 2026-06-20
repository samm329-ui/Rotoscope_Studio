"""Install required Python packages for Rotoscope Studio.

This script is invoked by start.bat. It does not install
Python itself; it only installs the Python packages that
the application needs.
"""
import os
import subprocess
import sys
import pathlib


ROOT = pathlib.Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"


def _pip_install(extra_args=None) -> int:
    """Run pip with the given args and return the exit code."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(["-r", str(REQUIREMENTS)])
    print("Running: " + " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    """Install dependencies from requirements.txt."""
    if not REQUIREMENTS.is_file():
        print("Cannot find requirements.txt at " + str(REQUIREMENTS))
        return 1
    print("Installing dependencies from requirements.txt ...")
    rc = _pip_install(["--upgrade", "--disable-pip-version-check"])
    if rc != 0:
        print("pip install failed with exit code " + str(rc))
        return rc
    print("Dependencies installed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())