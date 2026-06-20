"""Environment check for Rotoscope Studio.

This script verifies that Python and the required
packages are present. It does NOT install anything,
it only reports status.
"""
import sys
import importlib

REQUIRED_MODULES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("python_multipart", "python-multipart"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("PIL", "pillow"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python-headless"),
]


def _check_python() -> bool:
    """Report whether Python is available."""
    version = sys.version_info
    major = version.major
    minor = version.minor
    try:
        major_num = int(major)
        minor_num = int(minor)
        if major_num < 3 or (major_num == 3 and minor_num < 9):
            print("Python " + str(major) + "." + str(minor) + " detected. Please install Python 3.9 or higher.")
            return False
    except Exception:
        pass
    print("Python " + str(major) + "." + str(minor) + " detected.")
    return True


def _check_module(import_name, package_name=None) -> bool:
    """Report whether a given module is importable."""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        name = package_name or import_name
        print("Missing package: " + name)
        return False


def main() -> int:
    """Run all the checks and return a non-zero exit code if anything fails."""
    print("Rotoscope Studio - setup check")
    print("-" * 40)
    print("")
    python_ok = _check_python()
    if not python_ok:
        return 1
    print("Checking packages...")
    all_ok = True
    for import_name, package_name in REQUIRED_MODULES:
        if not _check_module(import_name, package_name):
            all_ok = False
    if all_ok:
        print("All required packages are installed.")
    else:
        print("Some packages are missing. Run start.bat to install them.")
        return 1
    print("")
    print("Environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())