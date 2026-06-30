from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).parent.parent
MAX_LINES = 500


def check_file_lengths() -> list[str]:
    errors = []
    for path in ROOT.rglob("*.py"):
        if "venv" in str(path) or ".egg" in str(path):
            continue
        lines = path.read_text().splitlines()
        if len(lines) > MAX_LINES:
            errors.append(f"{path.relative_to(ROOT)}: {len(lines)} lines (> {MAX_LINES})")
    return errors


def check_cyclic_imports() -> list[str]:
    errors = []
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import factor_mining"],
            capture_output=True, text=True, cwd=ROOT,
        )
        if result.returncode != 0:
            errors.append(f"Import error: {result.stderr.strip()}")
    except Exception as e:
        errors.append(str(e))
    return errors


def check_init_files() -> list[str]:
    errors = []
    src = ROOT / "src" / "factor_mining"
    for d in src.rglob("__init__.py"):
        pass
    for d in src.rglob("*"):
        if d.is_dir() and not d.name.startswith("__"):
            init_file = d / "__init__.py"
            if not init_file.exists():
                errors.append(f"Missing {init_file.relative_to(ROOT)}")
    return errors


def main():
    errors = []
    errors.extend(check_file_lengths())
    errors.extend(check_cyclic_imports())
    errors.extend(check_init_files())

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        sys.exit(1)
    print("audit_check.py: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
