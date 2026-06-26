import argparse
import os
import subprocess
import sys


def train():
    subprocess.run([sys.executable, "-m", "src.pipeline.train", "--config", "configs/config.yaml"])


def serve():
    subprocess.run(
        ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    )


def test():
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--cov=src", "--cov-report=term-missing"])


def lint():
    subprocess.run(["ruff", "check", "src/", "tests/"])
    subprocess.run(["black", "--check", "src/", "tests/"])


def format():
    subprocess.run(["ruff", "check", "--fix", "src/", "tests/"])
    subprocess.run(["black", "src/", "tests/"])


def clean():
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d in ("__pycache__", ".pytest_cache", ".hypothesis", ".ipynb_checkpoints"):
                path = os.path.join(root, d)
                subprocess.run(["rmdir", "/s", "/q", path], shell=True)
    for f in ("mlflow.db",):
        if os.path.exists(f):
            os.remove(f)


COMMANDS = {
    "train": train,
    "serve": serve,
    "test": test,
    "lint": lint,
    "format": format,
    "clean": clean,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project tasks")
    parser.add_argument("command", choices=COMMANDS.keys())
    args = parser.parse_args()
    COMMANDS[args.command]()
