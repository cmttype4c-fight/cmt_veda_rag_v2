"""Runs deploy/check_context.py: .dockerignore must keep runtime data and env
files out of the build context and every source file in. Run: python3 test_docker_context.py"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def test_build_context_rules():
    r = subprocess.run([sys.executable, os.path.join(HERE, "deploy", "check_context.py"), HERE],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    print("PASS:", r.stdout.strip())


if __name__ == "__main__":
    test_build_context_rules()
    print("1/1 passed")
