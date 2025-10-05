"""Convenience script to run full pipeline sequentially without make.
Usage: python run_all.py
"""
from subprocess import run, CalledProcessError
import sys

COMMANDS = [
    [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
    [sys.executable, 'src/data_download.py'],
    [sys.executable, 'src/train.py', '--config', 'configs/baseline.yaml'],
    [sys.executable, 'src/train.py', '--config', 'configs/engineered.yaml'],
    [sys.executable, 'src/evaluate.py']
]

def main():
    for cmd in COMMANDS:
        print(f"\n>>> Running: {' '.join(cmd)}")
        try:
            r = run(cmd, check=True)
        except CalledProcessError as e:
            print(f"Command failed with code {e.returncode}: {' '.join(cmd)}")
            sys.exit(e.returncode)
    print('\nPipeline completed. See reports/ for outputs.')

if __name__ == '__main__':
    main()
