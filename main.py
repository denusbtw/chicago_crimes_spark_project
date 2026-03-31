import sys

from runners.run_preprocessing import run_preprocessing
from runners.run_stats import run_stats
from runners.run_correlation import run_correlation
from runners.run_pca import run_pca


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [preprocessing|stats|correlation|pca]")
        return

    command = sys.argv[1].strip().lower()

    if command == "preprocessing":
        run_preprocessing()
    elif command == "stats":
        run_stats()
    elif command == "correlation":
        run_correlation()
    elif command == "pca":
        run_pca()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: preprocessing, stats, correlation, pca")


if __name__ == "__main__":
    main()