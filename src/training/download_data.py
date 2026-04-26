import os


def main():
    os.makedirs("../../data/raw", exist_ok=True)
    print("Using local dataset in data/raw")


if __name__ == "__main__":
    main()
