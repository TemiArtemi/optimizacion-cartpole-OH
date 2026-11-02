from pathlib import Path

from src.uci.preparation import (
    UCI_URL,
    download_archive,
    prepare_letter_splits,
    read_letter_table,
    save_splits,
)


def main() -> None:
    """Downloads the UCI letter archive, preprocesses it and stores the three splits."""
    data_root = Path(__file__).resolve().parent
    archive_path = data_root / "raw" / "letter+recognition.zip"
    download_archive(UCI_URL, archive_path)
    labels, features = read_letter_table(archive_path)
    splits = prepare_letter_splits(labels, features)
    output_directory = data_root / "letter"
    save_splits(output_directory, splits)
    print(f"letter dataset stored at {output_directory}")


if __name__ == "__main__":
    main()
