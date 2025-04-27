"""Script to download flight price data from Kaggle and load it into a SQLite database.

This module provides functionality to:
1. Download the flight prices dataset from Kaggle using kagglehub
2. Load the CSV data into a SQLite database for analysis
"""

import os
import subprocess
from pathlib import Path

import kagglehub  # type: ignore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def download_dataset():
    """Download the dataset using kagglehub."""
    # Get dataset selector from environment variable or use default
    dataset_selector = os.getenv('KAGGLE_DATASET', 'dilwong/flightprices')

    # Download the dataset
    path = kagglehub.dataset_download(dataset_selector)
    return Path(path)

def load_to_sqlite(db_path: str) -> list[str]:  # pylint: disable=redefined-outer-name
    """Load the dataset into SQLite database using the sqlite3 CLI."""
    # Get the dataset path
    dataset_path = download_dataset()

    # Find all CSV files
    csv_files = [f for f in dataset_path.iterdir() if f.is_file() and f.name.endswith('.csv')]
    if not csv_files:
        raise RuntimeError(
            f"No CSV files found in downloaded dataset: {dataset_path}"
        )

    # Build SQLite import commands for all files
    import_commands = []
    for csv_file in csv_files:
        table_name = csv_file.stem
        import_commands.extend([
            f'DROP TABLE IF EXISTS {table_name};',
            f'.import "{os.fspath(csv_file)}" {table_name} --csv'
        ])

    # Run all imports in a single SQLite process
    subprocess.run(
        ['sqlite3', db_path],
        input='\n'.join(import_commands),
        text=True,
        check=True,
    )

    return [f.name for f in csv_files]

if __name__ == '__main__':
    # Allow database path to be configured via environment variable
    db_path = os.getenv('DB_PATH', 'flights.db')

    tables = load_to_sqlite(db_path)
    print(f"Data successfully loaded into SQLite database at: {db_path}")
    print("Imported tables:")
    for table in tables:
        print(f"- {table}")
