"""Data loading module for PredictLab."""

import csv
from pathlib import Path

import pandas as pd

from predictlab.config import get_settings
from predictlab.data.validate import DataValidationError, DuplicateColumnsError
from predictlab.logging_config import get_logger

logger = get_logger(__name__)


def load_raw_dataset(
    file_path: str | Path | None = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load the raw dataset from disk into a Pandas DataFrame.

    Parameters
    ----------
    file_path : Optional[Union[str, Path]]
        Explicit path to the CSV file. If None, the configured raw_data_path is used.
    encoding : str
        Character encoding of the CSV file (defaults to 'utf-8').

    Returns
    -------
    pd.DataFrame
        Loaded raw dataset as an immutable copy.

    Raises
    ------
    FileNotFoundError
        If the file does not exist on disk.
    DataValidationError
        If the file is empty, corrupted, or cannot be parsed as a CSV.
    """
    settings = get_settings()
    target_path = Path(file_path) if file_path is not None else settings.raw_data_path

    logger.info("Attempting to load raw dataset from: %s", target_path)

    if not target_path.exists():
        msg = f"Raw dataset not found at expected location: {target_path.resolve()}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    if not target_path.is_file():
        msg = f"Path exists but is not a regular file: {target_path.resolve()}"
        logger.error(msg)
        raise DataValidationError(msg)

    # Check for empty file (0 bytes)
    if target_path.stat().st_size == 0:
        msg = f"Raw dataset file is empty (0 bytes): {target_path.resolve()}"
        logger.error(msg)
        raise DataValidationError(msg)

    # Inspect CSV header row directly for duplicate column headers
    try:
        with open(target_path, encoding=encoding, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                duplicate_headers = [col for col in set(header) if header.count(col) > 1]
                if duplicate_headers:
                    msg = f"CSV header contains duplicate column names: {sorted(duplicate_headers)}"
                    logger.error(msg)
                    raise DuplicateColumnsError(msg)
    except DuplicateColumnsError:
        raise
    except Exception as exc:
        msg = f"Failed inspecting CSV header for {target_path.resolve()}: {exc}"
        logger.error(msg)
        raise DataValidationError(msg) from exc

    try:
        df = pd.read_csv(target_path, encoding=encoding)
    except pd.errors.EmptyDataError as exc:
        msg = f"CSV file contains no data or headers: {target_path.resolve()}"
        logger.error(msg)
        raise DataValidationError(msg) from exc
    except pd.errors.ParserError as exc:
        msg = f"Malformed CSV file could not be parsed: {target_path.resolve()} ({exc})"
        logger.error(msg)
        raise DataValidationError(msg) from exc
    except Exception as exc:
        msg = f"Unexpected failure loading dataset from {target_path.resolve()}: {exc}"
        logger.error(msg)
        raise DataValidationError(msg) from exc

    logger.info(
        "Successfully loaded raw dataset from %s (shape: %s rows, %s columns)",
        target_path.name,
        df.shape[0],
        df.shape[1],
    )

    # Return a clean copy to prevent caller side-effects on internal representations
    return df.copy()
