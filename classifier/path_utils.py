"""Path utilities."""
from pathlib import Path
from typing import Generator

from classifier.constants import VALID_IMAGE_FILE_EXTENSION


def get_path(path: str) -> Path:
    """
    Get the given path.
    :param path: a str
    :return: a Path
    """
    return Path(path)


def is_file(path: Path) -> bool:
    """
    Return true if the given path is a file, false otherwise.
    :param path: a Path
    :return: a bool
    """
    return path.is_file()


def is_dir(path: Path) -> bool:
    """
    Return true if the given path is a directory, false otherwise.
    :param path: a Path
    :return: a bool
    """
    return path.is_dir()


def has_valid_image_file_extension(path: Path) -> bool:
    """
    Return true if the given path has a valid image file extension, false otherwise.
    :param path: a Path
    :return: a bool
    """
    return path.suffix == VALID_IMAGE_FILE_EXTENSION


def get_entries(path: Path) -> Generator[Path, None, None]:
    """
    Get the given path entries.
    :param path: a Path
    :return: a Generator
    """
    return path.iterdir()


def create_dir(path: Path):
    """
    Create the directory at the given path.
    :param path: a Path
    :return: nothing
    """
    Path.mkdir(path, exist_ok=True)


def get_parent_dir(path: Path) -> Path:
    """
    Get the parent directory of the given path.
    :param path: a Path
    :return: a Path
    """
    return path.parents[0]


def get_path_as_str(path: Path) -> str:
    """
    Get the given path as a string.
    :param path: a Path
    :return: a str
    """
    return str(path)
