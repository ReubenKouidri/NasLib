from importlib import import_module
from typing import Any

__all__ = [
    "cbam"
]


def get_all_genes():
    genes = []
    ss_module = import_module(".__init__", package="search_space")
    search_spaces = ss_module.__all__

    # Dynamically import the genetics module from each search space and extend
    for space in search_spaces:
        try:
            genetics_module = import_module(f".{space}.genetics",
                                            package="search_space")
            genes.extend(genetics_module.__all__)
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not import genetics from {space} due to {e}")

    # Remove duplicates if any
    genes = list(set(genes))
    return genes


def get_search_space(space_name: str) -> Any:
    """
    Dynamically import and return the specified search space module.

    Args:
        space_name (str): The name of the search space to import.

    Returns:
        The imported search space module.
    """
    try:
        return import_module(f".{space_name}.{space_name}_search_space",
                             __name__)
    except ModuleNotFoundError:
        raise ValueError(f"Search space '{space_name}' is not available.")
