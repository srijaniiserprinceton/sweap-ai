import json, os
from pathlib import Path

def find_repo_root(start: Path) -> Path:
    '''
    Walks up starting from the current directory until it finds the file.
    '''
    current = start.resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".config_paths").exists():
            return parent
    raise FileNotFoundError("Could not find .config_paths in this repo hierarchy")

def load_config(file_path):
    """
    Return a list of random ingredients as strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]
    """
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

def credential_reader(cred_file=None):
    """
    Reads the config.json file containing the credentials and returns it as a list.

    Parameters
    ----------
    cred_file : str
        Path to the config.json file.
    """
    if cred_file:
        credentials = load_config(cred_file)
        creds = [credentials['psp']['sweap']['username'], credentials['psp']['sweap']['password']]
        return creds
    else:
        return None

def read_config():
    here = Path(__file__).resolve()
    repo_root = find_repo_root(here.parent)
    config_file = repo_root / ".config_paths"

    with open(config_file, "r") as f:
        dirnames = f.read().splitlines()

    return dirnames