import json

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
        # TODO: Add FIELDS credentials for variance analysis
        creds = [credentials['psp']['sweap']['username'], credentials['psp']['sweap']['password']]
        return creds
    else:
        return None