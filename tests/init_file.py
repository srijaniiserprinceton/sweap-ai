# init_gdf_default.py
# This is a template for primary initization file.
# 'global': parameters needed for any method of reconstruction
# 'polcap', 'cartesian', 'hybrid' are parameters specific to particular methods of reconstruction

config = {
    'global': {                                                             #--------GLOBAL PARAMETERS FOR GDF-----------#
        'TRANGE'          : ['2020-01-26T14:28:00', '2020-01-26T20:30:00'], # Define the time range to load in from pyspedas
        'SYNTHDATA_FILE'  : None,                                           # Path to a data file containing synthetic observation
        'CLIP'            : True,                                           # If you want to clip the loaded day's data to the specified TRANGE
        'START_INDEX'     : 0,                                              # Starting index with respect to the first timestamp in TRANGE
        'CREDS_PATH'      : '../config.json',                                # path to the <.json> file containing credentials
    },
    'span': {
        'COUNT_THRESHOLD': 3,
    },
}