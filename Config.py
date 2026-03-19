# This file contains configuration variables used across the overall project.
# For example, this will contain the name of dataframe columns we will work on in each file.
import os


class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test
    TYPE_COLS = ['y2', 'y3', 'y4']
    CLASS_COL = 'y2'
    GROUPED = 'y1'

    # Data file paths (relative to project directory)
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    DATA_FILES = ['AppGallery.csv', 'Purchasing.csv']

    # Column rename mapping from raw CSV to internal names
    COLUMN_RENAME = {
        'Type 1': 'y1',
        'Type 2': 'y2',
        'Type 3': 'y3',
        'Type 4': 'y4',
    }

    # Chained multi-output configuration (Design Choice 1)
    CHAIN_SEPARATOR = ' + '
    CHAINED_TARGETS = {
        'chain_1': ['y2'],
        'chain_2': ['y2', 'y3'],
        'chain_3': ['y2', 'y3', 'y4'],
    }

    # Hierarchical modelling configuration (Design Choice 2)
    HIERARCHICAL_LEVELS = ['y2', 'y3', 'y4']

    # Minimum number of instances a class must have to be kept
    MIN_CLASS_COUNT = 5

    # Random seed for reproducibility
    SEED = 0
