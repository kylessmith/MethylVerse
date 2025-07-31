import os
from os.path import join
import glob


def get_data_file(filename):
    """
    """

    # Check if filename already exists
    #if os.path.exists(filename):

    # Find data directory
    data_dir = os.path.split(os.path.realpath(__file__))[0]
    #data_dir = os.path.join(data_dir, filename)
    data_dir = glob.glob(join(join(data_dir,"**"), filename), recursive=True)
    if len(data_dir) == 0:
        raise FileNotFoundError("Data file not found! ("+filename+")")
    data_dir = data_dir[0]

    return data_dir