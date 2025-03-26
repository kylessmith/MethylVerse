import os


def get_data_file(filename):
    """
    """

    # Check if filename already exists
    #if os.path.exists(filename):

    # Find data directory
    data_dir = os.path.split(os.path.realpath(__file__))[0]
    data_dir = os.path.join(data_dir, filename)

    # Check if file exists
    if os.path.exists(data_dir) == False:
        raise FileExistsError("Data file does not exist! ("+data_dir+")")

    return data_dir