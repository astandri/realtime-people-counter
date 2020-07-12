import json
import os
import time
import ast
import logging


def read_config(file_path):
    """
        Function for reading config file. Using absolute path to prevent file not found when creating the documentations

        Parameters
        ----------
            file_path : (str)
                Path of designated config file (usually it is on the configs folder)
        
        Returns
        -------
            dict
                The dictionary of read config file

    """
    # this_directory = os.path.dirname(__file__)
    # __file__ is the absolute path to the current python file.
    # with open(os.path.join(this_directory, file_path)) as config_file:
    #     conf = json.load(config_file)
    config_file = open(file_path, 'r')
    conf = json.load(config_file)
    config_file.close()
    return conf