# -*- coding: utf-8 -*-
"""load config.json Parameters
"""

import json
import numpy as np

with open("config.json") as json_file:
    config = json.load(json_file)

class ParamLoader():
    """load config.json
    """
    version = config['version']
    algorithm = config['algorithm']
    realtime = config['realtime']
    gateway_names = config['gateway_names']

    hardware = config['hardware']
    ant1offsets   = np.array(hardware['ant1offset'])
    ant2offsets   = np.array(hardware['ant2offset'])
    ant3offsets   = np.array(hardware['ant3offset'])
    atnseq = np.array(hardware['atnseq'])

    server = config['server']
    ip = server['IP']
    port = server['Port']
    username = server['Username']
    password = server['Password']

    loc = config['loc']
    music_param = loc['music']
    freq = music_param['frequency']
    element_dis = music_param["element_dis"]
    array_length = music_param["array_length"]
    subarray_length = music_param["subarray_length"]
    xRange = music_param['xRange']
    yRange = music_param['yRange']
    zRange = music_param['zRange']
    coarse_resolution = music_param['coarse_resolution']
    fine_resolution = music_param['fine_resolution']
