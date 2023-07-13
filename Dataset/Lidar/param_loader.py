# -*- coding: utf-8 -*-
"""载入config.json参数
"""

import json
import numpy as np

with open("config.json") as json_file:
    config = json.load(json_file)

class ParamLoader():
    """config配置加载
    """

    ip = config['IP']
    port = config['Port']
    target_streamID = config["target_streamID"]
    gateway_names = config["gateway_names"]
