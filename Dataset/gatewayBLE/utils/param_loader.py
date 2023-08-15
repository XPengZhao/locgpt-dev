# -*- coding: utf-8 -*-
""" load config.json
"""

import json
import os

print(os.listdir())
with open("./config.json") as json_file:
    config = json.load(json_file)

class Paramloader():
    """load config.json
    """

    gateway_name = config['gateway_name']
    gen2 = config['Gen2']
    samplerate = gen2['SampleRate']
    BLF = gen2['BLF']
    M = gen2['Encoding']
    FrT = gen2['FrT']
    Tari = gen2['Tari']
    RTcal = gen2['RTCal']

    server = config['server']
    ip = server['ip']
    port = server['port']
    username = server['user']
    password = server['pwd']

    array = config['Array']
    cycleduration = array['CycleDuration']
    arrt1duration = array['arrT1Duration']
    arrt2duration = array['arrT2Duration']
    arrt3duration = array['arrT3Duration']

    tagparam = config['MultiTag']
    tagNum = tagparam['TagNum']
    T2Length = tagparam['T2Length']
