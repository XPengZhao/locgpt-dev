# -*- coding: utf-8 -*-
"""从文件获取时间戳
"""

def get_timestamp(filename):
    """从文件名获得时间戳
    """
    t, dot = filename.find('T'), filename.find('.')
    h, m, s, us = filename[t + 1:dot].split('-')
    year, month, day = filename[t-11:t-1].split('_')
    timestamp = "{year}-{month}-{day} {h}:{m}:{s}:{us}".format(year=year, month=month, day=day,
                                                                    h=h, m=m, s=s, us=us)
    return timestamp


def get_s_today(timestamp):
    """获得从今天零点开始算的s数
    """
    today_time = timestamp.find(' ')
    h, m, s, us = timestamp[today_time+1:].split(':')
    today_s = float(h)*3600 + float(m)*60 + float(s) + float(us)/1000000
    return today_s


def get_filename(filename):
    """从时间戳得到要保存的json文件名
    """
    t, dot = filename.find('T'), filename.find('.')
    h, m, s, us = filename[t + 1:dot].split('-')
    year, month, day = filename[t-11:t-1].split('_')
    json_name = "{year}_{month}_{day}_{h}_{m}_{s}".format(year=year, month=month, day=day,
                                                                    h=h, m=m, s=s)

    return json_name
