#!/usr/bin/env python

import datetime

def ymd():
    now = datetime.datetime.now()
    y = str(now.year)
    m = str("{0:0=2d}".format(now.month))
    d = str("{0:0=2d}".format(now.day))
    date = y+m+d
    return date

def hms():
    now = datetime.datetime.now()
    y = str("{0:0=2d}".format(now.hour))
    m = str("{0:0=2d}".format(now.minute))
    d = str("{0:0=2d}".format(now.second))
    date = y+m+d
    return date

def datenum_complete():
    ymdstr = ymd()
    hmsstr = hms()
    date = ymdstr+hmsstr
    return date
