from .sqliteinterface import getGnssData, getImuData, getMagnetometerData
import ahrs
# general camera stuff here

def getEulerAngles(since, until=None):
    data = getImuData(since, until)
    angles = foo(data)
    return angles

def foo(data):
    return 'idk 90 degrees'