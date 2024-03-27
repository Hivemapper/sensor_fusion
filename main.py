import fusion
from datetime import datetime, timedelta


dateFormat = '%Y-%m-%d %H:%M:%S'

testDate = datetime.strptime('2024-03-27 19:46:19', dateFormat)
values = fusion.getImuData(testDate - timedelta(seconds=1), testDate)
print(values)

try:
    testdate = datetime.strptime('2024-03-27 19:45:05', dateFormat)
    values = fusion.getMagnetometerData(testDate - timedelta(seconds=1), testDate)
    print(values)
except Exception as e:
    print(e)

testDate = datetime.strptime('2024-02-28 21:11:28', dateFormat)
values = fusion.getGnssData(testDate - timedelta(seconds=1), testDate)
print(values)

testDate = datetime.strptime('2024-02-28 21:11:28', dateFormat)
print(fusion.getEulerAngles(testDate, testDate + timedelta(seconds=1)))