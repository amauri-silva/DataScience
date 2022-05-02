
from datetime import datetime, timezone
import re
import time

dt = datetime( 2021, 3, 5, 14, 30, 21)
#2019-07-05 12:00:00
tm = ["2021-03-05 17:30:29", "2021-03-05 15:30:30", "2021-03-05 14:30:27", "2021-03-05 14:30:28"]


def datetime_to_timestamp(date_time):
    d1 = []
    d2 = []
    for dt in date_time:
        times_tamp = time.mktime(datetime.strptime(dt, '%Y-%m-%d %H:%M:%S').timetuple())
        print("#### times_tamp {}".format(times_tamp))
        timestamp = int(times_tamp)
        d1.append(timestamp)

        dt = datetime.fromtimestamp( timestamp )
        print("#### dt {}".format(dt))
        d2.append(dt)
    print("#### D1 {}".format(d1[0]))
    print("#### D2 {}".format(d2[0]))

    return  d1, d2

d1, d2 = datetime_to_timestamp(tm)

print("#### D2 {}".format(d1))
