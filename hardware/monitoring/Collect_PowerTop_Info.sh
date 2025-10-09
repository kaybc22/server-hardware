#!/bin/bash

#Create the crontab to collect the Logs per 5m
#crontab -e
#*/5 * * * * /path/to/Collect.sh
*/30 * * * * /opt/testing/utls/PowerTop_Info.sh

DATE=$(date +%Y%m%d%H%M)
powertop --html=/path/log/test_$DATE.html