#!/bin/sh
kill -9 $(ps -ef|grep 'python'|grep -v grep|awk '{print $2}')
echo "stop.sh over...."
