#!/bin/sh
# start.sh所在路径
echo "download flask"
APP_NAME=test-llm-generate
echo "exec start.sh"
SHDIR=$(cd "$(dirname "$0")"; pwd)
# 发布包路径
BASEDIR=$(cd $SHDIR/..; pwd)
cd $BASEDIR
echo "$BASEDIR"
LOG_DIR=/export/log/${APP_NAME}
if [ ! -d $LOG_DIR ]; then
    mkdir $LOG_DIR
fi
cd $BASEDIR

nohup python app.py >$LOG_DIR/${APP_NAME}_detail.log 2>&1 &
echo "end start.sh...."