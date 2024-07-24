#!/bin/bash

set -x
GOOS=linux GOARCH=arm64 go build ./cmd/datalogger && \
ssh -t hdc 'systemctl stop odc-api' && \
ssh -t hdc 'systemctl stop hivemapper-data-logger' && \
ssh -t hdc 'mount -o remount,rw /' &&
scp datalogger hdc:/opt/dashcam/bin && \
sleep 2 && \
# ssh -t hdc 'rm /data/recording/data-logger.v1.4.5.db' && \
# ssh -t hdc 'rm /data/recording/data-logger.v1.4.5.db-shm' && \
# ssh -t hdc 'rm /data/recording/data-logger.v1.4.5.db-wal' && \
ssh -t hdc 'systemctl start hivemapper-data-logger' && \
ssh -t hdc 'journalctl -feu hivemapper-data-logger'
set +x
