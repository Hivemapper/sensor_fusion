#!/bin/bash

# Start hivemapper-data-logger.service
echo "Starting hivemapper-data-logger.service..."
systemctl start hivemapper-data-logger.service
sleep 2

# Start odc-api.service
echo "Starting odc-api.service..."
systemctl start odc-api.service
sleep 2

# Start sensor-fusion.service
echo "Starting sensor-fusion.service..."
systemctl start sensor-fusion.service
sleep 2

# Start object-detection.service
echo "Starting object-detection.service..."
systemctl start object-detection.service
sleep 2

echo "All services started successfully."
