# mpv rtsp://192.168.2.5:8554/front.mp4 --title=Front --no-cache --untimed --no-demuxer-thread --vd-lavc-threads=1 --profile=low-latency --no-correct-pts --osc=no &
# mpv rtsp://192.168.2.5:8554/bottom.mp4 --title=Front --no-cache --untimed --no-demuxer-thread --vd-lavc-threads=1 --profile=low-latency --no-correct-pts --osc=no &

#!/bin/bash

# Get PID of this script
SCRIPT_PID=$$

# Handle Ctrl+C
trap 'echo "Stopping streams..."; pkill -P $SCRIPT_PID; exit' SIGINT

# Start both mpv streams as child processes
mpv rtsp://192.168.2.5:8554/front.mp4 --title=Front --no-cache --untimed --no-demuxer-thread --vd-lavc-threads=1 --profile=low-latency --no-correct-pts --osc=no &
mpv rtsp://192.168.2.5:8554/bottom.mp4 --title=Bottom --no-cache --untimed --no-demuxer-thread --vd-lavc-threads=1 --profile=low-latency --no-correct-pts --osc=no &
mpv rtsp://192.168.2.5:8554/front_annotated.mp4 --title=Bottom --no-cache --untimed --no-demuxer-thread --vd-lavc-threads=1 --profile=low-latency --no-correct-pts --osc=no &


# Wait for child processes
wait

