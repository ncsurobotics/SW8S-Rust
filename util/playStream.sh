#!/usr/bin/env bash

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
mpv --title=front --no-cache --untimed --profile=low-latency --no-correct-pts --fps=30 --osc=no rtsp://192.168.2.5:8554/front.mp4 &
mpv --title=bottom --no-cache --untimed --profile=low-latency --no-correct-pts --fps=30 --osc=no rtsp://192.168.2.5:8554/bottom.mp4
 