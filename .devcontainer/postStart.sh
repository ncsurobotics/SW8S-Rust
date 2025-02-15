#!/bin/bash

source ~/.bashrc

echo -e "Setting up the environment..."

echo -e "\n----COMMANDS----"
echo "seawolf build          : Build the project (local)"
echo "seawolf build jetson   : Cross compile for jetson"
echo "seawolf build local    : Compile for local machine (x86_64)"
echo "seawolf run            : Run the project (local build + run)"

echo -e "\n----Build Command----"
echo -e "ctrl + shift + b   : Build the project for jetson"