#!/usr/bin/env bash

# add this script to your autostarts

# set your own csv file and command for setting wallpaper
CSV_FILE="/media/SSD/media/bg/wpu-darkscore_output.csv"
EXEC_CMD="plasma-apply-wallpaperimage"

if pgrep -f "wpu-darkscore-select" ; then
    echo "Process is running"
else
    echo "Process is not running, starting..."
    wpu-darkscore-select -i "$CSV_FILE" -e "$EXEC_CMD" -l -d
fi
