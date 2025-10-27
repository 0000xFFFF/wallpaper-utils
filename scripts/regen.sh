#!/usr/bin/env bash

WALLPAPERS_DIR="/media/SSD/media/bg"
CSV_FILE="/media/SSD/media/bg/wpu-darkscore_output.csv"
wpu-darkscore -i "$WALLPAPERS_DIR" -o "$CSV_FILE" -s
