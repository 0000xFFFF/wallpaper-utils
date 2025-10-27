#!/usr/bin/env bash

if pgrep -f "wpu-darkscore-select" ; then
    echo "Stopping process..."
    pkill -f wpu-darkscore-select
fi
