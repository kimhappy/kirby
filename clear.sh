#!/bin/bash

PATTERNS=(".venv" "__pycache__" "uv.lock")
CURRENT_DIR=$(pwd)
CLEAN_UV_CACHE=false

while getopts "u" opt; do
  case ${opt} in
    u)
      CLEAN_UV_CACHE=true
      ;;
    *)
      echo "Usage: $0 [-u]"
      exit 1
      ;;
  esac
done

if [ "$(basename "$CURRENT_DIR")" = "cli" ]; then
    cd ..
elif [ -d "./cli" ]; then
    :
else
    exit 1
fi

FIND_EXPR=""

for pattern in "${PATTERNS[@]}"; do
    if [ -z "$FIND_EXPR" ]; then
        FIND_EXPR="-name \"$pattern\""
    else
        FIND_EXPR="$FIND_EXPR -o -name \"$pattern\""
    fi
done

eval "find . $FIND_EXPR" | while read -r item; do
    rm -rf "$item"
done

if [ "$CLEAN_UV_CACHE" = true ]; then
    uv cache clean
fi

cd "$CURRENT_DIR"
