#! /bin/bash

cd "$(dirname "$0")"

uvclean=false

for arg in "$@"; do
    if [ "$arg" = "--uvclean" ]; then
        uvclean=true
    fi
done

if $uvclean; then
    find . -type d -name ".venv" -exec rm -rf {} \; 2>/dev/null || true
    find . -type f -name "uv.lock" -delete 2>/dev/null || true

    cd kirby-cli
    uv sync --refresh
    cd ..
fi
