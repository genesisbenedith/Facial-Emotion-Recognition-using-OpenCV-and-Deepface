#!/bin/zsh

# Auto-activate venv if not already active
if [ -d "./venv" ] && [ "$VIRTUAL_ENV" != "$(pwd)/venv" ]; then
    source ./venv/bin/activate
    echo "(venv auto-activated)"
fi
