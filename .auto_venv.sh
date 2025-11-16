#!/bin/zsh

# Path to venv (change if yours is named differently)
VENV_PATH="./venv"

# If leaving a venv project folder → deactivate
if [[ -n "$VIRTUAL_ENV" ]]; then
    # if we're leaving the directory that contains our venv
    if [[ "$PWD" != "$VIRTUAL_ENV:h:h"* ]]; then
        deactivate 2>/dev/null
        echo "(venv deactivated)"
    fi
fi

# If entering a folder with a venv that is not active → activate
if [[ -d "$VENV_PATH" ]]; then
    if [[ "$VIRTUAL_ENV" != "$PWD/venv" ]]; then
        source "$VENV_PATH/bin/activate"
        echo "(venv auto-activated)"
    fi
fi
