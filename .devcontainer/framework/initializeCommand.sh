#!/bin/bash
# Create the mounted config directories if they don't already exist

mkdir -p ~/.aws
mkdir -p ~/.ngc
mkdir -p ~/.cache
mkdir -p ~/.ssh
[ ! -f ~/.netrc ] && touch ~/.netrc

# Create the ~/.bash_history_devcontainer file if it doesn't exist
[ ! -f ~/.bash_history_devcontainer ] && touch ~/.bash_history_devcontainer

exit 0
