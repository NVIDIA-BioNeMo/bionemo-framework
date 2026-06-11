#!/bin/bash
# Create the mounted config directories if they don't already exist

mkdir -p ~/.devcontainer_cache
mkdir -p ~/.ssh
mkdir -p ~/.cache/pre-commit-devcontainer
mkdir -p ~/.gnupg
mkdir -p ~/.config
mkdir -p ~/.cursor
mkdir -p ~/.claude
mkdir -p ~/.codex
[ ! -f ~/.netrc ] && touch ~/.netrc

[ ! -f ~/.bash_history_devcontainer ] && touch ~/.bash_history_devcontainer
[ ! -f ~/.claude.json ] && touch ~/.claude.json

exit 0
