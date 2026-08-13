SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$SCRIPT_DIR/.venv/bin/activate"
export PATH="$VIRTUAL_ENV/bin:$SCRIPT_DIR/data/external/bin:${PATH#"$VIRTUAL_ENV/bin:"}"
if [ -d "$SCRIPT_DIR/data/external/checkv/checkv-db-v1.5" ]; then
  export CHECKVDB="$SCRIPT_DIR/data/external/checkv/checkv-db-v1.5"
fi
