#!/bin/bash

# Check if an argument was provided
if [ -z "$1" ]; then
  echo "❌ Usage: $0 <path_to_folder_or_tfevents_file>"
  exit 1
fi

# Get the input path
PATH_INPUT="$1"

# If it's a .tfevents file, get its parent directory
if [[ -f "$PATH_INPUT" && "$PATH_INPUT" == *.tfevents* ]]; then
  LOGDIR=$(dirname "$PATH_INPUT")
else
  LOGDIR="$PATH_INPUT"
fi

# Check that the directory exists
if [ ! -d "$LOGDIR" ]; then
  echo "❌ Error: directory $LOGDIR does not exist."
  exit 1
fi

# Launch TensorBoard
echo "🚀 Launching TensorBoard with logdir: $LOGDIR"
echo "➡️  Open http://localhost:6006 in your browser."
tensorboard --logdir="$LOGDIR" --port=6006 --host=localhost
