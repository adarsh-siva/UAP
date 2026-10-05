#!/bin/bash
# Start the local Ollama server (installed in ~/ollama) in the background, if not already running.
OLLAMA=${OLLAMA_BIN:-$HOME/ollama/bin/ollama}
if curl -s http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
  echo "ollama already running: $(curl -s http://127.0.0.1:11434/api/version)"
  exit 0
fi
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_CONTEXT_LENGTH=32768 \
  setsid nohup "$OLLAMA" serve > "$HOME/ollama/serve.log" 2>&1 < /dev/null &
until curl -s http://127.0.0.1:11434/api/version >/dev/null 2>&1; do sleep 1; done
echo "ollama started; compute devices:"
grep "inference compute" "$HOME/ollama/serve.log" | sed 's/.*name=/  /; s/ description=/ (/; s/ libdirs.*total=/, /; s/ available.*/)/'
