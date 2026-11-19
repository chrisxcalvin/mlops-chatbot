#!/bin/bash
# Resumable download of the 3 quantised local models (~2.3 GB total). Safe to re-run.
# Files are downloaded as *.part and renamed only when complete, so the app never sees half-files.
cd "$(dirname "$0")"; mkdir -p models
get(){
  [ -f "models/$1" ] && return 0
  until curl -L -C - --retry 5 -m 3000 -o "models/$1.part" "$2"; do sleep 3; done
  mv "models/$1.part" "models/$1"
}
get qwen2.5-0.5b-instruct-q4_k_m.gguf "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
get Llama-3.2-1B-Instruct-Q4_K_M.gguf "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
get qwen2.5-1.5b-instruct-q4_k_m.gguf "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
echo ALL_DONE
