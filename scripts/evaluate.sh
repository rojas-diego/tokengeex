#!/bin/bash

vocab_dir="hub/vocab/v2"
input_dir="./hub/data/test/*.bin"
output_dir="hub/eval/v2"

for json_file in "$vocab_dir"/*.json; do
  filename=$(basename "$json_file")
  python scripts/evaluate.py -f "$json_file" -i "$input_dir" -o "$output_dir/$filename" -l tokengeex &
  echo "Evaluating $filename, writing to $output_dir/$filename"
done

wait
