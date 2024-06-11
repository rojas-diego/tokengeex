#!/bin/bash

vocabdir="hub/vocab/v2"
outdir="hub/eval/v2"

for json_file in "$vocabdir"/*.json; do
  filename=$(basename "$json_file")

  # If the output file already exists, skip
  if [ -f "$outdir/$filename" ]; then
    echo "Skipping $filename, output file already exists"
    continue
  fi

  echo "Evaluating $filename, writing to $outdir/$filename"
  python scripts/evaluate.py -f "$json_file" -i "./hub/data/test/*.bin" -o "$outdir/$filename" -l tokengeex &
done

wait
