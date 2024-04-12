# Ablation Study

- [x] Data size
  - 100%
  - 10%
- [x] Idioms
  - [ ]
- [x] BPE
  - +0.2 to +0.3 cpt for base 16k, 65k, 131k
- [x] Added tokens & suggested tokens
- [x] Initial vocabulary size

```bash
python scripts/evaluate.py -f hub/vocab/base-16k-10pct-i500k.json -i './hub/data/test/*.bin' -o hub/eval/base-16k-10pct-i500k.log -l tokengeex &
python scripts/evaluate.py -f hub/vocab/base-65k-10pct-i500k.json -i './hub/data/test/*.bin' -o hub/eval/base-65k-10pct-i500k.log -l tokengeex &
python scripts/evaluate.py -f hub/vocab/base-131k-10pct-i500k.json -i './hub/data/test/*.bin' -o hub/eval/base-131k-10pct-i500k.log -l tokengeex &
python scripts/evaluate.py -f hub/vocab/base-16k-10pct-i1M.json -i './hub/data/test/*.bin' -o hub/eval/base-16k-10pct-i1M.log -l tokengeex &
python scripts/evaluate.py -f hub/vocab/base-65k-10pct-i1M.json -i './hub/data/test/*.bin' -o hub/eval/base-65k-10pct-i1M.log -l tokengeex &
python scripts/evaluate.py -f hub/vocab/base-131k-10pct-i1M.json -i './hub/data/test/*.bin' -o hub/eval/base-131k-10pct-i1M.log -l tokengeex &

python scripts/evaluate.py -f hub/vocab/base-16k-10pct-bpe.json -i './hub/data/test/*.bin' -o hub/eval/base-16k-10pct-bpe.log -l tokengeex &
python scripts/evaluate.py -f hub/vocab/base-65k-10pct-bpe.json -i './hub/data/test/*.bin' -o hub/eval/base-65k-10pct-bpe.log -l tokengeex &
python scripts/evaluate.py -f hub/vocab/base-131k-10pct-bpe.json -i './hub/data/test/*.bin' -o hub/eval/base-131k-10pct-bpe.log -l tokengeex &
```

```zsh
for vocab in "base-131k-10pct-i1M" "base-65k-10pct-i1M" "base-16k-10pct-i1M"; do
  RUST_LOG=info RAYON_NUM_THREADS=120 tokengeex bpe \
    --output ./${vocab}-bpe.json \
    --vocab ./${vocab}.json \
    --num-merges 1000 \
    --step 100 \
    --score-scale-factor 0.9 \
    --max-merge-length 16 \
    --ignore '^$' \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.3 --test ${lang}:./hub/data/test/${lang}.bin "; done)
done
```

```zsh
for inicfg in "-i500k:500000" "-i1M:1000000"; do
  for sizecfg in "-16k:16384" "-65k:65536" "-131k:131072"; do
    IFS=':' read -r -a splits <<< "$inicfg"
      iniext=${splits[0]}
      inisize=${splits[1]}

    IFS=':' read -r -a splits <<< "$sizecfg"
      sizeext=${splits[0]}
      size=${splits[1]}

    echo "Training base${sizeext}-10pct${iniext}.json"

    RUST_LOG=debug RAYON_NUM_THREADS=120 tokengeex train \
      --model "unigram" \
      --output "base${sizeext}-10pct${iniext}.json" \
      --logfile "base${sizeext}-10pct${iniext}.log" \
      --vocab-size ${size} \
      --processor "nfc" \
      --processor "crlf" \
      --initial-vocab-max-token-length 32 \
      --initial-vocab-size ${inisize} \
      --initial-vocab-insert-probability 0.01 \
      --initial-vocab-allow "$(cat data/base.regex)" \
      --unigram-shrinking-factor 0.8 \
      --unigram-num-sub-iterations 2 \
      --unigram-sample-regularization "log" \
      --added-tokens-file "./hub/tokens/base/added.json" \
      --suggested-tokens-file "./hub/tokens/base/suggested.json" \
      $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.1 --test ${lang}:./hub/data/test/${lang}.bin --suggested-tokens-file ./hub/tokens/base/suggested-${lang}.json "; done)
  done
done
```

```zsh
for sizecfg in "16k:16384" "65k:65536" "131k:131072"; do
  IFS=':' read -r -a splits <<< "$sizecfg"
    sizeext=${splits[0]}
    size=${splits[1]}

  echo "Training base${sizeext}-10pct.json"

  RUST_LOG=debug RAYON_NUM_THREADS=120 tokengeex train \
    --model "unigram" \
    --output "base-${sizeext}-10pct-gpt4.json" \
    --logfile "base-${sizeext}-10pct-gpt4.log" \
    --vocab-size ${size} \
    --processor "nfc" \
    --processor "crlf" \
    --initial-vocab-max-token-length 32 \
    --initial-vocab-size 1000000 \
    --initial-vocab-insert-probability 0.01 \
    --initial-vocab-allow "$(cat data/gpt4.regex)" \
    --unigram-shrinking-factor 0.8 \
    --unigram-num-sub-iterations 2 \
    --unigram-sample-regularization "log" \
    --added-tokens-file "./hub/tokens/base/added.json" \
    --suggested-tokens-file "./hub/tokens/base/suggested.json" \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.1 --test ${lang}:./hub/data/test/${lang}.bin --suggested-tokens-file ./hub/tokens/base/suggested-${lang}.json "; done)
done
```

# Training Baselines

```bash
# SentencePiece
python scripts/trainbpe.py -l sentencepiece -v 16384 -o sp-bpe-16k-10pct -i ./hub/data/train -p 0.1
scp zhipu-qinkai-a800:/data/workspace/luojiesi/tokengeex/sp-bpe-16k-10pct.model hub/vocab/sp-bpe-16k-10pct.model

# HuggingFace
python scripts/trainbpe.py -l huggingface -v 16384 -o hf-bpe-16k-10pct.json -i ./hub/data/train -p 0.1
scp zhipu-qinkai-a800:/data/workspace/luojiesi/tokengeex/hf-bpe-16k-10pct.json hub/vocab/hf-bpe-16k-10pct.json
```

# Evaluate Baselines

```zsh
for pct in "" "-10pct"; do
  for size in "-16k" "-65k" "-131k"; do
    # Define tuples as strings with a common delimiter, here we use ':'
    types=("sp-bpe:model:sentencepiece" "hf-bpe:json:tokenizers" "base:json:tokengeex" "capcode:json:tokengeex")

    for type in "${types[@]}"; do
      # Split the string into an array using ':' as the delimiter
      IFS=':' read -r -A type_info <<< "$type"
      slug=${type_info[1]}
      ext=${type_info[2]}
      lib=${type_info[3]}

      echo "Running config: ${slug}${size}${pct}.${ext}"

      python scripts/evaluate.py -f hub/vocab/${slug}${size}${pct}.${ext} -i './hub/data/test/*.bin' -o hub/eval/${slug}${size}${pct}.log -l ${lib} &
    done
  done
done
```

# Train 10pct

Experiments with limited data.

```bash
RUST_LOG=debug RAYON_NUM_THREADS=120 tokengeex train \
    --model 'unigram' \
    --output 'base-16k-10pct.json' \
    --logfile 'base-16k-10pct.log' \
    --vocab-size 16536 \
    --processor 'nfc' \
    --processor 'crlf' \
    --initial-vocab-max-token-length 32 \
    --initial-vocab-size 3000000 \
    --initial-vocab-insert-probability 0.01 \
    --initial-vocab-allow "$(cat data/base.regex)" \
    --unigram-shrinking-factor 0.8 \
    --unigram-num-sub-iterations 2 \
    --unigram-sample-regularization 'log' \
    --added-tokens-file './hub/tokens/base/added.json' \
    --suggested-tokens-file './hub/tokens/base/suggested.json' \
    $(for lang in infilling assembly cuda hcl kotlin php shell xml c-sharp dart html powershell sql yaml c diff java lua python swift zig chinese-markdown dockerfile javascript makefile r tex cmake elixir json markdown ruby toml cpp go jsx pascal rust typescript css haskell julia perl scala vue; do echo "--train ${lang}:./hub/data/train/${lang}.bin:0.1 --test ${lang}:./hub/data/test/${lang}.bin --suggested-tokens-file ./hub/tokens/base/suggested-${lang}.json "; done)
```

10x increase in corpus size bears no effect on the compression of SentencePiece tokenizers across sizes 16k, 65k, 131k.

We notice an ever so slight drop in performance for base-131k vs base-131k-10pct. However, we observe that the token frequency distributions differ. That of base-131k is smoother indicating that the least frequently occuring tokens in base-131k-10pct are results from overfitting.

# Added tokens & suggested tokens

Use the token frequency buckets and BPE to show that few tokens constitute the majority of tokens. Hence few tokens can greatly benefit the overall compression.

# Comments

Re-iterate that BPE generates many junk tokens along the way.

# Better handling of numbers

```
1|234|567|890
```

# Pipeline

- Generate Initial Vocabulary
  - Added Tokens
  - Suggested Tokens
  - Train Sources
  - Allow
- Merge
  - Train Sources
    - Per Source?
  - Test Sources
  - Allow
- Prune
  - Train Sources
  - Test Sources
- Edit
  - Train Sources
  - Test Sources
  - Remove
  - Add
- Combine
  - Vocab A
  - Vocab B
  - Train Sources

# Neighbouring tokens analysis

Look at how many unique neighbours each token has on average. Ref SAGE paper.

# Add min frequency limit

Prevent entries from being considered in the original vocabulary if they occur less than N times.
