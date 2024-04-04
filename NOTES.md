# Ablation Study

- Data size
  - 100%
  - 10%
- Multi-word
- Idioms
- Regualarization
- BPE
- Added tokens & suggested tokens

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

```bash
# SentencePiece
python scripts/evaluate.py -f hub/vocab/sp-bpe-16k-10pct.model -i './hub/data/test/*.bin' -o hub/eval/sp-bpe-16k-10pct.log -l sentencepiece
```

# Train 10pct

Experiments with limited data.

```bash
RUST_LOG=debug RAYON_NUM_THREADS=120 tokengeex train \
    --model 'unigram' \
    --output 'base-131k-10pct.json' \
    --logfile 'base-131k-10pct.log' \
    --vocab-size 131072 \
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

# Added tokens & suggested tokens

Use the token frequency buckets and BPE to show that few tokens constitute the majority of tokens. Hence few tokens can greatly benefit the overall compression.

# Comments

Re-iterate that BPE generates many junk tokens along the way.
