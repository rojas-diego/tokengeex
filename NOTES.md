# Ablation Study

- Data size
  - 100%
  - 10%
- Multi-word
- Idioms
- Regualarization
- BPE

# Training Baselines

```bash
# SentencePiece
python scripts/trainbpe.py -l sentencepiece -v 16384 -o sp-bpe-16k-10pct -i ./hub/data/train -p 0.1
scp zhipu-qinkai-a800:/data/workspace/luojiesi/tokengeex/sp-bpe-16k-10pct.model hub/vocab/sp-bpe-16k-10pct.model

python scripts/trainbpe.py -l sentencepiece -v 65536 -o sp-bpe-65k-10pct -i ./hub/data/train -p 0.1
scp zhipu-qinkai-a800:/data/workspace/luojiesi/tokengeex/sp-bpe-65k-10pct.model hub/vocab/sp-bpe-65k-10pct.model

python scripts/trainbpe.py -l sentencepiece -v 131072 -o sp-bpe-131k-10pct -i ./hub/data/train -p 0.1
scp zhipu-qinkai-a800:/data/workspace/luojiesi/tokengeex/sp-bpe-131k-10pct.model hub/vocab/sp-bpe-131k-10pct.model

python scripts/trainbpe.py -l sentencepiece -v 16384 -o sp-bpe-16k.json -i ./hub/data/train
scp zhipu-qinkai-a800:/data/workspace/luojiesi/tokengeex/sp-bpe-16k.model hub/vocab/sp-bpe-16k.model

python scripts/trainbpe.py -l sentencepiece -v 65536 -o sp-bpe-65k.json -i ./hub/data/train
scp zhipu-qinkai-a800:/data/workspace/luojiesi/tokengeex/sp-bpe-65k.model hub/vocab/sp-bpe-65k.model

python scripts/trainbpe.py -l sentencepiece -v 131072 -o sp-bpe-131k.json -i ./hub/data/train
scp zhipu-qinkai-a800:/data/workspace/luojiesi/tokengeex/sp-bpe-131k.model hub/vocab/sp-bpe-131k.model

# HuggingFace
python scripts/trainbpe.py -l huggingface -v 16384 -o hf-bpe-16k-10pct.json -i ./hub/data/train -p 0.1
scp zhipu-qinkai-a800:/data/workspace/luojiesi/tokengeex/hf-bpe-131k-10pct.json hub/vocab/hf-bpe-131k-10pct.json

python scripts/trainbpe.py -l huggingface -v 65536 -o hf-bpe-65k-10pct.json -i ./hub/data/train -p 0.1
scp zhipu-qinkai-a800:/data/workspace/luojiesi/tokengeex/hf-bpe-131k-10pct.json hub/vocab/hf-bpe-131k-10pct.json

python scripts/trainbpe.py -l huggingface -v 131072 -o hf-bpe-131k-10pct.json -i ./hub/data/train -p 0.1
scp zhipu-qinkai-a800:/data/workspace/luojiesi/tokengeex/hf-bpe-131k-10pct.json hub/vocab/hf-bpe-131k-10pct.json

python scripts/trainbpe.py -l huggingface -v 16384 -o hf-bpe-16k.json -i ./hub/data/train
scp zhipu-qinkai-a800:/data/workspace/luojiesi/tokengeex/hf-bpe-16k.json hub/vocab/hf-bpe-16k.json

python scripts/trainbpe.py -l huggingface -v 65536 -o hf-bpe-65k.json -i ./hub/data/train
scp zhipu-qinkai-a800:/data/workspace/luojiesi/tokengeex/hf-bpe-65k.json hub/vocab/hf-bpe-65k.json

python scripts/trainbpe.py -l huggingface -v 131072 -o hf-bpe-131k.json -i ./hub/data/train
scp zhipu-qinkai-a800:/data/workspace/luojiesi/tokengeex/hf-bpe-131k.json hub/vocab/hf-bpe-131k.json
```
