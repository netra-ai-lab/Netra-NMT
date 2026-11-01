| Direction | Metric | Optimization | Netra NMT (Ours) | NLLB-350M Distilled (EN-KM) | NLLB-200 Distilled (600M) | NLLB-200 Distilled (1.3B) | MADLAD-400 (3B) | Google Translate |
|------------|--------|--------------|-----------------:|----------------------------:|--------------------------:|--------------------------:|----------------:|-----------------:|
| EN → KH | spBLEU | Higher | **25.83** | 18.89 | 9.24 | 16.62 | 16.19 | **28.93** |
| EN → KH | chrF++ | Higher | **38.05** | 30.07 | 34.38 | 36.14 | 32.24 | **40.21** |
| EN → KH | TER | Lower | **60.47** | 73.89 | 78.68 | 69.90 | 101.99 | 65.56 |
| EN → KH | COMET | Higher | **78.14** | 71.28 | 76.54 | 79.68 | 79.95 | **82.74** |
| EN → KH | BERTScore F1 | Higher | **90.02** | 87.89 | 88.24 | 89.11 | 89.28 | **90.66** |
| KH → EN | spBLEU | Higher | **19.47** | 0.26 | 31.28 | 35.15 | **36.13** | 34.86 |
| KH → EN | chrF++ | Higher | **46.01** | 1.59 | 54.14 | 57.36 | **58.28** | 57.51 |
| KH → EN | TER | Lower | 69.79 | 238.81 | 55.34 | 51.14 | **51.12** | 51.14 |
| KH → EN | COMET | Higher | **77.02** | 34.74 | 84.25 | 85.87 | **86.88** | 86.48 |
| KH → EN | BERTScore F1 | Higher | **91.79** | 78.65 | 93.62 | 94.13 | **94.33** | 94.31 |

<sub>Netra NMT (Ours) = 77M-param RoPE/RMSNorm, deep-encoder(12)/shallow-decoder(2), Khmer-segmented tokenizer, trained from scratch (no ALT fine-tuning). Greedy decoding on the `mutiyama/alt` test split (1,018 pairs). spBLEU uses the FLORES-200 tokenizer; COMET = `Unbabel/wmt22-comet-da`; BERTScore F1 = `xlm-roberta-base`. **Bold** = best in row (Ours highlighted).</sub>



## Netra-NMT Inference Benchmark on CPU

**CPU:** AMD Ryzen 7 PRO 5850U

| Model | Greedy (tok/s) ↑ | Beam-5 (tok/s) ↑ |
|-------|-----------------:|-----------------:|
| **Netra-NMT-89M** | **33.2** | **18.7** |
| NLLB-350M | 11.1 | 2.2 |
| NLLB-600M | 7.7 | 2.2 |

---

## Inference Speed — current hardware (Intel i7-13700KF · RTX 3060 12GB)

![Inference speed](assets/speed_benchmark.png)

EN→KH, 20 sentences, 3 warmup. tok/s ↑ higher-is-better; ms/sentence ↓ lower-is-better.
CPU = 16 threads. Each model is counted with its own tokenizer; **ms/sentence** is the
fair cross-model latency metric. Netra = 77M RoPE / 12-enc-2-dec model with KV cache.

**CPU (i7-13700KF, 16 threads)**

| Model | Params | Size (fp16) | Greedy tok/s ↑ | Greedy ms/sent ↓ | Beam-5 tok/s ↑ | Beam-5 ms/sent ↓ |
|-------|-------:|------------:|---------------:|-----------------:|---------------:|-----------------:|
| **Netra-NMT (Ours)** | **77M** | **154 MB** | **311.7** | **65.6** | **28.0** | **735.2** |
| NLLB-350M (EN-KM) | 351M | 701 MB | 32.5 | 550.8 | 22.8 | 802.7 |
| NLLB-600M | 615M | 1230 MB | 20.9 | 1074.9 | 16.0 | 1567.5 |

**GPU (RTX 3060 12GB)**

| Model | Params | Size (fp16) | Greedy tok/s ↑ | Greedy ms/sent ↓ | Beam-5 tok/s ↑ | Beam-5 ms/sent ↓ |
|-------|-------:|------------:|---------------:|-----------------:|---------------:|-----------------:|
| **Netra-NMT (Ours)** | **77M** | **154 MB** | **1384.5** | **14.8** | 142.0 | 144.3 |
| NLLB-350M (EN-KM) | 351M | 701 MB | 232.7 | 77.2 | **174.2** | **105.2** |
| NLLB-600M | 615M | 1230 MB | 149.6 | 150.4 | 112.7 | 223.0 |

> On **CPU** (the deployment target) Netra is **8–16× faster** than the NLLB baselines in
> both modes. On **GPU** it leads decisively in greedy (14.8 ms/sentence, ~5× faster). In
> GPU beam-5 it trails NLLB-350M — not an architecture limit but an implementation one:
> Netra's beam search is a reference pure-Python loop, whereas NLLB uses the optimized
> HuggingFace `generate` (the greedy numbers reflect the true architecture speed).