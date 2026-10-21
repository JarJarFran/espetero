# Benchmarks on a Mac Studio M4 Max (128 GB)

Espetero main 70f3bb4 (0.1.2 in preparation) against llama.cpp and mlx-lm on one Mac Studio with an
M4 Max (128 GB), macOS 27.0.1, measured on 2026-10-20 and 2026-10-21.

Engines: Espetero main 70f3bb4 (`espetero --version` still prints 0.1.1), llama.cpp b11429 (Homebrew 0.6.0),
mlx-lm 0.32.0 (MLX) with mlx-vlm 0.7.6 for the images. oMLX, vLLM-metal and TensorFold are not installed on this
Mac and nothing was installed for these runs.

## Method

The method of [README.md](README.md) (the M1 Max runs): one server at a time on the same model files, driven by
[tools/bench_compare.py](../../tools/bench_compare.py) (the copy in main 70f3bb4) through each server's
OpenAI-compatible chat API; greedy decoding with thinking off; each cell is the median over its requests; the same
workloads (`decode,code,prefill,context,users,vision,quality` with `--prefill-tokens 1024,8192`,
`--context-tokens 1024,8192,32768`, `--users 1,4,8,16`), the memory watcher (`--watch`) started before each server,
the load time from `--started-at`. Every server had 16 parallel slots and a 40K context and its defaults otherwise:

- Espetero: `espetero serve --model M --users 16 --context 40960`, plus `--speculation off --no-lookup` in the
  columns without drafts. *+ MTP*: the mlx-community Qwen3.8-27B file has no MTP head, so `--mtp` borrows
  unsloth's `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`; the unsloth GGUF uses its own; Qwen3.8-Flash-Next takes the bf16
  drafter directory `Qwen3.8-Flash-Next-MTP-Drafter-MLX-bf16` (`--mtp DIR`; the oQ4e file's own 4-bit MTP tensors
  are not used by Espetero). Flash-Next keeps every expert resident (the whole 77 GB store fits: 8 slots, 81,920
  KV tokens). DeepSeek-V4.1-Flash-1017 runs with the routed experts and Engram tables on the SSD (an expert cache
  of 93 GB, 6 sequence slots, 245,760 KV tokens, so the 8- and 16-user runs queue); its prompt template is the
  original V4.1-Flash release's `chat_template.jinja` beside the 1017 weights (the 1017 release ships none, and its
  config, tokenizer config and `encoding/encoding.py` are byte-identical to the original's).
- llama.cpp: `llama-server -m M -c 81920 -np 16 -kvu -fa on --jinja --mmproj mmproj-F16.gguf`; *+ MTP* adds
  `--spec-type draft-mtp -md MTP/mtp-Qwen3.8-27B-Q4_0.gguf`.
- MLX: `mlx_lm.server --model M --decode-concurrency 16 --chat-template-args '{"enable_thinking": false}'`; the
  images through `mlx_vlm.server`. No draft model was tried.

Model files: mlx-community/Qwen3.8-27B-4bit, unsloth/Qwen3.8-27B-GGUF:UD-Q6_K and
unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q6_K were downloaded for these runs; mlx-community/Qwen3.6-35B-A3B-4bit,
Jundot/Qwen3.8-Flash-Next-oQ4e-mtp and deepseek-ai/DeepSeek-V4.1-Flash-1017 were already on the Mac (mlx-lm
fetched its own copy of the 35B from the Hub at its first request, which its load time includes).

Another session shares this Mac's GPU under a lock; every run here held it, and the memory watcher's record of
other GPU users is empty for every run in the tables except where a note says otherwise. Every request, with the
replies to the answer checks, is in [2026-10-21-m4-max.jsonl](2026-10-21-m4-max.jsonl).

## The README table

| Model · file | Engine | Decode, tok/s | First token, s | Prompt of 8K tokens, tok/s | Decode at 32K context, tok/s | 8 users at once, tok/s | Photo, s |
|:--|:--|--:|--:|--:|--:|--:|--:|
| **Qwen3.8-27B** · MLX 4-bit | Espetero 0.1.2 (main 70f3bb4) | 29.9 · **59.2** with MTP¹ | 0.20 | 293 | 26.2 | 157.6 | 7.3 |
|  | MLX (mlx-lm 0.32 · mlx-vlm 0.7.6) | 27.9 | 0.40 | 256 | 23.1 | 77.9 | 7.8 |
| **Qwen3.6-35B-A3B** · MLX 4-bit | Espetero 0.1.2 (main 70f3bb4) | 130.0 | 0.05 | 2122 | 109.3 | 423.5 | 1.7 |
|  | MLX (mlx-lm 0.32 · mlx-vlm 0.7.6) | 110.8 | 0.10 | 1848 | 80.9 | 318.3 | 1.8 |
| **Qwen3.8-Flash-Next** (125B-A6B) · MLX oQ4e, experts resident | Espetero 0.1.2 (main 70f3bb4) | 58.7 · **72.9** with MTP | 0.16 | 758 | 54.0 | 127.4 | 3.5 |
|  | MLX (mlx-lm 0.32)² | – | – | – | – | – | – |
| **DeepSeek-V4.1-Flash** (552B-A16B) · FP8/FP4, experts on the SSD | Espetero 0.1.2 (main 70f3bb4)³ | 9.7 | 5.39 | 253 | 10.2 | 6.1 | 43.6 |
| **Qwen3.8-27B** · GGUF UD-Q6_K | Espetero 0.1.2 (main 70f3bb4) | 20.2 · **46.9** with MTP | 0.22 | 279 | 18.3 | 123.9 | 9.6 |
|  | llama.cpp b11429 | 19.3 · **33.6** with MTP | 0.39 | 231 | 17.0 | 73.9 | 10.4 |
| **Qwen3.6-35B-A3B** · GGUF UD-Q6_K | Espetero 0.1.2 (main 70f3bb4) | 90.5 | 0.05 | 1873 | 79.8 | 307.1 | 2.2 |
|  | llama.cpp b11429 | 88.8 | 0.11 | 1509 | 73.8 | 213.6 | 2.4 |

¹ MTP head borrowed from the GGUF (Qwen3.8-27B MLX) or the bf16 drafter directory (Flash-Next).
² mlx-lm 0.32 loads the Flash-Next file but its generation thread dies on the first request (HTTP 404).
³ 6 sequence slots; *8 users* queue, 6 at a time.

Notes on the tables below:

- Qwen3.8-27B MLX, Espetero: the first run of the *Espetero* column shared the GPU with another session's job for
  about a minute; it was repeated alone and the repeat is what the tables and the JSONL hold (the two agreed
  within 0.1 tok/s on decode).
- Flash-Next, *Espetero + MTP*: the bf16 drafter directory. Without `--mtp DIR` the server reports `mtp no` on the
  oQ4e file and *+ MTP* only adds prompt-lookup drafts (58.4 tok/s on the same prompts, not in the tables).
- DeepSeek-V4.1-Flash: a prompt of 256 rows or more streams nearly every routed expert from the SSD (about 290 GB
  at ~7 GB/s), so *Prefill 1K* is 35 tok/s (the ~30 s of reads) while *Prefill 8K* and *32K* reach 253 and 313
  tok/s; the first token of a short prompt (5.4 s) is the experts it reads. With 6 slots, *16 users* wait: the
  median first token is 238 s.
- MLX, Qwen3.6-35B-A3B: `mlx_lm.server` fetched the model from the Hub at its first request (the requested model
  name, not the `--model` path), so its load time (214 s) includes the download. Its *16 users* runs lost 2 of 16
  requests to connection resets on both Qwen models (the 14 that finished are what the row holds).
- mlx-vlm serves only the image column (its own server).

## Qwen3.8-27B MLX 4-bit

| Measure | Espetero + MTP | MLX | mlx-vlm | Espetero |
|:--|--:|--:|--:|--:|
| Load time (s) | 16.3 | 6.6 | 7.1 | 10.8 |
| Decode, 1 user (tok/s) | 59.2 | 27.9 | – | 29.9 |
| Time to first token, short prompt (s) | 0.20 | 0.40 | – | 0.20 |
| Decode on code (tok/s) | 90.4 | 27.9 | – | 30.0 |
| Prefill 1K (tok/s) | – | 251 | – | 293 |
| Prefill 8K (tok/s) | – | 256 | – | 293 |
| Prefill 32K (tok/s) | 251 | 235 | – | 255 |
| Decode after 1K context (tok/s) | 52.6 | 27.6 | – | 29.8 |
| Decode after 8K context (tok/s) | 48.9 | 26.2 | – | 28.6 |
| Decode after 32K context (tok/s) | 38.6 | 23.1 | – | 26.2 |
| 4 users, total (tok/s) | 89.9 | 70.3 | – | 90.2 |
| 8 users, total (tok/s) | 157.3 | 77.9 | – | 157.6 |
| 16 users, total (tok/s) | 170.7 | 74.3 | – | 172.2 |
| 16 users, first token (s) | 2.27 | 2.77 | – | 2.57 |
| Image 1024x768, 128 tokens (s) | – | – | 7.79 | 7.33 |
| Peak GPU memory (GB) | 50.1 | 27.8 | 30.1 | 47.6 |
| Short answers right | 20/20 | 20/20 | – | 20/20 |
| Python functions passing tests | 12/12 | 12/12 | – | 12/12 |

## Qwen3.6-35B-A3B MLX 4-bit

| Measure | Espetero | MLX | mlx-vlm |
|:--|--:|--:|--:|
| Load time (s) | 10.8 | 214.3 | 8.5 |
| Decode, 1 user (tok/s) | 130.0 | 110.8 | – |
| Time to first token, short prompt (s) | 0.05 | 0.10 | – |
| Decode on code (tok/s) | 129.9 | 110.9 | – |
| Prefill 1K (tok/s) | 2242 | 1828 | – |
| Prefill 8K (tok/s) | 2122 | 1848 | – |
| Prefill 32K (tok/s) | 1335 | 1460 | – |
| Decode after 1K context (tok/s) | 128.8 | 109.3 | – |
| Decode after 8K context (tok/s) | 123.1 | 100.8 | – |
| Decode after 32K context (tok/s) | 109.3 | 80.9 | – |
| 4 users, total (tok/s) | 296.2 | 249.5 | – |
| 8 users, total (tok/s) | 423.5 | 318.3 | – |
| 16 users, total (tok/s) | 501.3 | 357.1 | – |
| 16 users, first token (s) | 0.54 | 0.49 | – |
| Image 1024x768, 128 tokens (s) | 1.67 | – | 1.75 |
| Peak GPU memory (GB) | 59.9 | 24.5 | 21.2 |
| Short answers right | 20/20 | 20/20 | – |
| Python functions passing tests | 12/12 | 12/12 | – |

## Qwen3.8-Flash-Next oQ4e MLX

| Measure | Espetero | MLX | Espetero + MTP |
|:--|--:|--:|--:|
| Load time (s) | 32.3 | – | 44.2 |
| Decode, 1 user (tok/s) | 58.7 | – | 72.9 |
| Time to first token, short prompt (s) | 0.16 | – | 0.15 |
| Decode on code (tok/s) | 59.3 | – | 100.5 |
| Prefill 1K (tok/s) | 702 | – | – |
| Prefill 8K (tok/s) | 758 | – | – |
| Prefill 32K (tok/s) | 751 | – | 733 |
| Decode after 1K context (tok/s) | 58.0 | – | 63.9 |
| Decode after 8K context (tok/s) | 54.4 | – | 58.2 |
| Decode after 32K context (tok/s) | 54.0 | – | 61.9 |
| 4 users, total (tok/s) | 113.1 | – | 116.9 |
| 8 users, total (tok/s) | 127.4 | – | 134.5 |
| 16 users, total (tok/s) | 140.1 | – | 151.2 |
| 16 users, first token (s) | 11.97 | – | 11.57 |
| Image 1024x768, 128 tokens (s) | 3.45 | – | – |
| Peak GPU memory (GB) | 84.0 | 0.0 | 87.1 |
| Short answers right | 20/20 | – | 20/20 |
| Python functions passing tests | 12/12 | – | 12/12 |

- MLX, load: the server answers HTTP 404: {"error": "generation thread died"}

## DeepSeek-V4.1-Flash FP8/FP4

| Measure | Espetero |
|:--|--:|
| Load time (s) | 40.8 |
| Decode, 1 user (tok/s) | 9.7 |
| Time to first token, short prompt (s) | 5.39 |
| Decode on code (tok/s) | 8.8 |
| Prefill 1K (tok/s) | 35 |
| Prefill 8K (tok/s) | 253 |
| Prefill 32K (tok/s) | 313 |
| Decode after 1K context (tok/s) | 10.3 |
| Decode after 8K context (tok/s) | 11.0 |
| Decode after 32K context (tok/s) | 10.2 |
| 4 users, total (tok/s) | 5.5 |
| 8 users, total (tok/s) | 6.1 |
| 16 users, total (tok/s) | 6.3 |
| 16 users, first token (s) | 237.68 |
| Image 1024x768, 128 tokens (s) | 43.62 |
| Peak GPU memory (GB) | 103.0 |
| Short answers right | 20/20 |
| Python functions passing tests | 12/12 |

## Qwen3.8-27B GGUF UD-Q6_K

| Measure | Espetero | Espetero + MTP | llama.cpp | llama.cpp + MTP |
|:--|--:|--:|--:|--:|
| Load time (s) | 17.7 | 14.6 | 12.5 | 3.9 |
| Decode, 1 user (tok/s) | 20.2 | 46.9 | 19.3 | 33.6 |
| Time to first token, short prompt (s) | 0.22 | 0.22 | 0.39 | 0.39 |
| Decode on code (tok/s) | 20.2 | 73.6 | 19.3 | 45.7 |
| Prefill 1K (tok/s) | 277 | – | 233 | – |
| Prefill 8K (tok/s) | 279 | – | 231 | – |
| Prefill 32K (tok/s) | 245 | 241 | 207 | 202 |
| Decode after 1K context (tok/s) | 20.1 | 38.0 | 19.3 | 28.5 |
| Decode after 8K context (tok/s) | 19.6 | 39.1 | 18.8 | 27.0 |
| Decode after 32K context (tok/s) | 18.3 | 35.9 | 17.0 | 25.6 |
| 4 users, total (tok/s) | 69.4 | 69.3 | 48.3 | 59.0 |
| 8 users, total (tok/s) | 123.9 | 123.9 | 73.9 | 58.2 |
| 16 users, total (tok/s) | 157.5 | 157.2 | 78.6 | 74.3 |
| 16 users, first token (s) | 2.49 | 2.53 | 4.75 | 5.01 |
| Image 1024x768, 128 tokens (s) | 9.56 | – | 10.45 | – |
| Peak GPU memory (GB) | 72.3 | 72.6 | 29.6 | 39.0 |
| Short answers right | 20/20 | 20/20 | 20/20 | 20/20 |
| Python functions passing tests | 12/12 | 12/12 | 12/12 | 12/12 |

## Qwen3.6-35B-A3B GGUF UD-Q6_K

| Measure | Espetero | llama.cpp |
|:--|--:|--:|
| Load time (s) | 29.8 | 10.5 |
| Decode, 1 user (tok/s) | 90.5 | 88.8 |
| Time to first token, short prompt (s) | 0.05 | 0.11 |
| Decode on code (tok/s) | 90.5 | 88.8 |
| Prefill 1K (tok/s) | 1946 | 1512 |
| Prefill 8K (tok/s) | 1873 | 1509 |
| Prefill 32K (tok/s) | 1235 | 1152 |
| Decode after 1K context (tok/s) | 89.5 | 88.2 |
| Decode after 8K context (tok/s) | 87.1 | 84.6 |
| Decode after 32K context (tok/s) | 79.8 | 73.8 |
| 4 users, total (tok/s) | 211.3 | 167.7 |
| 8 users, total (tok/s) | 307.1 | 213.6 |
| 16 users, total (tok/s) | 388.9 | 231.0 |
| 16 users, first token (s) | 0.55 | 0.98 |
| Image 1024x768, 128 tokens (s) | 2.17 | 2.45 |
| Peak GPU memory (GB) | 74.1 | 31.7 |
| Short answers right | 20/20 | 20/20 |
| Python functions passing tests | 12/12 | 12/12 |

