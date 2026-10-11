# Benchmarks

Espetero against the other engines for Apple silicon on one Mac: a MacBook Pro
with an M1 Max (64 GB), macOS 26.5, measured on 2026-10-10 and 2026-10-11.

Engines: Espetero 0.1.0, llama.cpp b11429, mlx-lm 0.32.0 (MLX) with
mlx-vlm 0.7.6 for the images, oMLX 0.7.0, vLLM-metal 0.30.0, TensorFold 1.0.4.

## Method

One server at a time, on the same model files, driven by the same client,
[tools/bench_compare.py](../../tools/bench_compare.py), through each server's
OpenAI-compatible chat API. Greedy decoding with thinking off; each table cell
is the median over its requests.

- **Decode, 1 user**: 8 prompts of about 50 tokens, 512 new tokens each, one
  at a time. *Time to first token* is measured on the same requests.
- **Decode on code**: 6 programming prompts, 512 new tokens each (drafts are
  accepted more often on code).
- **Prefill**: documents of about 1K and 8K tokens with 16 new tokens; the
  prompt's tokens over the time to the first token. The 32K figure comes from
  the 32K context run below.
- **Decode after a context**: 256 new tokens after a document of about 1K, 8K
  and 32K tokens (31.5K for Qwen).
- **Users**: 4, 8 and 16 different requests sent together, 256 new tokens
  each: total tokens per second, and the median time to the first token.
- **Image**: a 1024×768 picture and a 128-token description (788 prompt
  tokens for Qwen in every engine).
- **Memory**: the largest amount of GPU memory in use while the server ran,
  over what was in use before it started.
- **Answers**: 20 short questions with one right answer and 12 Python functions
  run against tests, as a check that each engine runs the model correctly.

Every server had 16 parallel slots and a 40K context, and its defaults
otherwise:

- Espetero: `espetero serve --model M --users 16 --context 40960`, plus
  `--speculation off --no-lookup` in the columns without drafts. *+ MTP* is its
  default: the model's MTP head and prompt lookup. The mlx-community
  Qwen3.8-27B files have no MTP head, so `--mtp` borrows the one unsloth ships
  for the GGUF (`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`).
- llama.cpp: `llama-server -m M -c 81920 -np 16 -kvu -fa on --jinja --mmproj
  mmproj-F16.gguf`; *+ MTP* adds `--spec-type draft-mtp -md
  MTP/mtp-Qwen3.8-27B-Q4_0.gguf`.
- MLX: `mlx_lm.server --model M --decode-concurrency 16 --chat-template-args
  '{"enable_thinking": false}'`; the images through `mlx_vlm.server`. *+ draft
  model* adds `--draft-model` (Qwen3.8-2B, 4-bit MLX) and
  `--num-draft-tokens 3`.
- oMLX: `omlx serve --max-concurrent-requests 16 --paged-ssd-cache-max-size
  4GB` with an empty cache directory; *+ DFlash2* uses
  [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
  and *+ MTP* [mlx-community/Qwen3.8-27B-MTP-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-4bit)
  as drafters.
- vLLM-metal: `vllm serve M --max-model-len 40960 --max-num-seqs 16
  --gpu-memory-utilization 0.75`.
- TensorFold: `tensorfold serve M --context 40960 --parallel 16 --no-thinking
  --no-drafts`. Its native engine has no backend for the Qwen3.6
  mixture-of-experts or Gemma 4 checkpoints yet, and it serves images only
  with `--vision`, which these runs did not pass.

Some runs shared the GPU with another app (the screen's window server, or
another app's GPU process) for 5 to 33 % of the run's time on the GPU, which
can slow the engine down. Their notes say so. Every request, with the replies to the answer checks, is in
[2026-10-11-m1-max.jsonl](2026-10-11-m1-max.jsonl).

## Qwen3.8-27B MLX 4-bit

[mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)

| Measure | Espetero | Espetero + MTP | MLX | MLX + draft model | oMLX | oMLX + MTP | oMLX + DFlash2 | TensorFold |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Load time (s) | 9.9 | 16.3 | **6.4** | – | 12.3 | 13.5 | 16.4 | 53.7 |
| Decode, 1 user (tok/s) | 14.7 | **20.2** | 12.5 | – | 11.2 | 13.6 | 15.1 | 11.0 |
| Time to first token, short prompt (s) | **0.42** | 0.44 | 1.11 | – | 1.34 | 1.24 | 1.36 | 1.03 |
| Decode on code (tok/s) | 15.0 | **31.9** | 12.9 | – | 11.4 | 16.6 | 26.2 | 11.0 |
| Prefill 1K (tok/s) | **148** | 135 | 83 | – | 77 | – | 83 | 55 |
| Prefill 8K (tok/s) | **137** | 124 | 85 | – | 89 | – | 97 | 45 |
| Prefill 32K (tok/s) | **119** | 103 | 80 | – | 83 | – | 86 | 23 |
| Decode after 1K context (tok/s) | 15.4 | **18.3** | 12.9 | – | 11.1 | – | 12.5 | 10.7 |
| Decode after 8K context (tok/s) | 14.9 | **17.2** | 12.4 | – | 11.1 | – | 11.4 | 10.2 |
| Decode after 32K context (tok/s) | **13.6** | 13.0 | 10.8 | – | 9.7 | – | 11.3 | 7.4 |
| 4 users, total (tok/s) | **38.1** | 27.4 | 17.4 | – | 15.8 | – | 18.9 | 10.9 |
| 8 users, total (tok/s) | **63.5** | 46.0 | 18.8 | – | 17.5 | – | 19.6 | 11.0 |
| 16 users, total (tok/s) | **87.2** | 72.3 | 34.3 | – | 26.6 | – | 20.0 | 11.1 |
| 16 users, first token (s) | **3.90** | 4.72 | 13.15 | – | 18.08 | – | 17.51 | 173.55 |
| Image 1024x768, 128 tokens (s) | **14.21** | – | 21.55 | – | 21.16 | – | – | – |
| Peak GPU memory (GB) | 40.1 | 39.9 | 28.5 | – | **28.0** | – | 44.8 | 37.4 |
| Short answers right | 20/20 | 20/20 | 20/20 | – | 20/20 | – | 20/20 | 20/20 |
| Python functions passing tests | 12/12 | 12/12 | 12/12 | – | 12/12 | – | 12/12 | 12/12 |

- Espetero: another app used the GPU during this run (see Method)
- MLX + draft model, load: speculative decoding with a draft model (Qwen3.8-2B 4-bit) fails on every request: 'Speculative decoding requires a trimmable prompt cache (got ArraysCache)' (the model's linear-attention layers)
- TensorFold: another app used the GPU during this run (see Method)
- TensorFold, image: HTTP 400: it serves images only with `--vision`, which these runs did not pass

## Qwen3.8-27B MLX 8-bit

[mlx-community/Qwen3.8-27B-8bit](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit)

| Measure | Espetero | Espetero + MTP |
|:--|--:|--:|
| Load time (s) | **17.0** | 18.7 |
| Decode, 1 user (tok/s) | 10.1 | **17.1** |
| Time to first token, short prompt (s) | 0.56 | **0.50** |
| Decode on code (tok/s) | 10.3 | **27.6** |
| Prefill 1K (tok/s) | **136** | 136 |
| Prefill 8K (tok/s) | 128 | **128** |
| Prefill 32K (tok/s) | 112 | **113** |
| Decode after 1K context (tok/s) | 10.3 | **14.9** |
| Decode after 8K context (tok/s) | 10.0 | **15.4** |
| Decode after 32K context (tok/s) | 9.2 | **13.4** |
| 4 users, total (tok/s) | 27.3 | **27.4** |
| 8 users, total (tok/s) | 47.0 | **47.1** |
| 16 users, total (tok/s) | **74.5** | 74.0 |
| 16 users, first token (s) | **4.29** | 4.40 |
| Image 1024x768, 128 tokens (s) | 18.46 | – |
| Peak GPU memory (GB) | **43.7** | 44.8 |
| Short answers right | 20/20 | 20/20 |
| Python functions passing tests | 12/12 | 12/12 |

## Qwen3.8-27B GGUF UD-Q6_K

[unsloth/Qwen3.8-27B-GGUF:UD-Q6_K](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)

| Measure | Espetero | Espetero + MTP | llama.cpp | llama.cpp + MTP |
|:--|--:|--:|--:|--:|
| Load time (s) | 23.7 | 25.9 | **4.4** | 5.9 |
| Decode, 1 user (tok/s) | 9.1 | **15.6** | 6.9 | 11.5 |
| Time to first token, short prompt (s) | **0.52** | 0.52 | 0.92 | 0.94 |
| Decode on code (tok/s) | 9.3 | **25.8** | 7.0 | 15.5 |
| Prefill 1K (tok/s) | **124** | 120 | 90 | 91 |
| Prefill 8K (tok/s) | **116** | 113 | 90 | 93 |
| Prefill 32K (tok/s) | 103 | **103** | 82 | 82 |
| Decode after 1K context (tok/s) | 9.4 | **14.3** | 7.1 | 10.0 |
| Decode after 8K context (tok/s) | 9.2 | **14.1** | 7.0 | 10.2 |
| Decode after 32K context (tok/s) | 8.7 | **14.5** | 6.6 | 9.0 |
| 4 users, total (tok/s) | 25.7 | **26.6** | 16.4 | 23.7 |
| 8 users, total (tok/s) | 43.0 | **44.5** | 25.3 | 29.4 |
| 16 users, total (tok/s) | 66.9 | **67.6** | 31.8 | 32.1 |
| 16 users, first token (s) | 6.02 | **4.92** | 10.81 | 12.05 |
| Image 1024x768, 128 tokens (s) | **20.51** | – | 27.35 | – |
| Peak GPU memory (GB) | 42.2 | 42.9 | **30.0** | 39.6 |
| Short answers right | 20/20 | 20/20 | 20/20 | 20/20 |
| Python functions passing tests | 12/12 | 12/12 | 12/12 | 12/12 |

- llama.cpp: another app used the GPU during this run (see Method)

## Qwen3.6-35B-A3B MLX 4-bit

[mlx-community/Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit)

| Measure | Espetero | MLX | oMLX | vLLM-metal | TensorFold |
|:--|--:|--:|--:|--:|--:|
| Load time (s) | 15.2 | **5.8** | 10.7 | 60.3 | – |
| Decode, 1 user (tok/s) | **105.7** | 58.2 | 80.6 | 44.5 | – |
| Time to first token, short prompt (s) | **0.09** | 0.23 | 0.26 | 0.18 | – |
| Decode on code (tok/s) | **103.6** | 58.3 | 78.3 | 44.6 | – |
| Prefill 1K (tok/s) | **977** | 754 | 610 | 710 | – |
| Prefill 8K (tok/s) | **945** | 663 | 676 | 528 | – |
| Prefill 32K (tok/s) | **673** | 550 | 479 | 297 | – |
| Decode after 1K context (tok/s) | **103.1** | 57.6 | 79.5 | 42.5 | – |
| Decode after 8K context (tok/s) | **97.3** | 53.1 | 69.8 | 34.0 | – |
| Decode after 32K context (tok/s) | **84.5** | 40.8 | 58.7 | 22.8 | – |
| 4 users, total (tok/s) | **146.3** | 100.3 | 100.9 | 89.3 | – |
| 8 users, total (tok/s) | **194.0** | 118.1 | 114.4 | 112.8 | – |
| 16 users, total (tok/s) | **247.2** | 158.4 | 144.5 | 150.3 | – |
| 16 users, first token (s) | **0.86** | 1.92 | 2.97 | 1.59 | – |
| Image 1024x768, 128 tokens (s) | **3.04** | 3.58 | 3.43 | – | – |
| Peak GPU memory (GB) | 40.0 | **25.1** | 32.2 | 44.9 | – |
| Short answers right | 20/20 | 20/20 | 20/20 | 20/20 | – |
| Python functions passing tests | 12/12 | 12/12 | 12/12 | 12/12 | – |

- vLLM-metal, image: the server refused the image request (HTTP 400)
- TensorFold, load: its native engine has no backend for this checkpoint yet

## Qwen3.6-35B-A3B MLX 8-bit

[mlx-community/Qwen3.6-35B-A3B-8bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-8bit)

| Measure | Espetero |
|:--|--:|
| Load time (s) | 20.2 |
| Decode, 1 user (tok/s) | 72.2 |
| Time to first token, short prompt (s) | 0.11 |
| Decode on code (tok/s) | 70.9 |
| Prefill 1K (tok/s) | 858 |
| Prefill 8K (tok/s) | 860 |
| Prefill 32K (tok/s) | 645 |
| Decode after 1K context (tok/s) | 70.9 |
| Decode after 8K context (tok/s) | 68.5 |
| Decode after 32K context (tok/s) | 62.1 |
| 4 users, total (tok/s) | 120.5 |
| 8 users, total (tok/s) | 157.6 |
| 16 users, total (tok/s) | 207.5 |
| 16 users, first token (s) | 0.84 |
| Image 1024x768, 128 tokens (s) | 3.16 |
| Peak GPU memory (GB) | 46.9 |
| Short answers right | 20/20 |
| Python functions passing tests | 12/12 |

## Qwen3.6-35B-A3B GGUF UD-Q6_K

[unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q6_K](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)

| Measure | Espetero | llama.cpp |
|:--|--:|--:|
| Load time (s) | 40.4 | **15.6** |
| Decode, 1 user (tok/s) | **59.6** | 53.8 |
| Time to first token, short prompt (s) | **0.10** | 0.22 |
| Decode on code (tok/s) | **59.5** | 52.9 |
| Prefill 1K (tok/s) | **943** | 630 |
| Prefill 8K (tok/s) | **829** | 627 |
| Prefill 32K (tok/s) | **623** | 480 |
| Decode after 1K context (tok/s) | **59.2** | 52.9 |
| Decode after 8K context (tok/s) | **57.4** | 49.5 |
| Decode after 32K context (tok/s) | **52.9** | 41.6 |
| 4 users, total (tok/s) | **113.4** | 85.8 |
| 8 users, total (tok/s) | **145.9** | 99.3 |
| 16 users, total (tok/s) | **189.7** | 105.1 |
| 16 users, first token (s) | **0.87** | 1.98 |
| Image 1024x768, 128 tokens (s) | **3.94** | 4.69 |
| Peak GPU memory (GB) | 44.1 | **32.1** |
| Short answers right | 20/20 | 20/20 |
| Python functions passing tests | 12/12 | 12/12 |

## Gemma 4 26B-A4B MLX 4-bit

[mlx-community/gemma-4-26b-a4b-it-4bit](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit)

| Measure | Espetero | MLX | oMLX | vLLM-metal |
|:--|--:|--:|--:|--:|
| Load time (s) | 10.3 | **7.8** | 13.5 | 55.8 |
| Decode, 1 user (tok/s) | **81.4** | 49.9 | 60.5 | 47.9 |
| Time to first token, short prompt (s) | **0.08** | 0.33 | 0.32 | 0.23 |
| Decode on code (tok/s) | **81.1** | 48.6 | 58.8 | 47.9 |
| Prefill 1K (tok/s) | **824** | 428 | 504 | 508 |
| Prefill 8K (tok/s) | **706** | 420 | 531 | 247 |
| Prefill 32K (tok/s) | 471 | 452 | **477** | 88 |
| Decode after 1K context (tok/s) | **73.6** | 45.7 | 56.6 | 43.0 |
| Decode after 8K context (tok/s) | **71.2** | 42.8 | 50.8 | 23.6 |
| Decode after 32K context (tok/s) | **64.4** | 33.9 | 36.5 | 8.8 |
| 4 users, total (tok/s) | **124.7** | 74.8 | 73.5 | 75.6 |
| 8 users, total (tok/s) | **174.8** | 87.2 | 84.2 | 86.4 |
| 16 users, total (tok/s) | **244.5** | 121.0 | 109.9 | 118.8 |
| 16 users, first token (s) | **0.84** | 2.13 | 4.21 | 1.99 |
| Image 1024x768, 128 tokens (s) | – | – | 3.23 | – |
| Peak GPU memory (GB) | 36.3 | **22.5** | 23.8 | 44.8 |
| Short answers right | 20/20 | 20/20 | 20/20 | 20/20 |
| Python functions passing tests | 11/12 | 11/12 | 12/12 | 11/12 |

- Espetero, image: Espetero runs Gemma 4 text only so far (HTTP 422)
- vLLM-metal, image: the server refused the image request (HTTP 400)

## Gemma 4 26B-A4B GGUF UD-Q4_K_M

[unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_M](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)

| Measure | Espetero | llama.cpp |
|:--|--:|--:|
| Load time (s) | 21.4 | **9.7** |
| Decode, 1 user (tok/s) | **53.5** | 51.6 |
| Time to first token, short prompt (s) | **0.09** | 0.23 |
| Decode on code (tok/s) | **53.5** | 50.8 |
| Prefill 1K (tok/s) | **757** | 595 |
| Prefill 8K (tok/s) | **654** | 517 |
| Prefill 32K (tok/s) | **455** | 279 |
| Decode after 1K context (tok/s) | **50.3** | 49.5 |
| Decode after 8K context (tok/s) | **48.9** | 46.3 |
| Decode after 32K context (tok/s) | **45.8** | 41.0 |
| 4 users, total (tok/s) | **105.0** | 89.3 |
| 8 users, total (tok/s) | **148.9** | 104.6 |
| 16 users, total (tok/s) | **209.9** | 106.4 |
| 16 users, first token (s) | **0.89** | 1.60 |
| Image 1024x768, 128 tokens (s) | – | 4.25 |
| Peak GPU memory (GB) | 36.7 | **23.5** |
| Short answers right | 20/20 | 20/20 |
| Python functions passing tests | 11/12 | 11/12 |

- Espetero, image: Espetero runs Gemma 4 text only so far (HTTP 422)

## Gemma 4 31B MLX 4-bit

[mlx-community/gemma-4-31b-it-4bit](https://huggingface.co/mlx-community/gemma-4-31b-it-4bit)

| Measure | Espetero | MLX | oMLX | vLLM-metal | TensorFold |
|:--|--:|--:|--:|--:|--:|
| Load time (s) | 13.4 | **7.3** | 12.6 | 95.6 | – |
| Decode, 1 user (tok/s) | **10.5** | 9.6 | 9.5 | 9.6 | – |
| Time to first token, short prompt (s) | **0.54** | 1.18 | 1.50 | 1.05 | – |
| Decode on code (tok/s) | **10.7** | 9.8 | 9.6 | 9.8 | – |
| Prefill 1K (tok/s) | **110** | 69 | 68 | 66 | – |
| Prefill 8K (tok/s) | **96** | 68 | 69 | 46 | – |
| Prefill 32K (tok/s) | **72** | 63 | 64 | 19 | – |
| Decode after 1K context (tok/s) | **10.4** | 9.5 | 9.3 | 9.2 | – |
| Decode after 8K context (tok/s) | **10.3** | 8.9 | 8.6 | 5.7 | – |
| Decode after 32K context (tok/s) | **9.4** | 6.9 | 6.6 | 2.2 | – |
| 4 users, total (tok/s) | **28.6** | 13.2 | 12.9 | 12.2 | – |
| 8 users, total (tok/s) | **29.5** | 16.6 | 15.7 | 14.9 | – |
| 16 users, total (tok/s) | **33.0** | 29.5 | 26.1 | 27.6 | – |
| 16 users, first token (s) | 47.35 | 13.53 | 17.93 | **12.58** | – |
| Image 1024x768, 128 tokens (s) | – | **17.59** | 18.07 | – | – |
| Peak GPU memory (GB) | **39.0** | 44.7 | 41.8 | 45.0 | – |
| Short answers right | 20/20 | 20/20 | 20/20 | 20/20 | – |
| Python functions passing tests | 12/12 | 12/12 | 12/12 | 12/12 | – |

- Espetero, image: Espetero runs Gemma 4 text only so far (HTTP 422)
- vLLM-metal, image: the server refused the image request (HTTP 400)
- TensorFold, load: its native engine has no backend for this checkpoint yet

## Gemma 4 31B GGUF Q6_K

[unsloth/gemma-4-31B-it-GGUF:Q6_K](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF)

| Measure | Espetero |
|:--|--:|
| Load time (s) | 56.2 |
| Decode, 1 user (tok/s) | 8.0 |
| Time to first token, short prompt (s) | 0.71 |
| Decode on code (tok/s) | 8.2 |
| Prefill 1K (tok/s) | 105 |
| Prefill 8K (tok/s) | 93 |
| Prefill 32K (tok/s) | 72 |
| Decode after 1K context (tok/s) | 8.1 |
| Decode after 8K context (tok/s) | 7.0 |
| Decode after 32K context (tok/s) | 7.5 |
| 4 users, total (tok/s) | 23.3 |
| 8 users, total (tok/s) | 25.9 |
| 16 users, total (tok/s) | 29.7 |
| 16 users, first token (s) | 50.48 |
| Peak GPU memory (GB) | 40.9 |
| Short answers right | 20/20 |
| Python functions passing tests | 12/12 |

- Espetero, image: Espetero runs Gemma 4 text only so far (HTTP 422)
