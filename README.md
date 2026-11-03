# Espetero

**Fast LLM inference for Apple silicon.** A native Metal engine (C++20 and Metal, no Python) that puts every request in one GPU pass: chats, images and classifications grilled together, the way an *espetero* grills a whole cane of sardines over the embers.

*Espetero* (es-peh-TEH-ro) is the cook on the beaches of Málaga who grills sardines threaded on canes, the *espetos*, over a fire in a boat full of sand.

## Try it

```bash
brew install JarJarFran/espetero/espetero                 # builds from source; needs the Xcode Command Line Tools
espetero recommend                                        # the models that fit this Mac, best first
espetero serve --model mlx-community/Qwen3.6-35B-A3B-4bit   # downloads it into the Hugging Face cache the first time
```

Then open `http://127.0.0.1:8080` for the built-in chat, or point any OpenAI, Anthropic or Ollama client at it (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`). More in [Quick start](#quick-start).

## Speed on a MacBook Pro M1 Max (64 GB)

Same model files, same client ([tools/bench_compare.py](tools/bench_compare.py)), one server at a time, medians:

| Model · file | Engine | Decode, tok/s | First token, s | Prompt of 8K tokens, tok/s | Decode at 32K context, tok/s | 8 users at once, tok/s | Photo, s |
|:--|:--|--:|--:|--:|--:|--:|--:|
| **Qwen3.8-27B** · MLX 4-bit | Espetero 0.1.0† | 14.7 · **20.2** with MTP¹ | **0.42** | **137** | **13.6** | **63.5** | **14.2** |
|  | MLX (mlx-lm 0.32 · mlx-vlm 0.7.6) | 12.5 | 1.11 | 85 | 10.8 | 18.8 | 21.6 |
|  | oMLX 0.7.0 | 11.2 · 15.1 with DFlash2 | 1.34 | 89 | 9.7 | 17.5 | 21.2 |
|  | TensorFold 1.0.4²† | 11.0 | 1.03 | 45 | 7.4 | 11.0 | – |
| **Qwen3.8-27B** · GGUF UD-Q6_K | Espetero 0.1.0 | 9.1 · **15.6** with MTP | **0.52** | **116** | **8.7** | **43.0** | **20.5** |
|  | llama.cpp b11429† | 6.9 · 11.5 with MTP | 0.92 | 90 | 6.6 | 25.3 | 27.3 |
| **Qwen3.6-35B-A3B** (MoE) · MLX 4-bit | Espetero 0.1.0 | **105.7** | **0.09** | **945** | **84.5** | **194.0** | **3.0** |
|  | MLX (mlx-lm 0.32 · mlx-vlm 0.7.6) | 58.2 | 0.23 | 663 | 40.8 | 118.1 | 3.6 |
|  | oMLX 0.7.0 | 80.6 | 0.26 | 676 | 58.7 | 114.4 | 3.4 |
|  | vLLM-metal 0.30.0 | 44.5 | 0.18 | 528 | 22.8 | 112.8 | – |
| **Qwen3.6-35B-A3B** (MoE) · GGUF UD-Q6_K | Espetero 0.1.0 | **59.6** | **0.10** | **829** | **52.9** | **145.9** | **3.9** |
|  | llama.cpp b11429 | 53.8 | 0.22 | 627 | 41.6 | 99.3 | 4.7 |
| **Gemma 4 26B-A4B** (MoE) · MLX 4-bit | Espetero 0.1.0 | **81.4** | **0.08** | **706** | **64.4** | **174.8** | –³ |
|  | MLX (mlx-lm 0.32 · mlx-vlm 0.7.6) | 49.9 | 0.33 | 420 | 33.9 | 87.2 | – |
|  | oMLX 0.7.0 | 60.5 | 0.32 | 531 | 36.5 | 84.2 | 3.2 |
|  | vLLM-metal 0.30.0 | 47.9 | 0.23 | 247 | 8.8 | 86.4 | – |
| **Gemma 4 26B-A4B** (MoE) · GGUF UD-Q4_K_M | Espetero 0.1.0 | **53.5** | **0.09** | **654** | **45.8** | **148.9** | –³ |
|  | llama.cpp b11429 | 51.6 | 0.23 | 517 | 41.0 | 104.6 | 4.2 |

Greedy decoding with thinking off. *Decode*: 8 short prompts, 512 new tokens each, one at a time; after the dot, with the engine's own drafts (checked by the full model, so the replies do not change). *First token*: the same prompts. *Prompt of 8K tokens*: prefill speed on a 7.9K-token document. *Decode at 32K context*: 256 new tokens after a 31.5K-token document. *8 users*: 8 different requests at once, 256 new tokens each, in total. *Photo*: a 1024×768 picture and a 128-token description. Every server had 16 parallel slots and a 40K context and its defaults otherwise; Espetero ran without drafts except *with MTP*. Files: [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit), [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), [mlx-community/Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit), [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF), [mlx-community/gemma-4-26b-a4b-it-4bit](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit), [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF). The flags, every measure (load time, prefill at 1K and 32K, 16 users, memory, answers checked), more files (8-bit MLX, Gemma 4 31B) and every request: [docs/benchmarks](docs/benchmarks/README.md). Measured on 2026-10-10 and 11 with macOS 26.5.

**Re-measured with 0.1.2** on 2026-10-26 and 27 ([docs/benchmarks/2026-10-27-m1-max.md](docs/benchmarks/2026-10-27-m1-max.md)): Qwen3.8-27B MLX 4-bit 14.2 · **22.3** tok/s with MTP (mlx-lm 12.8), 8 users 58.6 (mlx-lm 17.8); the UD-Q6_K GGUF 8.9 · **16.7** with MTP (llama.cpp 6.7 · 11.0), 8 users 47.5 (llama.cpp 24.5). **Qwen3.8-Flash-Next** (125B-A6B, 106 GB) runs on this 64 GB Mac with its experts read from the SSD and the ones in use cached in memory (31 GB): 12.9 · **17.8** tok/s with MTP, a first token in 1.6 s, an 8K prompt at 265 tok/s, a photo in 20 s. The SSD bounds it to one or a few users: 8 users together get ~19 tok/s.

† Another app used the GPU during this run (5 to 33 % of the run's time on the GPU), which can slow the engine down.
¹ The mlx-community file has no MTP head; `--mtp` borrows the one unsloth ships for the GGUF.
² TensorFold's native engine has no backend for the Qwen3.6 mixture-of-experts or Gemma 4 checkpoints yet, and it serves images only with `--vision`, which these runs did not pass.
³ Not measured: these runs used Espetero 0.1.0, which ran Gemma 4 text only.

## Speed on a Mac Studio M4 Max (128 GB)

The same client and protocol on a Mac Studio with an M4 Max (128 GB, macOS 27.0.1), measured on 2026-10-26 and 27 with Espetero 0.1.2, medians, every run alone on the GPU:

| Model · file | Engine | Decode, tok/s | First token, s | Prompt of 8K tokens, tok/s | Decode at 32K context, tok/s | 8 users at once, tok/s | Photo, s |
|:--|:--|--:|--:|--:|--:|--:|--:|
| **Qwen3.8-Flash-Next** (125B-A6B) · MLX oQ4e, experts resident | Espetero 0.1.2 | 58.6 · **78.7** with MTP | **0.12** | **770** | **54.3** · 66.1 with MTP | **170.7** | 3.4 |
|  | oMLX 0.7.2¹ | **81.1** with MTP | 0.25 | 547 | – | 110.0 | **3.0** |
|  | MLX (mlx-lm 0.32)² | – | – | – | – | – | – |
| **Qwen3.8-27B** · MLX 4-bit | Espetero 0.1.2 | 30.1 · **59.4** with MTP³ | **0.18** | **294** | **26.3** · 38.7 with MTP | **161.0** | **7.3** |
|  | MLX (mlx-lm 0.32) | 27.5 | 0.41 | 256 | – | 77.7 | – |
|  | oMLX 0.7.2 | 30.1 | 0.42 | 259 | – | 74.0 | 7.7 |
|  | BaseRT 0.3.0 (its own 4-bit build) | 28.7 | 0.39 | 267 | – | 92.4 | 8.3 |
| **Qwen3.8-27B** · GGUF UD-Q6_K | Espetero 0.1.2 | 20.2 · **47.0** with MTP | **0.20** | **280** | **18.4** · 32.6 with MTP | **125.8** | **9.6** |
|  | llama.cpp b11429 | 19.5 · 33.3 with MTP | 0.39 | 231 | 17.2 | 74.0 | 10.4 |
| **DeepSeek-V4.1-Flash** (552B-A16B) · FP8/FP4, experts on the SSD | Espetero 0.1.2⁴ | 9.7 | 5.39 | 253 | 10.2 | 6.1 | 43.6 |

Same protocol as above (greedy, thinking off, 16 parallel slots, 40K context, no drafts except *with MTP*); the other engines ran decode, prefill, 8 and 16 users and the photo, so their empty cells were not run. With 16 users Flash-Next serves 152 tok/s (154 with MTP) and the 27B 175. Files: [Jundot/Qwen3.8-Flash-Next-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp) (every expert resident in memory), [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit), [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (the routed experts and Engram tables read from the SSD). Every measure, the flags and every request: [docs/benchmarks/2026-10-27-m4-max.md](docs/benchmarks/2026-10-27-m4-max.md); the Qwen3.6-35B-A3B rows of 0.1.2's code: [docs/benchmarks/m4-max.md](docs/benchmarks/m4-max.md).

¹ oMLX turns on the oQ4e file's MTP head from its model preset; its column is oMLX as it ships. Greedy replies were identical across repeats.
² mlx-lm 0.32 loads the Flash-Next file but its generation thread dies on the first request (HTTP 404).
³ The mlx-community file has no MTP head; `--mtp` borrows the one unsloth ships for the GGUF. Flash-Next drafts with the oQ4e file's own 4-bit MTP head.
⁴ DeepSeek-V4.1 (measured with 0.1.2's code on 2026-10-21) runs with 6 sequence slots (its state is a slot of its own), so *8 users* queue: 6 at a time. Its first token includes the experts a new prompt reads from the SSD.

## Quick start

```bash
make                                   # Xcode Command Line Tools only
build/espetero serve --model mlx-community/Qwen3.8-27B-4bit
```

`--model` takes a Hugging Face repo id, a local MLX directory or a `.gguf` file. A repo id is read from the Hugging Face cache (`~/.cache/huggingface/hub`, or `$HF_HUB_CACHE` / `$HF_HOME/hub`) and downloaded there first when it is missing, by Espetero itself (no Python needed): in the cache's own layout, so `hf`, transformers, mlx-lm and Espetero share one copy, verified against the Hub's checksums and resumed after an interruption (`HF_TOKEN` for gated models; `--offline` or `HF_HUB_OFFLINE=1` never downloads). For a GGUF repository, name the quantization after a colon: `--model unsloth/Qwen3.8-27B-GGUF:UD-Q6_K` fetches that file, the model's MTP head and its vision projector.

Before downloading or loading a model, Espetero checks that it fits this Mac: one that cannot (its weights, the conversation state and the scratch against the GPU memory macOS gives one process) is refused with what it needs, a `--context` that fits and smaller files of the same model; `--skip-memory-check` tries anyway.

### Which model for my Mac

`espetero recommend` lists the models that fit your Mac, best first, with the `--model` to pass and the ones you have downloaded already (`espetero serve` without `--model` prints the same). With macOS's default GPU memory limit:

| Mac memory | Best model | Also good |
|:--|:--|:--|
| 8 GB | `mlx-community/Qwen3.5-2B-8bit` | `unsloth/Qwen3.5-0.8B-MTP-GGUF:Q8_0` |
| 16 GB | `mlx-community/Qwen3.5-9B-4bit` | `unsloth/Qwen3.5-4B-MTP-GGUF:Q8_0` |
| 24 GB | `unsloth/Qwen3.5-9B-MTP-GGUF:Q8_0` | |
| 32 GB | `mlx-community/Qwen3.8-27B-4bit --context 16384` | `unsloth/Qwen3.5-9B-MTP-GGUF:Q8_0` |
| 36 GB | `mlx-community/Qwen3.8-27B-4bit` | `mlx-community/Qwen3.6-35B-A3B-4bit` (MoE, several times faster) |
| 48 GB | `mlx-community/Qwen3.8-27B-8bit` | `mlx-community/Qwen3.6-35B-A3B-6bit` |
| 64 GB and more | `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp` (125B-A6B; 106 GB on the SSD, the experts that do not fit memory read from it) | `unsloth/Qwen3.8-27B-GGUF:Q8_0`, `mlx-community/Qwen3.6-35B-A3B-8bit` |

8-bit files come first, 6-bit next and 4-bit only when memory forces it; at one width a GGUF file with an MTP head (faster decoding, the same replies) comes before an MLX file without one. The table is for Espetero's defaults (a KV cache for 2 x 32K tokens at least); `--context` changes it, and `--ram-gb N` shows another Mac.

### Models on this Mac

```bash
espetero pull unsloth/Qwen3.8-27B-GGUF:UD-Q6_K   # download only (the newest revision); warns when it cannot fit this Mac
espetero list                                    # supported models in the Hugging Face cache and in ~/models/<family>/<name>
espetero rm unsloth/Qwen3.8-27B-GGUF:UD-Q6_K     # remove it from the Hugging Face cache (asks first; --yes does not)
```

They work on the shared Hugging Face cache: `list` shows what `hf download` fetched too, with each model's format, quantization, size, the memory it needs and whether it fits; `rm` deletes as `hf cache delete` does (the repository's files, or one GGUF quantization's, and the blobs nothing else links to) and never touches anything outside that cache.

Any OpenAI, Anthropic or Ollama client works against `http://127.0.0.1:8080`: the OpenAI SDKs (`base_url="http://127.0.0.1:8080/v1"`), Claude Code (`ANTHROPIC_BASE_URL=http://127.0.0.1:8080`), Codex CLI and opencode ([coding agents](docs/configuration.md#coding-agents)), Ollama's CLI, libraries and apps (`OLLAMA_HOST=http://127.0.0.1:8080`), curl:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

The port opens at once; `GET /health` answers 503 `{"status": "loading"}` until the model is ready, then 200. Open `http://127.0.0.1:8080` in a browser for the built-in chat: streamed replies, images pasted or attached, thinking on or off, a system prompt (`--no-webui` turns it off).

`espetero serve --model PATH` picks the best settings for that model on that Mac, and requests that leave out `temperature`, `top_p`, `top_k` or `min_p` get the values the model ships in its `generation_config.json`. `--preset vision`, `--preset classify` and `--preset flow` tune it for photo-first services, classification and many photos under the same instructions; `--help-all` lists every option ([docs/configuration.md](docs/configuration.md)).

## Install

```bash
make && sudo make install      # /usr/local/bin/espetero (PREFIX=... and DESTDIR=... change where)
sudo make uninstall
make dist                      # build/espetero-0.1.2-macos-arm64.tar.gz: the binary, README, licenses and notices
```

Homebrew: `brew install JarJarFran/espetero/espetero` builds Espetero from the release's source ([packaging/homebrew/espetero.rb](packaging/homebrew/espetero.rb)).

In the background, at every login:

```bash
espetero service install --model mlx-community/Qwen3.8-27B-4bit   # saved in ~/.config/espetero/config.json
espetero service status                                           # running? GET /health ready?
espetero service logs --follow                                    # ~/Library/Logs/espetero.log
espetero service uninstall
```

`espetero serve` reads `~/.config/espetero/config.json` too when it exists (any option as `{"option": value}`; the command line wins), so `brew services start espetero` serves the model named there.

## What it runs

- **Models**: the Qwen3.5 family (Qwen3.5, Qwen3.6, Qwen3.8 and their finetunes), dense and mixture-of-experts, with their vision towers and MTP heads, from MLX safetensors (4, 5, 6 and 8 bits, mixed per matrix) or GGUF (K-quants, Q4_0 to Q8_0, IQ4_NL/XS, F16/BF16), local or by Hugging Face repo id. Gemma 4 (31B dense and 26B-A4B mixture-of-experts, and the unified 12B with its encoder-free image embedder) from the same MLX and GGUF formats (the GGUF's vision from its `mmproj`), with its vision tower, thinking channel and tool calls; decision heads over the last hidden state (`"head"` in a chat request), on Gemma 4 and on the Qwen3.5 family, also for decision models trained without the causal mask in their full-attention layers (`perplexity-ai/pplx-decider-v1.1-27b`). Qwen3.8-Flash-Next (125B-A6B) runs on a 64 GB Mac with its experts streamed from the SSD, and keeps every expert resident on a 128 GB Mac. DeepSeek-V4.1-Flash (552B, 16B active; experimental) runs on a 128 GB Mac with its routed experts and Engram tables read from the SSD through an automatic expert cache ([docs/deepseek_v41.md](docs/deepseek_v41.md)). Qwen3-Omni's thinker (`Cloudflare/clef-omni`, images and text) answers the decision API. LiquidAI's LFM2.5-VL (with images) from its Hugging Face release or an MLX conversion. OpenBMB's MiniCPM-OCR 1.0 (document OCR: text, tables, formulas) from its Hugging Face release. Cloudflare's Clef classifiers, Kev (`jaredpalmer/kev-27b`, a pointer head on Qwen3.8-27B) and LiquidAI's d1 decision models run through the decision API (`/v1/systemone`). Other models (Gemma 3, Gemma 4 E2B/E4B, Llama, Qwen3, Qwen3-VL, ...) are refused at start with a message naming their type.
- **APIs**: OpenAI chat completions and completions (streaming, tools and forced tool calls, structured output with `response_format` `json_object` and `json_schema`, images, also by URL with `--image-urls`, logprobs of greedy and sampled replies, `n` choices from one prefill), embeddings of the loaded model, Responses (also `text.format`), Files and Batches; Anthropic Messages (also `output_config.format`); Ollama's API (`/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/show`, ...; `format` too); `/v1/decisions`, which scores every field of a structured decision from the logits in one pass instead of generating it; tokenize and detokenize; Prometheus metrics.
- **Engine**: one GPU pass per step for all the work in flight; paged KV cache with shared prefixes; checkpoints of system prompts, instructions and images, also kept on disk; speculative decoding with the model's MTP head (verified, so the reply is the model's own) and prompt lookup; seeded sampling that is reproducible; memory budgets sized from the Mac.
- **Structured output**: a JSON Schema (or "any JSON object") is compiled into a grammar that masks every token the reply could not continue with, so the reply parses and matches the schema; the model thinks freely first when thinking is on, and drafts keep working (they are verified against the masked choice). What is enforced and what is refused: [docs/configuration.md](docs/configuration.md#structured-output).
- **Lossless by default**: every automatic setting either leaves the output unchanged or was validated on 6- and 8-bit files. Faster approximations exist only as explicit options.

## Requirements

macOS 15 or later on Apple silicon, and the Xcode Command Line Tools (`xcode-select --install`). Tuned and tested on a MacBook Pro M1 Max with 64 GB and a Mac Studio M4 Max with 128 GB; the GEMM tiles are autotuned for each Mac at the first load.

## Tests

```bash
make test        # JSON, sampling, chat templates, HTTP, image limits, model checks, Hugging Face ids, downloads and
                 # the memory check (a fake Hub on 127.0.0.1), API request parsing, Ollama request translation, the
                 # web chat page, the LaunchAgent (in a temporary home), image URL fetches and structured output
                 # (schemas, grammar, token masks on the cached Qwen3.5-0.8B tokenizer when there is one), Gemma 4's
                 # configurations and tool calls: no model or GPU needed
build/hub_test --live   # two tiny repositories (~30 MB) from huggingface.co, checked with huggingface_hub's own reader
```

The model-level tests (parity against reference runtimes, vision fidelity against mlx-vlm, the decision API) need model files; see `tests/` and `tools/`.

## Security

Espetero listens on `127.0.0.1` by default. This release is meant for one trusted user: every API key holder shares the prompt cache and the Batch API. Request, image and upload sizes are bounded, images are fetched by URL only with `--image-urls` and never from this machine or its networks unless `--image-urls-private` allows it, and without an API key the server refuses requests from web pages (DNS rebinding, cross-site requests); see [limits for untrusted clients](docs/configuration.md#limits-for-untrusted-clients). Keep it on localhost, or behind an authenticating proxy, and set `--api-key` (or `ESPETERO_API_KEY`).

## License

MIT ([LICENSE](LICENSE)), except `src/engine/clef.h` and `src/engine/clef.cpp` (Apache License 2.0). Some files contain code derived from libjpeg-turbo, Pillow, Jinja2, ggml, JEVfire, Cloudflare Clef, Hugging Face transformers, mlx-lm and LiquidAI's d1 release; their notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This software is based in part on the work of the Independent JPEG Group.

Espetero grew out of [JEVfire](https://github.com/kikoncuo/jevfire)'s Metal engine. Copyright (c) 2026 Francisco José García Barbero.
