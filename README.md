# Espetero

**Fast LLM inference for Apple silicon.** A native Metal engine (C++20 and Metal, no Python) that puts every request in one GPU pass: chats, images and classifications grilled together, the way an *espetero* grills a whole cane of sardines over the embers.

*Espetero* (es-peh-TEH-ro) is the cook on the beaches of Málaga who grills sardines threaded on canes, the *espetos*, over a fire in a boat full of sand.

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

† Another app used the GPU during this run (5 to 33 % of the run's time on the GPU), which can slow the engine down.
¹ The mlx-community file has no MTP head; `--mtp` borrows the one unsloth ships for the GGUF.
² TensorFold's native engine has no backend for the Qwen3.6 mixture-of-experts or Gemma 4 checkpoints yet, and it serves images only with `--vision`, which these runs did not pass.
³ Espetero runs Gemma 4 text only so far.

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

Any OpenAI, Anthropic or Ollama client works against `http://127.0.0.1:8080`: the OpenAI SDKs (`base_url="http://127.0.0.1:8080/v1"`), Claude Code (`ANTHROPIC_BASE_URL=http://127.0.0.1:8080`), Ollama's CLI, libraries and apps (`OLLAMA_HOST=http://127.0.0.1:8080`), curl:

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
make dist                      # build/espetero-0.1.0-macos-arm64.tar.gz: the binary, README, licenses and notices
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

- **Models**: the Qwen3.5 family (Qwen3.5, Qwen3.6, Qwen3.8 and their finetunes), dense and mixture-of-experts, with their vision towers and MTP heads, from MLX safetensors (4, 5, 6 and 8 bits, mixed per matrix) or GGUF (K-quants, Q4_0 to Q8_0, IQ4_NL/XS, F16/BF16), local or by Hugging Face repo id. Gemma 4 (31B dense and 26B-A4B mixture-of-experts; text only: images get a 422) from the same MLX and GGUF formats, with its thinking channel and tool calls. Qwen3.8-Flash-Next (125B-A6B) runs on a 64 GB Mac with its experts streamed from the SSD. Cloudflare's Clef classifiers run through the decision API. Other models (Gemma 3, Gemma 4 E2B/E4B, Llama, Qwen3, Qwen3-VL, ...) are refused at start with a message naming their type.
- **APIs**: OpenAI chat completions and completions (streaming, tools and forced tool calls, structured output with `response_format` `json_object` and `json_schema`, images, also by URL with `--image-urls`, logprobs of greedy and sampled replies, `n` choices from one prefill), embeddings of the loaded model, Responses (also `text.format`), Files and Batches; Anthropic Messages (also `output_config.format`); Ollama's API (`/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/show`, ...; `format` too); `/v1/decisions`, which scores every field of a structured decision from the logits in one pass instead of generating it; tokenize and detokenize; Prometheus metrics.
- **Engine**: one GPU pass per step for all the work in flight; paged KV cache with shared prefixes; checkpoints of system prompts, instructions and images, also kept on disk; speculative decoding with the model's MTP head (verified, so the reply is the model's own) and prompt lookup; seeded sampling that is reproducible; memory budgets sized from the Mac.
- **Structured output**: a JSON Schema (or "any JSON object") is compiled into a grammar that masks every token the reply could not continue with, so the reply parses and matches the schema; the model thinks freely first when thinking is on, and drafts keep working (they are verified against the masked choice). What is enforced and what is refused: [docs/configuration.md](docs/configuration.md#structured-output).
- **Lossless by default**: every automatic setting either leaves the output unchanged or was validated on 6- and 8-bit files. Faster approximations exist only as explicit options.

## Requirements

macOS 15 or later on Apple silicon, and the Xcode Command Line Tools (`xcode-select --install`). Tuned and tested on an M1 Max with 64 GB; the GEMM tiles are autotuned for each Mac at the first load.

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

Espetero listens on `127.0.0.1` by default. Version 0.1.0 is meant for one trusted user: every API key holder shares the prompt cache and the Batch API. Request, image and upload sizes are bounded, images are fetched by URL only with `--image-urls` and never from this machine or its networks unless `--image-urls-private` allows it, and without an API key the server refuses requests from web pages (DNS rebinding, cross-site requests); see [limits for untrusted clients](docs/configuration.md#limits-for-untrusted-clients). Keep it on localhost, or behind an authenticating proxy, and set `--api-key` (or `ESPETERO_API_KEY`).

## License

MIT ([LICENSE](LICENSE)), except `src/engine/clef.h` and `src/engine/clef.cpp` (Apache License 2.0). Some files contain code derived from libjpeg-turbo, Pillow, Jinja2, ggml, JEVfire, Cloudflare Clef, Hugging Face transformers and mlx-lm; their notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This software is based in part on the work of the Independent JPEG Group.

Espetero grew out of [JEVfire](https://github.com/kikoncuo/jevfire)'s Metal engine. Copyright (c) 2026 Francisco José García Barbero.
