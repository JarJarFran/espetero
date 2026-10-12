# Configuring Espetero

Three levels, one engine. Most users never pass more than `--model`.

## 1. Automatic (no experience needed)

```
espetero serve --model mlx-community/Qwen3.8-27B-4bit   # a Hugging Face repo id
espetero serve --model /path/to/model                   # or an MLX directory or a .gguf file
```

The engine picks the best settings for the model file and this Mac. Every
automatic default below was measured on 6- and 8-bit MLX and Q6_K/Q8_0 GGUF
files and changes quality by nothing, except the sampling defaults,
which are the model's own recommendation:

| What | Automatic choice | Why |
|:--|:--|:--|
| Speculative decoding | MTP drafts, depth 5, while drafting beats plain decoding (adaptive: the request's per-depth acceptance against verify / draft / plain costs timed at load; a losing head pauses for 8, 16, 32, 64, 128 steps and probes again); accepted drafts become rows in place, so a pass verifies the new token and its drafts only (Flash-Next included); sampled requests with `top_k` 1–32 draw their drafts as the target draws its tokens; penalties count the drafts accepted earlier in a pass | 27B Q6_K: ~2.6 tokens per verify pass; drafts are verified, so replies are the model's own |
| Draft head | coarse 2-bit copy of the LM head, 256 blocks rescored with the real head | draft passes 13 % cheaper, drafts equal to the exact head's on 47 of 48 test steps, MoE GPQA decode +3.7 % |
| Images | ~1 MP budget per image, EXIF orientation applied, image checkpoints shared across questions | a good balance of detail and time (~1,000 image tokens, ~10 s per photo on the 27B). At the same budget Espetero's pixels are identical to mlx-vlm's processor and its embeddings 6–7× closer to fp32 than mlx-vlm's bf16 tower. mlx-vlm's own default is the model's 16.8 MP ceiling, which costs minutes per phone photo on an M1 Max: it is `--image-detail max` here |
| Memory | slots and KV sized from free GPU memory | |
| Model choice | `espetero recommend` (and `espetero serve` without `--model`) lists the models of a curated table that fit this Mac, best first, with the `--model` to pass and the downloaded ones marked: 8-bit files first, 6-bit next, 4-bit only when memory forces it, a smaller `--context` (16384, then 8192) only when nothing else fits | see below |
| Memory check | a model that cannot fit this Mac's GPU memory is refused before it is downloaded and before it is loaded, with what it needs, a `--context` that fits and smaller files of the same model (`--skip-memory-check` to try anyway) | a 16 GB Mac does not download 30 GB of a 27B 8-bit to fail at load; see below |
| Sampling defaults | requests that leave out `temperature`, `top_p`, `top_k` or `min_p` get the model's `generation_config.json` values (GGUF: `general.sampling.*`; Qwen3.8-27B: 1.0, 0.95, 20), the rest OpenAI's; set values always win, greedy requests unchanged; `--no-generation-config` for OpenAI's (1, 1, off, 0) | the settings the model's authors recommend, applied as vLLM and llama.cpp apply them; `top_k` 1–32 also puts sampled requests on the GPU sampling path. Details below |
| Model by repo id | `--model ORG/NAME` or `ORG/NAME:QUANT` (GGUF) runs the Hugging Face cache's snapshot, downloaded into that cache by Espetero itself when missing, resumable and verified (`--offline`: never) | no separate download step, no Python; one copy shared with `hf`, transformers and mlx-lm; see below |
| Prompt caching | system-prompt, instruction and image checkpoints; a shared prefix is computed once even when many requests arrive together | repeated system prompts, instructions placed before a photo (shared by a set of photos) and several questions about one photo cost one prefill; with `--prefix-dir` (on in `--preset flow`) a restarted server loads them from disk instead of computing them again (a shared prompt of tens of thousands of tokens loads in a fraction of a second instead of minutes on Flash-Next); checkpoints that running requests restored from or saved are evicted last, so the next question about the same image still finds it (`--no-keep-live` for plain LRU) |
| Models larger than memory (Qwen3.8-Flash-Next) | the n-gram table read from the SSD by rows; the most recently used experts kept in GPU memory that macOS never compresses or swaps (automatic: what the GPU working set and the memory available before loading leave after the model, the slots + KV and 2.5 GB of scratch: 44.6 GB on a 64 GB Mac, every expert on 128 GB; `--expert-cache-gb X` to set it, `-1` for none; `--memory-gb` for slots + KV, 3 GB by default for this model), the rest read from the SSD when a token routes to them; the next layer's likely experts read ahead; expert records (one read per missing expert instead of nine) used when built: `--expert-store-build` writes them once (~68 GB, ~40 s) to ~/Library/Caches/espetero/xstore.noindex/&lt;model&gt; (outside Time Machine and Spotlight), `--expert-store DIR\|none` to choose; MTP drafts (3 per pass) from a Qwen3.8-Flash-Next-MTP-Drafter directory found beside the model (8-bit at load, ~2.7 GB; `--mtp PATH\|none`) | answers identical to a model held in memory (drafts are verified); reads do not wear the SSD; on a 64 GB M1 Max it decodes ~31 tok/s served and describes a photo in ~7.7 s |

### Models and request defaults

`--model` takes an MLX directory, a `.gguf` file or a Hugging Face repo id:

| `--model` | What runs |
|:--|:--|
| `mlx-community/Qwen3.8-27B-4bit` | the newest complete snapshot of the repository in the Hugging Face cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`; the revision `refs/main` names first): `config.json`, `tokenizer.json` and every weight file |
| `unsloth/Qwen3.8-27B-GGUF:UD-Q6_K` | the GGUF file whose name ends with the quantization after the colon, in any letter case and after a `-`, `_` or `.` (`UD-Q6_K` is `Qwen3.8-27B-UD-Q6_K.gguf`, not `Qwen3.8-27B-UD-Q6_K_XL.gguf`; the shortest name wins a tie), with the repository's MTP head (`MTP/mtp-*.gguf`) and one vision projector (`mmproj-*.gguf`, F16 first) |
| a GGUF repository without `:QUANT` | its only GGUF model (or the only one downloaded); with several, the error lists the `--model ORG/NAME:QUANT` choices |

A repo id missing from the cache is downloaded there first, by Espetero
itself (no Python or `hf` CLI needed), with a progress line on the
terminal: of an MLX repository its configuration, tokenizer and weight files
(`*.json`, `*.safetensors`, `*.jinja`, `*.txt`, `tokenizer.model`, as mlx-lm
fetches them; `config.json` first, so a model of another family is refused
before its weights), of a GGUF repository only the files above. The files go
into the Hugging Face cache exactly as `huggingface_hub` lays it out, so `hf`,
transformers, mlx-lm and Espetero share one copy of each model:
`models--ORG--NAME/blobs/` holds the contents, named by their sha256 (LFS
files) or git blob id, `snapshots/<commit>/` links to them and `refs/main`
names the commit. Each file is verified against the Hub's checksum, an
interrupted download resumes where it stopped (`blobs/<id>.incomplete`, an
HTTP range), the same lock files as `huggingface_hub` keep two processes from
writing the same file, and the disk's free space is checked first. Up to four
files download at once. Gated and private repositories need a token:
`HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`, or the token file `hf auth login`
writes: `$HF_TOKEN_PATH`, else `$HF_HOME/token`, else
`~/.cache/huggingface/token`); it is sent to the Hub only, never to the CDN
its downloads redirect to, and never printed. `HF_ENDPOINT` names another
Hub (a mirror). `--offline` (or `HF_HUB_OFFLINE=1`) never downloads. The API
reports the repo id as the model name (`--name` changes it). GGUF models split
in parts (`-00001-of-00002.gguf`) are not read yet: merge them with
llama.cpp's `llama-gguf-split --merge`.

Before a download and before loading, Espetero checks that the model fits
this Mac. It adds up what the engine will hold in GPU memory: the
weights as the engine stores them (GGUF K-quants other than Q6_K widen a
little, IQ4 to 8-bit codes; the vision tower and MTP head unless turned off;
Qwen3.8-Flash-Next's experts and n-gram table stay on the SSD), the
conversation state at its floor (KV for two full contexts of `--context` and
17 sequence states; `--memory-gb`, `--kv-gb` and `--slots` when given) and
the pass scratch, and compares the sum with the GPU memory macOS gives one
process (Metal's recommended working set: 2/3 of the RAM up to 32 GB, 3/4
above, or what `iogpu.wired_limit_mb` sets). For a repo id the sizes come
from the Hub's file list and the headers of the files the engine would read
(a few MB of byte ranges; an MLX repository's `config.json` first), so its weights are not downloaded when it cannot fit:

```
$ espetero serve --model unsloth/Qwen3.8-27B-GGUF:Q8_0        # on a Mac with 48 GB
error: unsloth/Qwen3.8-27B-GGUF:Q8_0 needs about 39.3 GB of GPU memory, more than the 38.7 GB this Mac's GPU can use (48 GB of RAM):
   30.4 GB  weights (Qwen3.8-27B-Q8_0.gguf with its MTP head, projector mmproj-F16.gguf)
    7.2 GB  conversation state: KV cache for 2 x 32768 tokens (--context) and 17 sequence states
    1.7 GB  scratch for 4096-token passes
It fits with --context 16384 (about 37.1 GB).
Smaller files of this model that fit:
  --model unsloth/Qwen3.8-27B-GGUF:UD-Q8_K_L   29.0 GB of files, needs about 38.3 GB
  --model unsloth/Qwen3.8-27B-GGUF:UD-Q6_K_XL   26.2 GB of files, needs about 35.5 GB
  ...
Its weights were not downloaded. `espetero recommend` lists the models that fit this Mac; --skip-memory-check downloads and runs it anyway.
```

The suggestions are the repository's other GGUF quantizations, MLX
repositories of the same name with other bit widths (`-8bit`, `-6bit`,
`-4bit`, `-oQ6e-mtp`, ...), and for a local file the other quantizations
beside it. A model whose geometry the engine does not run (for example the
GQA group 16 of Qwen3.5-122B-A10B) is refused before its download too.
`--skip-memory-check` downloads and loads it anyway; `ESPETERO_MAC_RAM_GB=16`
answers the check as a Mac with 16 GB would, to see what another Mac can run.

`espetero recommend` answers which model to run on this Mac from a
table of public repositories of the supported families with their sizes on
the Hub: Qwen3.8-Flash-Next (oQ4e build; recommended where at least half of
its 69 GB of experts fit in memory), Qwen3.8-27B, Qwen3.6-35B-A3B and the
Qwen3.5 9B, 4B, 2B and 0.8B, as MLX (mlx-community, lmstudio-community) and
GGUF (unsloth) files. For each model it prints the best file that fits:
8-bit, then 6-bit, then 4-bit (an MTP GGUF before an MLX file without a
head at the same width, except for the MoE, whose MLX files are the
measured ones), at the plan's `--context` or 16384, and 8192 only as a last
resort. Files already in the Hugging Face cache are marked, and other
downloaded files of the same model that fit are named. `--all` lists every
file with what it needs; `--ram-gb N` answers for a Mac with N GB of RAM
and macOS's default GPU limit; `--context`, `--slots`, `--memory-gb` and the
presets change the plan as they change a server. The same table's files of
the model are among the suggestions of a refusal.

The models on this Mac are managed with three commands, all on the
same Hugging Face cache:

| Command | What it does |
|:--|:--|
| `espetero pull ORG/NAME[:QUANT]...` | downloads the repository's newest revision as `--model` would (files already there are linked, not fetched again; `... is up to date` when nothing changed); a model that cannot fit this Mac is downloaded with the memory warning (`--skip-memory-check` skips it), one of another family or geometry is refused |
| `espetero list` | the supported models in the Hugging Face cache (the newest complete revision of each repository; each GGUF model of a repository as `ORG/NAME:QUANT`) and under `~/models/<family>/<name>` (`ESPETERO_MODELS_DIR` to change it): the `--model` to pass, format, quantization, size on disk (each blob once, with the projector and MTP head a GGUF model uses), the GPU memory it needs at `--context`, whether it fits this Mac (`16K ctx`: with `--context 16384`), dense or MoE, MTP head, vision; MTP heads alone, non-MLX safetensors (the original transformers layout) and other families are counted, not listed |
| `espetero rm ORG/NAME[:QUANT]...` | removes from the cache what `hf cache delete` would: `ORG/NAME` the whole repository (every revision, its blobs, refs and lock files); `ORG/NAME:QUANT` that GGUF model's file in every revision (a revision left without a model loses its projector and MTP head too), the blobs no revision links to any more and the refs of revisions that are gone. It lists the files and the space they free and asks before deleting (`--yes`: without asking; without a terminal it refuses). Paths and anything outside `models--ORG--NAME` are never deleted, and links are never followed |

Only models Espetero runs are loaded: an MLX `config.json` whose `model_type`
(or `text_config.model_type`) is of the Qwen3.5 family (`qwen3_5`,
`qwen3_5_moe`, `qwen3_5_mtp`, and `_text` variants), Qwen3.8-Flash-Next
(`qwen4_exp`) or Gemma 4 (`gemma4`, `gemma4_text`), a GGUF whose
`general.architecture` is `qwen35`, `qwen35moe` or `gemma4`. Any other model
stops `serve`, `chat`, `batch` and `info` at start with one message naming its
type:

```
error: mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit: model_type 'llama4' is not supported by Espetero 0.1.0; it runs the Qwen3.5 family (qwen3_5, qwen3_5_moe: Qwen3.5, Qwen3.6, Qwen3.8, and finetunes such as Ornith 1.5), Qwen3.8-Flash-Next (qwen4_exp), Gemma 4 (gemma4: 31B, 26B-A4B; text only) and Cloudflare Clef classifiers
```

Gemma 4 runs its 31B dense and 26B-A4B mixture-of-experts models, text
only: a request with an image gets a 422 (`images are not supported for Gemma
4 yet`), and so does `tool_choice` `required` or a named function (its calls
are not in the `<tool_call><function=...>` format the forced call writes;
`auto` and `none` work, and the model's `<|tool_call>call:NAME{...}` calls
come back as OpenAI `tool_calls`). Its per-layer-embedding models (E2B, E4B)
are refused at load. Thinking follows the template: `enable_thinking: true`
(or Anthropic's `thinking`) opens the thought channel, whose text comes back
as `reasoning_content`; with `response_format` the grammar applies after the
channel closes. Its sliding-window layers keep their last 1024 positions in a
ring per sequence slot (about 0.84 GB per slot for the 31B, 0.21 GB for the
26B-A4B), and only the global layers' K/V grow with the context (80 KB per
token for the 31B), so the automatic plan gives fewer slots and more context
than a Qwen model of the same size. There is no MTP head: drafts come from
prompt lookup only. Chat templates see `bos_token` as transformers passes it
(the template starts the prompt with it); raw text (`/v1/completions` and
embeddings) gets `<bos>` in front as llama.cpp does it: the GGUF file asks for
it (`tokenizer.ggml.add_bos_token`) and SentencePiece vocabularies get it when
`tokenizer_config.json` does not say (Gemma 4 without it repeats one word).
Qwen's files never ask for it.

Requests that leave out `temperature`, `top_p`, `top_k` or `min_p` get the
model's own values, from its `generation_config.json` (GGUF:
`general.sampling.temp`, `top_p`, `top_k`, `min_p`), as vLLM and llama.cpp
do; Qwen3.8-27B ships temperature 1.0, top_p 0.95 and top_k 20. Values a
request sets always win, requests with `temperature` 0 stay greedy and
unchanged, and fields the model does not set keep OpenAI's defaults
(temperature 1, top_p 1, top_k off, min_p 0); `do_sample` is ignored, as in
vLLM. A `top_k` of 1 to 32 also puts sampled requests on the GPU sampling
path (drafts drawn as the target draws its tokens).
`--no-generation-config` gives every request OpenAI's defaults again.
`--print-config` and `GET /v1/config` show the defaults in effect and where
they come from (`effective.sampling-defaults`). `espetero chat` uses
`--temperature`, `--top-p` and `--top-k` when given, else the model's values,
else 0.7, 0.95 and 20.

## 2. Presets and common options (some knowledge)

All presets serve the same OpenAI-compatible API (`/v1/chat/completions`,
`/v1/completions`, `/v1/embeddings`, images in messages, `/v1/batches`) plus
`/v1/decisions`.
Other clients' APIs run the same chat job (same prompt, same tokens):
OpenAI's `/v1/responses` (`previous_response_id` continues one of the newest
256 stored responses; `GET` / `DELETE /v1/responses/{id}`), Anthropic's
`/v1/messages` and `/v1/messages/count_tokens` (Claude Code and Anthropic SDK
clients: `ANTHROPIC_BASE_URL=http://host:port`), Ollama's `/api/*` routes
([below](#ollama-clients)), and Prometheus metrics at `GET /metrics`
(the `/v1/stats` counters). `--api-key` is accepted as
`Authorization: Bearer KEY` or as Anthropic's `x-api-key: KEY`. Thinking
follows each API: Anthropic requests think only with `"thinking": {"type":
"enabled"}`, Responses requests turn it off with `reasoning.effort` `none` or
`minimal`, and their effort reaches templates that take one (Flash-Next:
`low`, `medium`, `high` = `xhigh`). Not supported (an error, never
ignored): a non-empty `logit_bias` (422), built-in or server tools, images
by file id, images inside tool results. `tool_choice` `none` renders the prompt without the
tools, so no call is parsed. An Anthropic reply that one of its
`stop_sequences` ended has `stop_reason` `stop_sequence` and that string in
`stop_sequence`.

Embeddings: `/v1/embeddings` (OpenAI's format: `input` a string, a
list of strings, a list of token ids or a list of such lists, at most 2,048
inputs and 8 context windows of tokens per request; `encoding_format`
`float` or `base64`; `dimensions` only equal to the model's width, else 422)
returns, per input, the loaded model's final-normed hidden state (the LM
head's input; no LM head runs), pooled over the input's tokens and
L2-normalized: the last token's by default, as Qwen3-Embedding pools,
`--embedding-pooling mean` for the mean of every token's. Text is tokenized
as `/v1/completions` does (no special tokens added). These are the chat
model's own representations: a dedicated embedding model such as
Qwen3-Embedding is another architecture, not supported yet; and
Qwen3.8-Flash-Next cannot read out hidden states yet (422). Inputs share a
pass while they fit (`--max-rows`, 4,096 rows; while sequences decode,
`--prefill-chunk` rows, so they wait at most one such pass per step); a
longer input is computed over several passes with its state in a slot. An
input's vector does not depend on what shares its pass or where a pass cuts
it: embedding passes run their matrix products on one tile shape whatever
their row count (other passes use the tile tuned for theirs), and a pass of
at most 8 rows gets a padding tree to stay off the few-row kernels, so an
input alone, in a batch or cut into passes gives the same bits.
`usage.prompt_tokens` counts every input's tokens; `timings` has the passes,
rows and GPU time. The Batch API takes `/v1/embeddings` lines too.

Forced tool calls: `tool_choice` `required` or a named function
(`{"type": "function", "function": {"name": ...}}`; Responses: `{"type":
"function", "name": ...}`; Anthropic: `{"type": "any"}` or `{"type": "tool",
"name": ...}`) make the reply a call. The reply is started with the opening
of a call in the model's own format, as its chat template renders an
assistant turn with calls: `<tool_call>\n<function=NAME>\n` for a named
function (or when only one tool is offered), `<tool_call>\n<function` for
`required`, whose continuation names one of the tools (the name is the
model's choice, not constrained: the prompt lists the tools). When the
prompt opens the reasoning block (thinking on), the model thinks first: the
reply stops at `</think>`, then continues the same prompt with the
reasoning, `</think>` and the call's opening (a second, short prefill; usage
counts the reasoning, the opening and the call as completion tokens, and
`timings.tool_call_ms` is the call's part). `allowed_tools` (`mode` `auto` or
`required`) renders only the listed tools. Templates whose calls are not
Qwen's `<tool_call><function=...>` XML (the format the server parses) get a
422 for a forced call, as does a function that is not in `tools`.
With thinking off, the opening is part of the prompt (`count_tokens` and
`usage.prompt_tokens` count it).

### Structured output

`response_format` on `/v1/chat/completions` and `/v1/completions`
(`{"type": "json_object"}`, or `{"type": "json_schema", "json_schema":
{"name": ..., "schema": {...}, "strict": true}}`), `text.format` on
`/v1/responses` (`{"type": "json_schema", "name": ..., "schema": {...},
"strict": true}` or `{"type": "json_object"}`; the response echoes it),
`output_config.format` on `/v1/messages` (Anthropic's `{"type":
"json_schema", "schema": {...}}`, strict; also the older `output_format`)
and the same bodies in Batch API lines constrain the reply with a grammar: at
every step the tokens that cannot continue a valid reply are masked before
sampling, so `json_object` gives one JSON object and `json_schema` JSON
valid under the schema. Greedy requests get the most likely allowed token
of the full logits; sampled ones draw from the distribution with the
disallowed tokens at -inf (top-k, top-p and min-p among the allowed
tokens). The schema is not added to the prompt: say in the messages what
the answer should hold (a model fills fields it was told about better).
Ollama's `format` (`"json"` or a schema) maps to these, non-strict.

Enforced: `type` (one or a list; OpenAPI's `nullable: true` adds null),
`enum` and `const` (any JSON values, kept when the rest of the schema
accepts them), `properties`, `required`, `additionalProperties` (false,
true or a schema), `minProperties`, `maxProperties`, `items` (a schema or
false; a draft-7 list with `additionalItems`), `prefixItems`, `minItems`,
`maxItems`, `minLength` and `maxLength` (code points), `minimum`, `maximum`,
`exclusiveMinimum` and `exclusiveMaximum` (numbers and draft-4 booleans) on
`integer` and `number`, `anyOf` and `oneOf` (as alternatives: oneOf's
"exactly one" is not checked, alternatives that overlap can give a value
two of them accept), `allOf` (merged), `$ref` to the schema itself (`#`,
`#/$defs/...`, `#/definitions/...`, any JSON pointer; recursive schemas
too), and `unevaluatedProperties` / `unevaluatedItems` false. Annotations
(`title`, `description`, `default`, `examples`, ...) and unknown keywords
are ignored, as validators ignore them.

Not enforced: `pattern`, `format`, `multipleOf`, `uniqueItems: true`,
`patternProperties`, `propertyNames`, `dependentRequired`,
`dependentSchemas`, `dependencies`, `if` / `then` / `else`, `not`,
`contains` (`minContains`, `maxContains`), `unevaluated*` schemas,
`$dynamicRef`, `$recursiveRef`, tuples in two branches of one `allOf`, and
a recursive `$ref` merged with other keywords. With `"strict": true` such a
schema gets a 422 naming the keyword and where it is (`"pattern" at
#/properties/zip`); otherwise (`strict` false or absent, OpenAI's default)
the reply follows the rest of the schema and the response lists what was
left out in `structured_output.unenforced` (streams: in the finish chunk).
Always a 422: a schema that admits no value (contradictory keywords, an
empty `enum`, a required property that is not allowed, a recursion that
never ends), a reference outside the schema or by anchor, an unknown type,
more than 256 alternatives once `allOf` / `anyOf` / `oneOf` combine, and a
format together with a forced tool call, `ignore_eos` or `score`.

How replies are written (each a valid instance): properties in the
schema's order (optional ones may be skipped), other keys after them, and
only where the schema allows them (an object schema that names properties
and has no `additionalProperties` gets no other keys; one that names none
takes any); between tokens one space, or a newline and up to 40 spaces or
tabs of indentation; at most two newlines before the value and nothing
after it; strings escape what Python's `json.dumps` escapes and nothing
else; numbers have at most 20 integer and 20 fraction digits and 3 exponent
digits, and with bounds no exponent and at most 15 significant digits (a
decimal and its float then compare alike); at most 47 nested values.

Thinking: when the chat template opens the reasoning block, the model
thinks freely (only the end of the turn is masked, so the JSON always
follows; a reasoning longer than `max_tokens` leaves none) and the grammar
applies after `</think>`: `reasoning_content` holds the reasoning,
`content` the JSON. Tools offered beside a format (`tool_choice` `auto`):
the reply may be a call instead (it starts with `<tool_call>` and is then
not constrained).

Speed: a constrained sequence picks its tokens on the CPU (the GPU argmax,
or the first allowed of the GPU's top 32 candidates, when they decide;
else a scan of the allowed tokens' logits), so it runs no decode chains
and no GPU draws; MTP and prompt-lookup drafts stay (cut before the first
token the grammar rejects, verified against the masked choice), so greedy
replies equal masked greedy decoding without drafts. Masks are cached per
parser state and shared by the requests with the same schema (32 schemas,
24 MB of masks each); a new state's mask takes 0.01 to 0.3 ms over a 248K
vocabulary, and inside free strings it is a precomputed token set. Decode
on an M1 Max, without and with a schema: Qwen3.5-0.8B 289 and 273 tok/s
greedy (286 and 276 sampled), Qwen3.6-35B-A3B 6-bit 71.9 and 71.6 (71.6 and
71.1), Qwen3.8-27B UD-Q6_K with its MTP head 21.2 and 20.5 (21.4 and 20.5).
Requests without a format are unchanged, token for token.
Non-streamed responses report `timings.structured` (masks computed and
found cached, their time); `ESPETERO_GRAMMAR_CHECK=1` (development) also checks
every masked argmax against a scan of the allowed logits
(`argmax_mismatches`).

| Preset | For | Settings it changes |
|:--|:--|:--|
| `--preset general` (default) | chat and completions with images | ~1 MP images, 32K context, 64 concurrent requests; a new request waits at most 200 ms for the running ones to decode before its prompt runs beside them (`--prefill-wait-ms`; the other presets wait up to 2.5 s, which batches photo prompts): on the 27B its first token comes 1.1 s after it arrives instead of 3.6 s while another reply streams |
| `--preset vision` | photo-first services | ~1 MP images, 16K context, 16 concurrent requests |
| `--preset classify` | multimodal classification | `/v1/decisions`: every field (enums, booleans) scored from the logits in one parallel pass instead of generated token by token (the JEV-style classifier); images allowed (up to 16 per request, `image_detail` per request); shared prefixes; 256-entry repeat memo; 16K context, 32 concurrent requests |
| `--preset flow` | many images under the same instructions, several requests per image (several questions per image) | 16 concurrent requests (with Flash-Next's sparse attention a long shared prompt does not make each extra request expensive) and 2 slots per request + 4 (36: each image in flight keeps its checkpoint while its requests run), prompts batched up to 4,096 rows per pass, drafts kept in batches of up to 16, no prompt-end checkpoints (flows never re-send a prompt), shared prefixes kept on disk across restarts (`--prefix-dir`, at most 20 GB, `--prefix-dir-gb`). With a very long shared prompt pass `--prefill-batch-rows 1024`: as fast, ~4 GB less GPU memory. Put the shared instructions BEFORE the image in the user message: they are computed once for every image; each image once for all its requests. |

Common options:

| Option | Values |
|:--|:--|
| `--context N` | tokens per conversation |
| `--users N` | concurrent requests |
| `--image-detail L` | `low` (512×512), `standard` (~1 MP), `high` (~4 MP), `max` (the model's own 16.8 MP ceiling, mlx-vlm's default; slow: a 12 MP photo is ~12K image tokens) |
| `--speculation M` | `auto` (default), `on` (draft every step), `off` |
| `--memory-gb X` | GPU memory for conversations |
| `--host`, `--port`, `--api-key`, `--name` | server (the key can also come from the environment variable `ESPETERO_API_KEY`, which keeps it out of `ps`) |

`serve` opens its port before it downloads or loads the model: a
port another program listens on, at any address of this Mac, stops it at
once (`error: port 8080 is in use (another program listens on it): pick
another with --port`), and until the model is ready every route answers 503
with `{"status": "loading"}` and `Retry-After: 5`, as llama-server does
(beside it an OpenAI-style `error` object; Ollama's routes get Ollama's
`{"error": "..."}`). A health check waits for `GET /health` to answer 200.

Per request, images take OpenAI's `detail` (`low`, `auto`, `high`, plus
`max`); `/v1/decisions` takes `image_detail` with the same levels. A request
may ask for the server's own level or less (`low` always): a level above it
(`high` or `max` on a `standard` server; `high` is 4× standard, at most `max`)
gets a 422 unless the operator starts the server with
`--allow-image-detail-max`, since one `max` image is up to 16,384 image tokens
and minutes of vision tower.

Chat requests with `"logprobs": true` (and `top_logprobs`, 0–20) get OpenAI's
`choice.logprobs.content`: per generated token its log-probability and the
most likely alternatives, plus the token `id`. They are read from the GPU's
top-k while the answer is written, so they cost nothing, greedy or
sampled, streamed or not. The values are the model's own distribution:
the log-softmax of the raw logits, at temperature 1 and before penalties,
`top_k`, `top_p` and `min_p` (vLLM's default "raw logprobs"), so a sampled
token outside the alternatives still has its exact value, and sampling draws
the same tokens with or without logprobs. A streamed chunk carries the
entries of the tokens whose text it carries (text held back for a stop
string, a tool call or a partial tag brings its entries later; the last
chunk carries the rest), so the chunks' entries concatenate to the
non-streamed list. `/v1/completions` takes OpenAI's legacy `logprobs: N`
(0–20): `tokens`, `token_logprobs`, `top_logprobs` (the N most likely plus the
sampled token) and `text_offset` (bytes); not with `echo` (the prompt's own
log-probabilities are not computed: 422). A flow that needs a label's
probability reads it from the answer itself instead of scoring the answer
again (for example, the probability of each classification label).
`/v1/responses` refuses `top_logprobs` and `include`
`message.output_text.logprobs` (422).

`n` (1–16) on `/v1/chat/completions` and `/v1/completions` (and in batch
files) returns n choices of one prompt, whose prefill runs once: one
sequence computes the prompt, its last readout gives every choice its first
token, and the other choices continue from copies of its state (the KV blocks
are shared). With a `seed`, choice 0 draws with that seed (it equals the
`n: 1` reply) and choice i with splitmix64(seed + i × 0x9E3779B97F4A7C15), so
a request is reproducible; greedy choices are identical, as OpenAI's. Streamed
chunks carry their choice's `index`; `usage` counts the prompt once and the
completion tokens of every choice, `timings` are the first choice's. A choice
that finds the server at `--max-seqs`, or no state slot or KV room for its
answer, waits as a request of its own (it computes the prompt again, or
continues from a cached checkpoint of it). `best_of` other than `n` is
refused (422).

A text part with `"cache_control": {"type": "ephemeral"}` (as Anthropic's API; `/v1/messages` passes it
through) or `"checkpoint": true` ends a prefix the server keeps for later requests. The first mark is the
instruction boundary (shared rules, computed once for every item); the last mark, or the end of the last image,
the item boundary (a text context shared by every question about it). Without marks a text-only request shares
only its system prompt. With `--preset flow` the server sizes its state slots for many items under one set of
rules (2 per item in flight plus 4), evicts the checkpoints that running requests depend on last, and computes a
shared prefix once even when many requests need it at the same time. Requests whose own prompt end will not be
reused can set `cache_prompt: false`. Cached prefixes are lossless: the answer is the same as with a computed one.

### Web chat

`GET /` (and `/chat`) serves a chat page built into the binary: one HTML
file with its script and style, nothing from another origin. It streams
replies from `/v1/chat/completions` and keeps the conversation (in the
browser's localStorage, as its settings), with a system prompt, the
temperature (the model's default when empty), max tokens (4,096 when
empty), thinking on or off, images pasted, dropped or attached (sent as
data URLs), a stop button and tokens per second, in light or dark. Model
text is set as text through a small Markdown subset (code blocks, lists,
tables, links to http(s) addresses only), never parsed as HTML, and a
Content-Security-Policy allows the page's own script and style (by hash),
images from data URLs and requests to this server only. The page is served
while the model loads and waits for it. Without an API key it needs no
setup: its requests come from the server's own origin, which the browser
guard lets through (other origins are still refused). With `--api-key` the
page itself is served without the key; it asks for the key and keeps it in
the browser. `--no-webui` turns it off (`GET /` then answers a line of
text, which Ollama clients use as a heartbeat).

### Ollama clients

The server answers Ollama's API on its own port: point an Ollama client at
it (`OLLAMA_HOST=http://127.0.0.1:8080` for the `ollama` CLI,
`Client(host="http://127.0.0.1:8080")` in the Python and JavaScript
libraries, the Ollama URL of apps such as Open WebUI). The routes:
`GET /api/version`, `GET /api/tags`, `GET /api/ps`, `POST /api/show`,
`POST /api/chat`, `POST /api/generate`, `POST /api/embed` and the legacy
`POST /api/embeddings`. Ollama's model management (`/api/pull`, `push`,
`create`, `copy`, `delete`, `blobs`) answers 404: the server runs the model
it started with.

- **One model.** Any `model` name is accepted, answered by the loaded model
  and echoed back, as Ollama echoes the name it was asked for. `/api/tags`
  and `/api/ps` list the loaded model as `NAME:latest` (its repo id or
  `--name`); `/api/show` gives its chat template (Jinja, not one of Ollama's
  Go templates), its sampling defaults as `parameters`, its context length in
  `model_info` and its `capabilities` (`completion`, `tools`, `thinking`,
  `vision`).
- **The chat route's job.** `/api/chat` and `/api/generate` (its `system` and
  `prompt` as two turns of the chat template; `"raw": true` sends the prompt
  as is, as `/v1/completions` does) become the body `/v1/chat/completions`
  would take, so a request gives the same tokens as the equivalent OpenAI
  request: messages with base64 `images` (placed before the message's text,
  where Ollama puts them), `tools` and `tool_calls` (arguments as objects),
  an assistant message's `thinking`, a tool result's `tool_name`, `logprobs`
  and `top_logprobs`. Replies stream as NDJSON (`application/x-ndjson`)
  unless `"stream": false`: the reasoning in `message.thinking` (`thinking`
  on `/api/generate`), tool calls in a `message.tool_calls` line before the
  last one, which has `done_reason` (`stop` or `length`) and Ollama's counts
  and durations (nanoseconds). A request with no messages or no prompt (how
  Ollama clients load a model) answers `"done_reason": "load"` at once.
- **think**: `true` or `false`, or `"low"`, `"medium"`, `"high"` for
  templates that take a reasoning effort (`high` is their `xhigh`); left
  out, the template's default. `true` on a model whose template has no
  thinking mode: 422.
- **options**: `temperature`, `top_p`, `top_k` (0 or less: off), `min_p`,
  `seed` (negative: a random one), `num_predict` (-1 or -2, the default:
  until the end of the turn or of the context window), `stop`,
  `repeat_penalty`, `presence_penalty` and `frequency_penalty` are the chat
  route's fields. Options a request leaves out take the model's own defaults
  (`generation_config.json`), as on `/v1/chat/completions`, not the defaults
  of Ollama's runner (temperature 0.8, top_k 40, repeat_penalty 1.1). The
  penalties are the chat route's: `repeat_penalty` over the whole
  conversation (Ollama's `repeat_last_n` -1; 0 turns every penalty off;
  another window while a penalty is on: 422), presence and frequency over
  the reply's tokens. `num_ctx`: the prompt and the reply within that many
  tokens (a longer prompt: 422, nothing is truncated; the server's
  `--max-ctx` still applies when it is smaller). `typical_p`, `tfs_z` and
  `mirostat` only at their off values (1, 1 and 0): 422 otherwise. The
  runner's own options (`num_gpu`, `num_thread`, `num_batch`, `num_keep`,
  `main_gpu`, `use_mmap`, `use_mlock`, `low_vram`, `numa`, ...) are accepted
  and have no effect: the server sizes and runs the model itself. Any other
  option: 422.
- **Accepted, no effect**: `keep_alive` (the model stays loaded while the
  server runs), `truncate` and `shift` (nothing is ever cut: a prompt that
  does not fit is an error, and a reply stops at the context window), and
  `/api/show`'s `verbose` (no tensor or vocabulary listing).
- **Structured output**: `format` (`"json"` or a JSON schema) constrains the
  reply as `response_format` does (`"json"`: one JSON object; a schema: a
  non-strict `json_schema`, so keywords the grammar cannot enforce are
  reported, not refused; see [structured output](#structured-output)).
- **Not supported (422)**: `/api/generate`'s `suffix`, `template` and
  `context`, and `images`, `system` or `think` with `"raw": true`.
- **Embeddings**: `/api/embed` (`input` a string or a list of strings,
  `dimensions`, `truncate`) and `/api/embeddings` (`prompt`) return
  `/v1/embeddings`' vectors, L2-normalized on both routes (Ollama's legacy
  route returns them unnormalized: dot products differ, cosine similarities
  do not). Generation options in an embedding request have no effect, as in
  Ollama; `num_ctx` there (which would cut inputs): 422.
- `GET /api/version` answers `0.12.0`, the Ollama API level these routes
  follow, for clients that check a minimum version; the server's own version
  is in `/health`. Errors have Ollama's shape, `{"error": "..."}`, also as
  the last line of a stream that fails.

### Coding agents

Claude Code, Codex CLI and opencode run against Espetero as they are: each
fixed a small Python package with failing unit tests (8 of 8 passing, the
tests untouched) on Qwen3.6-35B-A3B 6-bit served with `--context 65536`.
Give the server a context that holds the agent's system prompt and tools
plus the work; the prompt checkpoints keep the system prompt and the history
an agent re-sends every turn from being computed again.

- **Claude Code** (Anthropic Messages): any model name works, the server
  serves its one model.

  ```bash
  ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=local \
  ANTHROPIC_MODEL=local ANTHROPIC_DEFAULT_HAIKU_MODEL=local claude
  ```

- **Codex CLI** (OpenAI Responses), in `~/.codex/config.toml`. Espetero
  refuses tools it cannot run instead of ignoring them, so Codex's hosted web
  search and its sub-agent tools (sent as a namespace of tools) are turned off:

  ```toml
  model = "local"
  model_provider = "espetero"
  web_search = "disabled"

  [features]
  multi_agent = false

  [model_providers.espetero]
  name = "Espetero"
  base_url = "http://127.0.0.1:8080/v1"
  wire_api = "responses"
  env_key = "ESPETERO_API_KEY"   # the server's --api-key, or any value without one
  ```

- **opencode** (OpenAI Chat Completions), in `opencode.json`:

  ```json
  {"provider": {"espetero": {"npm": "@ai-sdk/openai-compatible", "name": "Espetero",
     "options": {"baseURL": "http://127.0.0.1:8080/v1", "apiKey": "local"},
     "models": {"local": {"name": "Espetero", "limit": {"context": 65536, "output": 8192}}}}},
   "model": "espetero/local"}
  ```

### Clef decision models

Cloudflare's Clef releases (`Cloudflare/clef-flash`: Qwen3.5-9B; `Cloudflare/clef`:
Qwen3.8-27B) are Jev-like classifiers: a Qwen3.5-family backbone plus a small
"joint schema head" that reads the backbone's final hidden states and gives a
probability for every allowed option of every question. They never generate
text, so Espetero serves them only through the decision API:

```
python tools/clef_convert.py ~/.cache/huggingface/hub/models--Cloudflare--clef-flash/snapshots/<rev> \
    ~/models/clef/clef-flash-bf16          # renames to Espetero's layout; --bits 8 for an 8-bit decoder
espetero serve --model ~/models/clef/clef-flash-bf16 --preset classify
```

A directory holding `joint_head.safetensors` + `joint_head_config.json` next to
the backbone is served as a Clef model: `POST /v1/systemone` (and `/v1/decisions`)
take the Jev / SystemOne body — `state` (text or JSON), `questions` (`noul`,
`choice`, `score` with `instructions` and `criteria`), optional `images` (data
URLs, `image_detail`), `max_length` (16,384), `return_logits` — and return
`answers` exactly as the release's `systemone()` does; the chat endpoints
(`/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/responses`) answer 422.
The prompt is token-identical to the release's `encode_record`; the backbone runs
the whole record once (no LM head), every row's final-normed state is read out
(`TreeSpec::out_hidden`), and the head runs on the CPU in fp32 (Accelerate).
The system prompt and the image block of a request with images are
checkpointed: the next request about the same images (another question set)
starts after them (4 kept, LRU; not shared across `cache_salt`s; a block longer
than one pass is computed uncached).

Images: by default a Clef request resizes images with the release's own
processor limits (65,536 to 16,777,216 pixels, from `processor_config.json`), so
a 1920×1080 photo is ~2,040 image tokens as in the release, not Espetero's ~1 MP
chat budget; `image_detail` (`standard` ~1 MP, `low`, `high`, `max`) or
`media_kwargs` (`min_pixels` / `max_pixels`, or `size`) choose another budget,
at most the release's own limit (larger `media_kwargs` values are clamped to
it); up to 16 images per request.
One image must fit one pass: raise `--max-rows` for very large photos (a 12 MP
photo is ~11,800 tokens). EXIF orientation is applied (as `transformers`'
`load_image` does). Known difference: integers beyond the int64 range in a JSON
state are rendered as floats.

### Limits for untrusted clients

Espetero 0.1.0 is meant for one trusted user, but the server refuses what a
stray web page or an abusive request could do with it:

- **Browsers.** Without `--api-key` the server answers only requests addressed
  to `localhost`, `127.0.0.1`, `[::1]` or its `--host` address (403 otherwise:
  DNS rebinding), and refuses requests whose `Origin` is another web origin
  unless `--cors` is given (403: cross-site requests). JSON routes take bodies
  sent as `Content-Type: application/json` only, `POST /v1/files`
  `multipart/form-data` only (415 otherwise; curl needs
  `-H 'Content-Type: application/json'`); Ollama's `/api/*` routes read any
  body as JSON, as Ollama does (curl's `-d` sends a form type), behind the
  same Host and Origin checks. Clients that are not browsers
  (curl, the OpenAI and Anthropic SDKs) send neither a foreign host name nor an
  `Origin`. The web chat page (`/`, `/chat`) is the one route served without
  the key: it holds no data and asks for the key. The server warns at
  startup when it listens beyond loopback, or allows CORS, without a key;
  `GET /v1/config` and `--print-config` never show the key.
- **Images.** At most 32 per chat request, 16 per `/v1/decisions` or Clef
  request. Each image's size is read from its header before any pixel is
  decoded: images over 64 MP get a 422 (`--image-max-mp N`; PIL's own limit of
  179 MP at most), and so does a request whose images and text cannot fit the
  context window (their token counts follow from the header with the
  processor's own resize rule). At most 4 images are decoded at once.
- **Images by URL**. Off by default: a chat, Responses
  (`input_image`) or Anthropic (`source.type` `url`) image given as an
  http(s) URL gets a 422 naming `--image-urls`. With `--image-urls` the server
  fetches it (Foundation's URL session: no cookies, cache, credentials or
  proxies): at most 32 MB per image and 128 MB per request (413 beyond), 20 s
  per image including redirects, at most 3 redirects, an `image/*` response
  with a 2xx status, 4 fetches at once; the bytes then take the data-URL path
  and limits above. Never from this machine or its networks: the host's
  addresses are checked before the request and again at each redirect, and
  the address each connection really used is checked after it (DNS
  rebinding), refusing loopback (127/8, ::1), private (10/8, 172.16/12,
  192.168/16), link-local (169.254/16, fe80::/10), unique-local (fc00::/7),
  carrier-grade NAT (100.64/10), multicast, unspecified and reserved
  addresses (and IPv4 addresses embedded in IPv6 ones), unless
  `--image-urls-private` is also given (images on this machine or its
  network). The check after the connection keeps the image from the client,
  not the request from the server: a name that resolves to a public address
  for the first check and to an internal one for the connection (DNS
  rebinding) still receives one GET from Espetero.
- **Text.** A prompt (the rendered chat, a completions prompt, `score`
  continuations, `/v1/tokenize` text) over 16 bytes per token of the context
  window (512 KB at 32K) gets a 413 before it is tokenized; real text averages
  3–5 bytes per token. `/v1/detokenize` takes at most a context window of ids.
- **Batch API.** Two batches run at once, the next 100 wait their turn (429
  beyond); uploads stop with a 413 once the data directory holds
  `--batch-disk-gb` (4 GB by default); a batch that fails to run is marked
  `failed`.
- **Memory per pass.** A request starts once the KV cache can hold its
  uncached prompt and its `max_tokens` beside what the running requests may
  still take (cached checkpoints no running request uses are evicted to make
  room, but only when that lets it start: a request that waits anyway leaves
  them cached); otherwise it waits (with nothing running it always
  starts). A `max_tokens` counts up to 8K tokens or a quarter of the KV
  cache, whichever is more, so a client that asks for a whole context window
  does not hold the others back.
  If the cache still runs out, the pass first skips the optional checkpoints
  of its prompts (system message, instructions, image, prompt end; a prompt's
  first chunk writes its rows twice for one) and of its decisions (their
  prefixes): the request runs and only that checkpoint is not cached
  (`kv_skipped_checkpoints` in `/v1/stats`). Then prompt chunks wait and the newest
  requests stop with a "KV cache full" error, so the earlier ones finish
  (`kv_stopped_sequences`); checkpoints are evicted only for what is still
  short after that. The error of a request that cannot fit even alone says
  what would: the KV free for its prompt, or the answer length it reached. A
  decode step takes a KV block only when its row opens one, and is counted so
  that a request alone fills the cache to its last block.
  An image or a `/v1/decisions` request that cannot fit one pass gets a 422
  naming `--max-rows`. Decode steps take a pass's rows first: a decision that
  fits a pass but not beside them waits at most 4 passes, then the newest
  requests' drafts, and if need be their decode steps, wait one pass for it
  (`decision_row_reservations` in `/v1/stats`; replies do not change).
  A stream that fails after it started ends with an
  `{"error": ...}` event and `[DONE]`.
- **Shutdown.** Ctrl-C (SIGINT or SIGTERM) ends queued and running requests
  once the current pass completes (they get a 503, unfinished batches resume
  at the next start); a second Ctrl-C exits at once. `--prefix-dir-gb` covers
  the whole directory: stored checkpoints of other builds or models go first
  (least recently used), and any unused for 30 days.
- **Files.** Batch files, stored chat checkpoints and caches are created
  readable by their user only (directories 0700, files 0600).

## 3. Advanced (every parameter)

`espetero serve --help-all` lists every option. Any of them can also go in a
JSON file, with the value it takes on the command line (`true` / `false` for
switches; a leading `--` on a key is allowed):

```json
{"preset": "classify", "image-detail": "high", "draft": 3, "no-lookup": true}
```

```
espetero serve --model PATH --config my.json --users 8
```

Without `--config`, `serve`, `chat`, `batch` and `info` read
`~/.config/espetero/config.json` (`$XDG_CONFIG_HOME/espetero/config.json`
when that is set) when it exists; `--config none` reads no file. A
file that names the model (`{"model": "mlx-community/Qwen3.8-27B-4bit"}`)
makes `espetero serve` alone serve it.

Precedence: preset < config file < command line; inside one layer an
explicit option beats a common option that expands to it. To see what the
engine will use and where each value comes from (`config_file` names the
file read, and its options have the source `config file`):

```
espetero serve --model PATH --print-config      # prints and exits
curl http://127.0.0.1:8080/v1/config         # the running server's settings
```

`effective.model` is the file or directory that runs (for a repo id,
`effective.model-repo` and `effective.model-revision` name it, and
`--prefix-dir` keeps each revision's checkpoints apart; `--print-config` never
downloads) and `effective.sampling-defaults` the values requests that leave
them out get.

Options that change the model's outputs (for example a smaller image budget
than the one requested, or checkpoints quantized below the file's own
precision) are never chosen automatically; they stay in the advanced tier.
Everything in the automatic tier either leaves outputs unchanged (drafts
are verified by the full model) or was validated on 6- and 8-bit MLX and Q6_K/Q8_0 GGUF files.

How a request shares a pass (beside other requests, after a prompt
checkpoint, verifying drafts) changes the order of the float sums in its
matrix products, so its log-probabilities move by about 0.001-0.01 nats on
dense models (`tools/invariance_check.py`: at most 0.02 on the Qwen3.8-27B
and the Gemma 4 31B). On mixture-of-experts models (Qwen3.6-35B-A3B, Gemma 4
26B-A4B) such a difference can, at a token whose k-th and next expert are
nearly tied, pick the other expert: that token's log-probabilities move by up
to a few tenths (0.2-0.9 nats at 2-5 tokens of 78 measured; mlx-lm's own
bf16 and fp32 runs of the 26B-A4B differ by more on average), and a greedy
reply can change from there, as in other engines that batch requests.
Drafts left the replies unchanged in every test (prompt lookup on and off,
dense and mixture-of-experts; the MTP head on the 27B). Embeddings run their
passes on one tile shape whatever their size and give the same bits in
every case.

Qwen3.8-Flash-Next past 2,051 tokens of context attends the way the model
was trained: each query reads its 512 best blocks of 4 tokens plus the last
≤ 3 (Qwen sparse attention), not every cached token. It is on with no
option (`ESPETERO_QSA=0` switches to dense attention, an
approximation of the model beyond 2,051 tokens, for comparisons only). It
needs 336 more bytes per token and attention layer for the index, which
`--memory-gb` accounts for.

Models that stream experts from the SSD (Flash-Next with an expert cache)
run each layer's MoE for the experts already in the cache while the missing
ones are read, then finish the rest once the reads are done (same
outputs). It is on with no option; `ESPETERO_Q4_SPLIT=0` turns it off for
comparisons. Decode and verify passes of up to 256 rows also guess the next
layer's experts and start reading the missing ones a layer early
(`ESPETERO_Q4_PF_ROWS=N` limits this read-ahead to passes of at most N rows).
`ESPETERO_MOE_CHECK=1` is a diagnostic: it reads the MoE tile table of long
passes back and checks it against the routing.

### Background service

`espetero service` runs the server in the background as a LaunchAgent of
this user (nothing runs as root or is written outside the home directory):

| Command | What it does |
|:--|:--|
| `espetero service install [--config FILE] [serve options]` | saves the options given (e.g. `--model ORG/NAME --port 8080`) in the config file (default `~/.config/espetero/config.json`, created readable by the user only: it may hold `api-key`), which must then name a model; writes `~/Library/LaunchAgents/espetero.server.plist`, which runs `espetero serve --config FILE` now, at every login and again when it fails or crashes (not after a clean stop; at most every 10 s), logging to `~/Library/Logs/espetero.log`; installing again restarts the server with the file's settings |
| `espetero service status` | whether the agent is installed and loaded, its process and last exit code, and what `GET /health` answers at the configured address (ready, loading the model, no answer) |
| `espetero service logs [--follow] [--lines N]` | the log's last lines (50), then what is appended |
| `espetero service uninstall` | stops the server and removes the LaunchAgent; the config file and the log stay |

`--dry-run` (install, uninstall) prints the files and the `launchctl`
commands without changing anything. The agent runs the binary by the path
it was installed with (Homebrew's `bin/espetero` link survives upgrades),
with the `PATH` and Hugging Face cache variables (`HF_HOME`,
`HF_HUB_CACHE`, `HF_HUB_OFFLINE`) of the shell that installed it, so it
finds the same model cache; secrets are not copied (put the key in the
file). It runs as an interactive process type: without one, launchd
throttles an agent's CPU and I/O. Homebrew's formula has the same service
for `brew services start espetero` (it reads the default config file, logs
to Homebrew's `var/log/espetero.log`); run one of the two, not both: they
would share the port.
