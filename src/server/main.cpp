// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// espetero — native Metal inference engine for Apple silicon.
//
//   espetero serve  --model PATH [--port 8080] ...   OpenAI-compatible server + decisions + batches
//   espetero chat   --model PATH                     interactive chat in the terminal
//   espetero batch  --model PATH --input in.jsonl --output out.jsonl
//   espetero info   --model PATH
//
// PATH is an MLX directory, a .gguf file or a Hugging Face repo id
// (ORG/NAME, ORG/NAME:QUANT for GGUF repositories; hub.h).
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <initializer_list>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "../engine/batch.h"
#include "../engine/chat.h"
#include "../engine/engine.h"
#include "../http/http.h"
#include "../tok/tokenizer.h"
#include "../vision/vision.h"
#include "api.h"
#include "hub.h"
#include "models.h"
#include "service.h"
#include "version.h"
#include "../util/paths.h"

using namespace espetero;

namespace {

struct Args {
  std::string cmd;
  std::vector<std::pair<std::string, std::string>> kv;
  std::vector<std::string> positional;
  std::vector<std::string> bare;  // options given without a value (switches; "1" in kv)
  std::string get(const std::string& k, const std::string& def = "") const {
    for (auto& [a, b] : kv)
      if (a == k) return b;
    return def;
  }
  bool has(const std::string& k) const {
    for (auto& [a, b] : kv)
      if (a == k) return true;
    return false;
  }
  int i(const std::string& k, int def) const { return has(k) ? std::atoi(get(k).c_str()) : def; }
  double d(const std::string& k, double def) const { return has(k) ? std::atof(get(k).c_str()) : def; }
  void set(const std::string& k, const std::string& v) {
    for (auto& [a, b] : kv)
      if (a == k) {
        b = v;
        return;
      }
    kv.push_back({k, v});
  }
};

Args parse(int argc, char** argv) {
  Args a;
  if (argc > 1) a.cmd = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {
      std::string key = s.substr(2), val = "1";
      const size_t eq = key.find('=');
      if (eq != std::string::npos) {
        val = key.substr(eq + 1);
        key = key.substr(0, eq);
      } else if (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
        val = argv[++i];
      } else {
        a.bare.push_back(key);
      }
      a.kv.push_back({key, val});
    } else {
      a.positional.push_back(s);
    }
  }
  return a;
}

void usage() {
  std::fprintf(stderr,
               "espetero — fast LLM inference for Apple silicon (Qwen3.5, Qwen3.6, Qwen3.8 and Flash-Next)\n\n"
               "quick start (the best settings for the model and this Mac are chosen automatically):\n"
               "  espetero serve --model PATH               server with OpenAI, Anthropic and Ollama APIs (chat, completions,\n"
               "                                            images, embeddings, batches)\n"
               "  espetero serve --model ORG/NAME           the same from Hugging Face (downloaded when missing):\n"
               "                                            mlx-community/Qwen3.8-27B-4bit, or a GGUF repository and its\n"
               "                                            quantization: unsloth/Qwen3.8-27B-GGUF:UD-Q6_K; a model\n"
               "                                            that cannot fit this Mac is refused before its download\n"
               "  then open http://127.0.0.1:8080 in a browser for the built-in chat (--no-webui turns it off)\n\n"
               "models:\n"
               "  espetero recommend                        the models that fit this Mac, best first, and the --model to pass\n"
               "  espetero pull ORG/NAME[:QUANT]            download one into the Hugging Face cache\n"
               "  espetero list                             the models on this Mac, what they need, whether they fit\n"
               "  espetero rm ORG/NAME[:QUANT]              remove one from the Hugging Face cache\n\n"
               "presets (same API; pick the workload):\n"
               "  --preset general      chat and completions with images (default)\n"
               "  --preset vision       photo-first: ~1 MP images, several questions per photo share its checkpoint\n"
               "  --preset classify     multimodal classification through /v1/decisions: every field scored from the\n"
               "                        logits in one pass (JEV-style), shared prefixes, repeat memo\n"
               "  --preset flow         many images under the same instructions, several requests per image (put the\n"
               "                        shared instructions before the image): instructions computed once, each image\n"
               "                        once for all its requests, prompts batched, drafts kept in batches\n\n"
               "common options:\n"
               "  --context N           tokens per conversation (default: preset)\n"
               "  --users N             concurrent requests (default: preset)\n"
               "  --image-detail L      image budget: low (0.26 MP), standard (~1 MP), high (~4 MP), max (the model's\n"
               "                        own 16.8 MP ceiling, mlx-vlm's default); requests can set detail per image,\n"
               "                        up to this level (--allow-image-detail-max lets them ask for more)\n"
               "  --speculation M       auto (default: MTP drafts while they pay), on (always), off\n"
               "  --memory-gb X         GPU memory for conversations (default: automatic)\n"
               "  --host H --port P     bind address (default 127.0.0.1:8080)   --api-key KEY   --name NAME\n"
               "                        (the key can come from ESPETERO_API_KEY instead, out of the process list)\n\n"
               "configuration:\n"
               "  --config FILE.json    any option below as {\"option\": value}; the command line wins (default:\n"
               "                        ~/.config/espetero/config.json when it exists; --config none: no file)\n"
               "  --print-config        show the resolved settings and where each comes from, then exit\n"
               "  --help-all            every option (advanced)\n\n"
               "background service (a LaunchAgent: at login, restarted if it fails):\n"
               "  espetero service install --model ORG/NAME [options]   options saved in the config file\n"
               "  espetero service status | logs [--follow] | uninstall\n");
}

void usage_all() {
  std::fprintf(stderr,
               "espetero — fast LLM inference for Apple silicon (Qwen3.5, Qwen3.6, Qwen3.8 and Flash-Next)\n\n"
               "usage:\n"
               "  espetero serve --model PATH [options]     HTTP server (OpenAI chat/completions/embeddings/batches,\n"
               "                                            Anthropic /v1/messages, Ollama /api/*, /v1/decisions)\n"
               "  espetero chat  --model PATH [options]     interactive chat\n"
               "  espetero batch --model PATH --input FILE --output FILE [--errors FILE] [--endpoint /v1/chat/completions]\n"
               "  espetero info  --model PATH\n"
               "  espetero service install|status|logs|uninstall [--config FILE] [serve options]   background server\n"
               "                                            (a LaunchAgent; espetero service help)\n"
               "  espetero recommend [--all] [--ram-gb N] [--context N]\n"
               "                                            the models of Espetero's table that fit this Mac, best first,\n"
               "                                            the downloaded ones marked (--all: every file and what it\n"
               "                                            needs; --ram-gb: for a Mac with N GB of RAM); also\n"
               "                                            `espetero models --recommend`, and serve without --model\n"
               "  espetero pull ORG/NAME[:QUANT]... [--skip-memory-check]\n"
               "                                            download into the Hugging Face cache (the newest revision;\n"
               "                                            a model that cannot fit this Mac is downloaded with a warning)\n"
               "  espetero list                             the supported models in the Hugging Face cache and in\n"
               "                                            ~/models/<family>/<name> (ESPETERO_MODELS_DIR): format,\n"
               "                                            quantization, size, need, fits, dense/MoE, MTP, vision\n"
               "  espetero rm ORG/NAME[:QUANT]... [--yes]   remove a repository (or one GGUF quantization) from the\n"
               "                                            Hugging Face cache, and the blobs nothing else links to\n"
               "  --config FILE.json    options as {\"option\": value}, the command line wins (default:\n"
               "                        ~/.config/espetero/config.json when it exists; --config none: no file)\n\n"
               "model / memory:\n"
               "  --model PATH|REPO     MLX directory, .gguf file (mmproj*.gguf next to it adds vision) or Hugging Face\n"
               "                        repo id: ORG/NAME (MLX) or ORG/NAME:QUANT (GGUF; QUANT ends the file name, e.g.\n"
               "                        unsloth/Qwen3.8-27B-GGUF:UD-Q6_K), from the hub cache ($HF_HUB_CACHE, else\n"
               "                        $HF_HOME/hub, else ~/.cache/huggingface/hub), downloaded there when missing (in\n"
               "                        huggingface_hub's layout, verified, resumable; HF_TOKEN for gated models,\n"
               "                        HF_ENDPOINT for a mirror)\n"
               "  --offline             never download a repo id (also HF_HUB_OFFLINE=1)\n"
               "  --skip-memory-check   download and load a model that does not fit this Mac's GPU memory (default: refuse\n"
               "                        it before its download and before loading it, with what it needs and smaller\n"
               "                        files of it that fit)\n"
               "  --name NAME           model name reported by the API (default: the repo id, else the file name)\n"
               "  --no-generation-config  requests that omit temperature / top_p / top_k / min_p get OpenAI's defaults\n"
               "                        (1, 1, off, 0) instead of the model's generation_config.json (GGUF:\n"
               "                        general.sampling.*); requests that set them always win\n"
               "  --max-ctx N           tokens per sequence (default 32768)\n"
               "  --slots N             state slots (default: automatic)\n"
               "  --kv-gb X             KV cache size (default: automatic)\n"
               "  --memory-gb X         budget for slots + KV (default: ~55%% of free GPU memory)\n"
               "  --expert-cache-gb X   models whose experts stay on the SSD (Qwen3.8-Flash-Next): GB of experts kept in\n"
               "                        memory (default: automatic, at most 40; -1: none)\n"
               "  --expert-store DIR|none  expert records for those models: one read per expert instead of nine\n"
               "                        (default: ~/Library/Caches/espetero/xstore.noindex/<model> when built)\n"
               "  --expert-store-build  write the records once where they are missing (~68 GB for Flash-Next)\n"
               "  --mtp PATH|none       MTP draft head (default: the model's own)\n"
               "  --mmproj PATH|none    GGUF vision projector\n"
               "  --dense LIST          decoder matrices served as fp16 (gate_up,down,in_proj,out|all)\n\n"
               "scheduling:\n"
               "  --max-seqs N          concurrent generations (default 64)\n"
               "  --draft N             MTP drafts per decode pass (default 5; Qwen3.8-Flash-Next 3; 0 = off)\n"
               "  --siblings N          MTP runner-ups as extra verify leaves: 1 depth 1, 2 depths 1-2 (default 0)\n"
               "  --draft-auto          MTP draft depth per pass from measured acceptance and costs (default: fixed --draft)\n"
               "  --mtp-vocab N         MTP drafts score only token ids below N (default 0: all; verification unchanged)\n"
               "  --mtp-coarse N        MTP drafts from a 2-bit LM head copy, N column blocks of 32 ids rescored exactly (default 256;\n"
               "                        0: the 4-bit draft head; replaces the 4-bit head unless --mtp-head4 is also given)\n"
               "  --no-mtp-q8           keep a bf16 MTP layer in bf16 (default: 8-bit copies; drafts only, verification unchanged)\n"
               "  --spec-max-seqs N     speculate only up to N decoding sequences (default 2)\n"
               "  --no-lookup           disable prompt-lookup drafts\n"
               "  --lookup-min N        a prompt-lookup copy needs N matching tokens (default 8; 3: any 3-gram)\n"
               "  --no-adaptive-drafts  draft on every step (default: only while drafting beats plain decoding)\n"
               "  --no-share-prefix     decisions: no shared-prefix scheduling (each job budgets and computes its own prefix)\n"
               "  --chain K             greedy decode steps per GPU chain when no drafts are in play (default 8, 0 = off)\n"
               "  --chain-after N       chain only after N generated tokens (default 16)\n"
               "  --no-gpu-topk         sampled requests read full logits on the CPU (default: GPU top-k when 1 <= top_k <= 32)\n"
               "  --legacy-sampling     mt19937 draws and random draft acceptance instead of exact position-keyed sampling\n"
               "  --no-gpu-draw         draw sampled tokens on the CPU (no chains for sampled requests)\n"
               "  --prefill-chunk N     prefill rows per pass while others decode (default 2048)\n"
               "  --no-image-checkpoint no chat checkpoint after the last image (default: questions about one image share it)\n"
               "  --no-instr-checkpoint no chat checkpoint after the user's instructions before the first image (default:\n"
               "                        photos under the same instructions share them)\n"
               "  --no-single-flight    requests sharing an uncached prefix each compute it (default: they wait for one)\n"
               "  --no-keep-live        plain LRU for checkpoints (default: those running requests continue are evicted last)\n"
               "  --no-prompt-checkpoint no chat checkpoint at the prompt end (later turns re-read the prompt; requests can\n"
               "                        ask for it with \"cache_prompt\": true)\n"
               "  --no-decode-first     prompt chunks run beside decoding sequences at once (default: they wait up to\n"
               "                        --prefill-wait-ms, 2500 (200 in --preset general), and prompts beyond\n"
               "                        --prefill-batch-rows, 512, wait a pass)\n"
               "  --image-pixels N      default image budget (default 131072)\n"
               "  --allow-image-detail-max  requests may ask for an image detail above the server's own budget (high, max:\n"
               "                        up to the 16.8 MP ceiling); default: such a request gets a 422\n"
               "  --image-max-mp N      largest image accepted, in megapixels as its header declares them (default 64;\n"
               "                        at most 178.96, PIL's limit); larger ones get a 422 before any pixel is decoded\n"
               "  --image-urls          fetch http(s) image URLs of chat, Responses and Anthropic requests (default: off,\n"
               "                        only data URLs; such a request gets a 422): at most 32 MB per image, 20 s, 3\n"
               "                        redirects, an image/* response, never from loopback, private, link-local or other\n"
               "                        non-public addresses (checked before the request, at each redirect and on the\n"
               "                        connection itself)\n"
               "  --image-urls-private  with --image-urls: those addresses too (images on this machine or its network)\n"
               "  --embedding-pooling P /v1/embeddings: last (default: the last token's final-normed hidden state, as\n"
               "                        Qwen3-Embedding pools) or mean (every token's); L2-normalized\n"
               "  --no-exif-rotate      ignore EXIF orientation (default: rotate like PIL's exif_transpose / mlx-vlm)\n"
               "  --memo N              exact-repeat decision memo entries (default 0)\n\n"
               "structured output (no option):\n"
               "  response_format json_object / json_schema (chat, completions, Batch lines), Responses text.format and\n"
               "                        Anthropic output_config.format constrain the reply with a JSON grammar (masked tokens;\n"
               "                        the model thinks first when thinking is on); \"strict\": true refuses keywords it cannot\n"
               "                        enforce (pattern, format, ...), otherwise the response lists them; details in\n"
               "                        docs/configuration.md. ESPETERO_GRAMMAR_CHECK=1 checks every masked argmax against the full\n"
               "                        logits (development)\n\n"
               "server:\n"
               "  --host H --port P     bind address (default 127.0.0.1:8080), opened before the model loads: a port in\n"
               "                        use fails at once, and requests get 503 {\"status\": \"loading\"} until it is ready\n"
               "  --api-key KEY         require Authorization: Bearer KEY (or x-api-key: KEY); the environment variable\n"
               "                        ESPETERO_API_KEY sets it too, out of the process list. Without a key the server\n"
               "                        only answers requests addressed to localhost, 127.0.0.1, [::1] or --host, and\n"
               "                        refuses requests from web pages of other origins (DNS rebinding, cross-site requests)\n"
               "  --cors                allow cross-origin requests (web pages of any origin)\n"
               "  --no-webui            no chat page at / and /chat (GET / answers a line of text instead)\n"
               "  --data-dir DIR        Batch API storage (default ~/.cache/espetero/batches)\n"
               "  --batch-concurrency N requests in flight per batch (default = max-seqs; 2 batches run at once, the\n"
               "                        rest wait their turn)\n"
               "  --batch-disk-gb X     Batch API storage at most X GB: uploads beyond it get a 413 (default 4)\n"
               "  --verbose             log requests\n"
               "  --profile             per-kernel GPU times at GET /v1/profile\n"
               "  --prefix-dir [DIR|none] persist chat checkpoints to disk (default ~/.cache/espetero/prefix): a\n"
               "                        restarted server loads a long system prompt or instructions instead of\n"
               "                        computing them (on in --preset flow)\n"
               "  --prefix-dir-gb N     disk used by --prefix-dir at most (default 20; least recently used first)\n");
}

// ---- configuration tiers ----------------------------------------------
// Basic: `espetero serve --model PATH` (preset general). Medium: --preset and the
// common options of --help. Advanced: every flag of --help-all, or the same
// keys in --config FILE.json. Precedence: preset < config file < command
// line; within a layer an explicit option beats a common option that expands
// to it. --print-config / GET /v1/config report each value and its source.
struct PresetDef {
  const char* name;
  const char* what;
  std::vector<std::pair<std::string, std::string>> kv;
};

const std::vector<PresetDef>& preset_defs() {
  static const std::vector<PresetDef> p = {
      // A new chat waits at most 200 ms while others decode (the
      // 2,500 ms decode-first wait made its first token 3.6 s on the 27B
      // beside a running reply; 1.1 s with 200 ms, the other reply no slower).
      // The photo presets keep the long wait, which batches their prompts.
      {"general", "chat and completions with images, balanced defaults",
       {{"image-pixels", "1003520"}, {"max-ctx", "32768"}, {"max-seqs", "64"}, {"prefill-wait-ms", "200"}}},
      {"vision", "photo-first: ~1 MP images, questions about one photo share its checkpoint",
       {{"image-pixels", "1003520"}, {"max-ctx", "16384"}, {"max-seqs", "16"}}},
      {"classify", "multimodal /v1/decisions: fields scored from the logits in one pass, shared prefixes, memo",
       {{"image-pixels", "1003520"}, {"max-ctx", "16384"}, {"max-seqs", "32"}, {"memo", "256"}}},
      // Many images under the same instructions, several requests per image
      // (e.g. many images, a few requests each, under a long shared system prompt).
      {"flow", "many images under the same instructions, several requests per image: system prompt and instructions "
               "computed once (kept on disk across restarts), each image once for all its requests, prompts batched, "
               "drafts kept in batches",
       {{"image-pixels", "1003520"}, {"max-ctx", "32768"}, {"max-seqs", "16"}, {"slots", "36"}, {"prefill-batch-rows", "4096"},
        {"spec-max-seqs", "16"}, {"no-prompt-checkpoint", "1"}, {"prefix-dir", "default"}}},
  };
  return p;
}

struct Resolved {
  Args args;
  json::Value report;
};

[[noreturn]] void config_error(const std::string& m) {
  std::fprintf(stderr, "espetero: %s\n", m.c_str());
  std::exit(2);
}

Resolved resolve_settings(const Args& cli) {
  std::vector<std::tuple<std::string, std::string, std::string>> set;  // option, value, source
  auto put = [&](const std::string& k, const std::string& v, const std::string& src) {
    for (auto& [kk, vv, ss] : set)
      if (kk == k) {
        vv = v;
        ss = src;
        return;
      }
    set.emplace_back(k, v, src);
  };
  auto drop = [&](const std::string& k) {
    for (size_t i = 0; i < set.size(); ++i)
      if (std::get<0>(set[i]) == k) set.erase(set.begin() + long(i)), --i;
  };
  // Common options expand to advanced ones, then the layer's own options apply.
  auto apply_layer = [&](const std::vector<std::pair<std::string, std::string>>& kv, const std::string& src) {
    for (const auto& [k, v] : kv) {
      if (k == "context") put("max-ctx", v, src + " (--context)");
      else if (k == "users") put("max-seqs", v, src + " (--users)");
      else if (k == "image-detail") {
        const char* px = v == "low" ? "262144" : v == "standard" ? "1003520" : v == "high" ? "4014080" : v == "max" ? "16777216" : nullptr;
        if (!px) config_error("--image-detail: low, standard, high or max");
        put("image-pixels", px, src + " (--image-detail " + v + ")");
      } else if (k == "speculation") {
        if (v == "off") {
          put("draft", "0", src + " (--speculation off)");
          put("no-lookup", "1", src + " (--speculation off)");
        } else if (v == "on") {
          put("no-adaptive-drafts", "1", src + " (--speculation on)");
        } else if (v != "auto") {
          config_error("--speculation: auto, on or off");
        }
      }
    }
    for (const auto& [k, v] : kv) {
      if (k == "context" || k == "users" || k == "image-detail" || k == "speculation" || k == "preset" || k == "config" ||
          k == "print-config")
        continue;
      if (v == "false") drop(k);
      else put(k, v == "true" ? "1" : v, src);
    }
  };
  // Config file: {"option": value}; true/false for switches. --config FILE,
  // else the user's own file when it exists (~/.config/espetero/
  // config.json, which `espetero service install` writes); --config none: no file.
  std::vector<std::pair<std::string, std::string>> file_kv;
  std::string file_preset, path;
  if (cli.has("config")) {
    path = cli.get("config");
    if (path == "none") path.clear();
  } else if (const std::string d = service::default_config(paths::home()); ::access(d.c_str(), F_OK) == 0) {
    path = d;
  }
  if (!path.empty()) {
    std::ifstream f(path);
    if (!f) config_error("cannot read the config file " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    json::Value j;
    try {
      j = json::parse(ss.str());
    } catch (const std::exception& e) {
      config_error("config file " + path + ": " + e.what());
    }
    if (!j.is_object()) config_error("config file " + path + ": expected a JSON object");
    const json::Object& obj = j.as_object();
    for (size_t oi = 0; oi < obj.size(); ++oi) {
      std::string k = obj.keys[oi];
      if (k.rfind("--", 0) == 0) k.erase(0, 2);  // "--port" as on the command line
      const json::Value& v = obj.vals[oi];
      std::string val;
      if (v.is_string()) val = v.as_string();
      else if (v.is_bool()) val = v.as_bool() ? "true" : "false";
      else val = json::dump(v);
      if (k == "preset") file_preset = val;
      file_kv.emplace_back(k, val);
    }
  }
  std::string preset = cli.has("preset") ? cli.get("preset") : (file_preset.empty() ? "general" : file_preset);
  if (preset == "auto") preset = "general";
  const PresetDef* pd = nullptr;
  for (const PresetDef& d : preset_defs())
    if (preset == d.name) pd = &d;
  if (!pd) config_error("--preset: general, vision, classify or flow");
  for (const auto& [k, v] : pd->kv) put(k, v, "preset " + preset);
  apply_layer(file_kv, "config file");
  apply_layer(cli.kv, "command line");
  // A flow holds a running request and its image checkpoint per user, beside
  // the system-prompt and instruction checkpoints: 2 slots per user and 4 more
  // unless --slots is given (fewer evict the image checkpoints of images still
  // decoding, and their next requests compute the image again).
  if (preset == "flow")
    for (auto& [k, v, src] : set)
      if (k == "slots" && src == "preset flow")
        for (const auto& [k2, v2, src2] : set)
          if (k2 == "max-seqs") {
            v = std::to_string(2 * std::max(1, std::atoi(v2.c_str())) + 4);
            src = "preset flow (2 x users + 4)";
          }
  Resolved r;
  r.args.cmd = cli.cmd;
  r.args.positional = cli.positional;
  json::Value settings = json::Value::object();
  for (const auto& [k, v, src] : set) {
    r.args.kv.push_back({k, v});
    json::Value e = json::Value::object();
    e.set("value", k == "api-key" ? "<redacted>" : v);  // served at GET /v1/config
    e.set("source", src);
    settings.set(k, e);
  }
  if (!cli.has("api-key") && !settings.get("api-key"))
    if (const char* k = std::getenv("ESPETERO_API_KEY"); k && *k) {
      json::Value e = json::Value::object();
      e.set("value", "<redacted>");
      e.set("source", "environment (ESPETERO_API_KEY)");
      settings.set("api-key", e);
    }
  r.report = json::Value::object();
  r.report.set("config_file", path.empty() ? json::Value() : json::Value(path));
  r.report.set("preset", preset);
  r.report.set("preset_description", pd->what);
  r.report.set("explicit", settings);
  return r;
}

// The resolved values the engine will use (every option of --help-all that
// matters for serving), for --print-config and GET /v1/config. `repo`: the
// Hugging Face repo id --model named, if any.
json::Value effective_json(const EngineOptions& o, const std::string& repo = "") {
  json::Value e = json::Value::object();
  e.set("model", o.model_path);
  if (!repo.empty()) e.set("model-repo", repo);
  if (!o.model_revision.empty()) e.set("model-revision", o.model_revision);
  // Defaults of requests that omit them (the model's generation_config.json).
  e.set("generation-config", o.generation_config);
  json::Value sd = read_sampling_defaults(o.model_path).to_json(o.generation_config);
  struct stat st;
  if (o.generation_config && ::stat(o.model_path.c_str(), &st) != 0) sd.set("source", "known once the model is downloaded");
  e.set("sampling-defaults", std::move(sd));
  e.set("max-ctx", int64_t(o.load.max_ctx));
  e.set("slots", o.load.slots ? json::Value(int64_t(o.load.slots)) : json::Value("auto"));
  e.set("memory-gb", o.load.memory_gb > 0 ? json::Value(o.load.memory_gb) : json::Value("auto"));
  e.set("expert-cache-gb", o.load.expert_cache_gb != 0 ? json::Value(o.load.expert_cache_gb) : json::Value("auto"));
  e.set("expert-store", o.load.expert_store.empty() ? json::Value("auto") : json::Value(o.load.expert_store));
  e.set("max-seqs", int64_t(o.max_seqs));
  e.set("draft", o.draft_tokens >= 0 ? json::Value(int64_t(o.draft_tokens)) : json::Value("auto"));
  e.set("adaptive-drafts", o.adaptive_drafts);
  e.set("draft-auto", o.draft_auto);
  e.set("siblings", int64_t(o.sibling_drafts));
  e.set("mtp-coarse", int64_t(o.load.mtp_coarse));
  e.set("prompt-lookup", o.prompt_lookup);
  e.set("image-pixels", int64_t(o.image_max_pixels));
  e.set("allow-image-detail-max", o.allow_image_detail_max);
  e.set("image-urls", o.image_urls);
  e.set("image-urls-private", o.image_urls_private);
  e.set("embedding-pooling", o.embedding_pooling);
  e.set("image-max-mp", double(vision::max_image_pixels()) / 1e6);
  e.set("image-checkpoint", o.image_checkpoint);
  e.set("instr-checkpoint", o.instr_checkpoint);
  e.set("single-flight", o.single_flight);
  e.set("keep-live", o.keep_live);
  e.set("prompt-checkpoint", o.prompt_checkpoint);
  e.set("exif-rotate", vision::exif_orientation_enabled());
  e.set("chain", int64_t(o.chain_steps));
  e.set("prefill-chunk", int64_t(o.prefill_chunk));
  e.set("memo", int64_t(o.memo));
  e.set("share-prefix", o.share_prefix);
  e.set("prefix-dir", o.prefix_dir.empty() ? json::Value("off") : json::Value(o.prefix_dir));
  e.set("prefix-dir-gb", o.prefix_dir_gb);
  e.set("vision", o.load.vision);
  return e;
}

std::string g_model_revision;  // the Hugging Face snapshot --model resolved to (resolve_model)

// The options the memory a model needs depends on (models.h).
models::Plan plan_of(const Args& a) {
  models::Plan p;
  p.max_ctx = a.i("max-ctx", 32768);
  p.slots = a.i("slots", 0);
  p.max_rows = a.i("max-rows", 4096);
  p.kv_gb = a.d("kv-gb", 0);
  p.memory_gb = a.d("memory-gb", 0);
  p.expert_cache_gb = a.d("expert-cache-gb", 0);
  p.vision = !a.has("no-vision");
  p.mtp = a.get("mtp") != "none";
  return p;
}

std::string recommendations(const Args& a, bool all);

EngineOptions engine_options(const Args& a) {
  EngineOptions o;
  o.model_path = a.get("model");
  if (o.model_path.empty()) {  // what to pass
    std::fprintf(stderr, "espetero %s needs a model: --model PATH (an MLX directory or a .gguf file) or --model ORG/NAME "
                         "from Hugging Face.\n%s\nRun one: espetero %s --model ID. espetero --help lists the options.\n",
                 a.cmd.c_str(), recommendations(a, false).c_str(), a.cmd.c_str());
    std::exit(2);
  }
  o.model_name = a.get("name");
  o.model_revision = g_model_revision;
  o.generation_config = !a.has("no-generation-config");
  o.load.max_ctx = a.i("max-ctx", 32768);
  o.load.slots = a.i("slots", 0);
  o.load.kv_gb = a.d("kv-gb", 0);
  o.load.memory_gb = a.d("memory-gb", 0);
  o.load.expert_cache_gb = a.d("expert-cache-gb", 0);
  o.load.expert_store = a.get("expert-store");
  o.load.expert_store_build = a.has("expert-store-build");
  o.load.mtp = a.get("mtp");
  o.load.mmproj = a.get("mmproj");
  o.load.dense = a.get("dense");
  o.load.max_rows = a.i("max-rows", 4096);
  o.load.attn_split = a.i("attn-split", 512);
  o.load.mtp_vocab = a.i("mtp-vocab", 0);
  // Automatic default: MTP drafts from the coarse 2-bit head with 256 blocks
  // rescored exactly (drafts nearly always equal the exact head's, draft
  // passes about 13 % cheaper than with the 4-bit head, a few percent faster
  // decoding on MoE and dense models); models without
  // an MTP head are unaffected. --mtp-coarse 0 restores the 4-bit head.
  o.load.mtp_coarse = a.i("mtp-coarse", 256);
  if (o.load.mtp_coarse > 0 && !a.has("mtp-head4")) o.load.mtp_head4 = false;  // the coarse head replaces it
  o.load.mtp_q8 = !a.has("no-mtp-q8");
  o.load.tune_max_m = a.i("tune-max-m", 1024);
  o.load.vision = !a.has("no-vision");
  o.max_seqs = a.i("max-seqs", 64);
  if (o.max_seqs < 1) config_error("--max-seqs (--users): at least 1");  // 0 left every request waiting forever
  o.draft_tokens = a.i("draft", -1);  // -1: the model's default (5; Qwen4-Exp 3)
  o.sibling_drafts = a.i("siblings", 0);
  o.draft_auto = a.has("draft-auto");
  o.draft_max = a.i("draft-max", 7);
  o.spec_max_seqs = a.i("spec-max-seqs", 2);
  o.prompt_lookup = !a.has("no-lookup");
  o.lookup_min = a.i("lookup-min", 8);
  o.adaptive_drafts = !a.has("no-adaptive-drafts");
  o.share_prefix = !a.has("no-share-prefix");
  o.chain_steps = a.i("chain", 8);
  o.chain_after = a.i("chain-after", 16);
  o.gpu_topk = !a.has("no-gpu-topk");
  o.exact_sampling = !a.has("legacy-sampling");
  o.gpu_draw = !a.has("no-gpu-draw");
  if (const char* v = std::getenv("ESPETERO_GPU_TOPK")) o.gpu_topk = *v == '1';
  if (const char* v = std::getenv("ESPETERO_CHAIN")) o.chain_steps = std::atoi(v);
  o.prefill_chunk = a.i("prefill-chunk", 2048);
  o.decode_first = !a.has("no-decode-first");
  o.image_checkpoint = !a.has("no-image-checkpoint");
  o.instr_checkpoint = !a.has("no-instr-checkpoint");
  o.single_flight = !a.has("no-single-flight");
  o.keep_live = !a.has("no-keep-live");
  o.prompt_checkpoint = !a.has("no-prompt-checkpoint");
  o.prefill_wait_ms = a.i("prefill-wait-ms", 2500);
  o.prefill_batch_rows = a.i("prefill-batch-rows", 512);
  o.image_max_pixels = a.i("image-pixels", 131072);
  o.allow_image_detail_max = a.has("allow-image-detail-max");
  o.image_urls = a.has("image-urls");
  o.image_urls_private = a.has("image-urls-private");
  if (o.image_urls_private && !o.image_urls) config_error("--image-urls-private needs --image-urls");
  o.embedding_pooling = a.get("embedding-pooling", "last");
  if (o.embedding_pooling != "last" && o.embedding_pooling != "mean") config_error("--embedding-pooling: last or mean");
  if (a.has("image-max-mp")) {
    const double mp = a.d("image-max-mp", 64);
    if (!(mp > 0)) config_error("--image-max-mp: a positive number of megapixels");
    vision::set_max_image_pixels(int64_t(std::min(mp, 1000.0) * 1e6));
  }
  if (a.has("no-exif-rotate")) vision::set_exif_orientation(false);
  o.memo = size_t(a.i("memo", 0));
  o.verbose = a.has("verbose");
  o.profile = a.has("profile");
  if (a.has("prefix-dir")) {
    std::string d = a.get("prefix-dir", std::string("default"));
    if (d == "default") d = paths::user_cache_dir() + "/prefix";
    o.prefix_dir = d == "none" || d == "off" ? std::string() : d;
  }
  o.prefix_dir_gb = a.d("prefix-dir-gb", 20);
  return o;
}

// --model ORG/NAME[:QUANT] (a Hugging Face repo id; hub.h) becomes its
// snapshot in the hub cache, downloaded there first when missing (only
// looked up for --print-config). The API reports the repo id as the model
// name unless --name is given. Returns the repo id ("" for a path).
std::string resolve_model(Args& a, bool look_only) {
  const std::string id = a.get("model");
  hub::Ref ref;
  if (!hub::parse_ref(id, ref)) return "";
  hub::Model m;
  if (look_only) {
    if (!hub::find_cached(hub::cache_dir(), ref, m)) return id;  // not downloaded yet: --model stays the id
  } else {
    hub::ResolveOptions ro;  // refused before its download when it cannot fit this Mac
    ro.offline = a.has("offline") || hub::offline_env();
    ro.memory = a.has("skip-memory-check") ? hub::MemoryCheck::Skip : hub::MemoryCheck::Refuse;
    ro.plan = plan_of(a);
    m = hub::resolve(ref, ro);
    try {  // the engine checks again; here a model of another family is named by its repo id
      check_model_supported(m.path);
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      throw std::runtime_error(msg.compare(0, m.path.size(), m.path) == 0 ? id + msg.substr(m.path.size()) : msg);
    }
  }
  a.set("model", m.path);
  if (!a.has("name")) a.set("name", id);
  if (!m.mmproj.empty() && !a.has("mmproj")) a.set("mmproj", m.mmproj);
  if (!m.mtp.empty() && !a.has("mtp")) a.set("mtp", m.mtp);
  g_model_revision = m.revision;
  return id;
}

// The model's footprint (its files as the engine holds them, the pools
// and scratch at the settings in `a`) against this Mac's GPU memory, before
// the engine loads it. A model that cannot fit is refused with the numbers,
// a --context that fits and smaller files of the same model; files that
// cannot be read are left to the loader's own messages.
void check_memory(const Args& a, const std::string& repo) {
  if (a.has("skip-memory-check")) return;
  const std::string path = a.get("model");
  models::Footprint f;
  try {
    f = models::footprint_local(path, a.get("mmproj"), a.get("mtp"));
  } catch (const std::exception&) {
    return;
  }
  if (!f.geo.known || !f.geo.unsupported.empty()) return;
  const models::Plan plan = plan_of(a);
  const models::Mac mac = models::this_mac();
  const models::Verdict v = models::judge(f, plan, mac);
  if (v.fits) return;
  hub::Ref ref;
  const std::vector<models::Alternative> alts =
      !repo.empty() && hub::parse_ref(repo, ref) ? hub::alternatives(ref, f, a.has("offline") || hub::offline_env())
                                                 : models::local_alternatives(path, a.get("mmproj"), a.get("mtp"));
  throw std::runtime_error(models::refusal(repo.empty() ? models::tilde(path) : repo, f, plan, mac, v, alts,
                                           "`espetero recommend` lists the models that fit this Mac; "
                                           "--skip-memory-check loads it anyway."));
}

// The catalog's models that fit this Mac (models.h), the ones in the
// Hugging Face cache marked. --ram-gb N: for a Mac with N GB instead.
std::string recommendations(const Args& a, bool all) {
  if (a.has("ram-gb") && !(a.d("ram-gb", 0) > 0)) config_error("--ram-gb: the RAM of the Mac, in GB");
  const models::Mac mac = a.has("ram-gb") ? models::mac_with_ram(a.d("ram-gb", 0)) : models::this_mac();
  const std::string cache = hub::cache_dir();
  auto cached = [&](const std::string& id) {
    hub::Ref r;
    hub::Model m;
    try {
      return hub::parse_ref(id, r) && hub::find_cached(cache, r, m);
    } catch (const std::exception&) {
      return false;
    }
  };
  return models::recommendations_text(mac, models::recommend(mac, plan_of(a), cached, all), all);
}

// ---- the models on this Mac: pull, list, rm --------------------------
// Switches that take no value: Args takes the word after a switch as its
// value (`rm --yes ORG/NAME`), so a value that is not a boolean goes back
// to the positional arguments.
void switches(Args& a, std::initializer_list<const char*> names) {
  for (const char* n : names)
    for (auto& [k, v] : a.kv)
      if (k == n && v != "1" && v != "true" && v != "false") {
        a.positional.insert(a.positional.begin(), v);
        v = "1";
      }
}

std::vector<hub::Ref> repo_ids(const Args& a, const char* cmd) {
  if (a.positional.empty()) config_error(std::string("espetero ") + cmd + " ORG/NAME[:QUANT] ...: which models?");
  std::vector<hub::Ref> out;
  for (const std::string& id : a.positional) {
    hub::Ref r;
    if (!hub::parse_ref(id, r))
      config_error(std::string("espetero ") + cmd + " takes Hugging Face repo ids (ORG/NAME, or ORG/NAME:QUANT for a GGUF "
                   "model); " + id + " is not one" + (std::strcmp(cmd, "rm") == 0 ? " (local models are removed by hand)" : ""));
    out.push_back(r);
  }
  return out;
}

// espetero pull ID...: download into the Hugging Face cache (the newest
// revision), warning when the model cannot fit this Mac.
int cmd_pull(const Args& a) {
  hub::ResolveOptions ro;
  ro.offline = a.has("offline") || hub::offline_env();
  ro.memory = a.has("skip-memory-check") ? hub::MemoryCheck::Skip : hub::MemoryCheck::Warn;
  ro.plan = plan_of(a);
  for (const hub::Ref& r : repo_ids(a, "pull")) {
    const std::string id = r.repo + (r.quant.empty() ? "" : ":" + r.quant);
    const hub::Model m = hub::pull(r, ro);
    std::printf("%s: %s\n", id.c_str(), m.path.c_str());
  }
  std::printf("Run it: espetero serve --model ID\n");
  return 0;
}

// The models directory: ESPETERO_MODELS_DIR, else ~/models.
std::string models_dir() {
  const char* d = std::getenv("ESPETERO_MODELS_DIR");
  return d && *d ? std::string(d) : paths::home() + "/models";
}

// espetero list: the supported models in the Hugging Face cache and under
// ~/models, with what they need and whether they fit this Mac.
int cmd_list(const Args& a) {
  int other = 0;
  const std::string cache = hub::cache_dir(), dir = models_dir();
  const std::vector<hub::Installed> items = hub::installed(cache, dir, &other);
  const models::Mac mac = models::this_mac();
  const models::Plan plan = plan_of(a);
  size_t w = 5;
  for (const hub::Installed& it : items) w = std::max(w, it.id.size());
  std::printf("%-*s  %-4s  %-18s  %8s  %9s  %-7s  %-24s  %s\n", int(w), "MODEL", "FMT", "QUANT", "ON DISK", "NEEDS", "FITS",
              "KIND", "MTP VISION");
  for (const hub::Installed& it : items) {
    const models::Footprint& f = it.footprint;
    const models::Verdict v = models::judge(f, plan, mac);
    const std::string fits = v.fits ? "yes" : v.context_that_fits ? std::to_string(v.context_that_fits / 1024) + "K ctx" : "no";
    const std::string kind = f.geo.q4 ? "MoE, experts on the SSD" : f.geo.E ? "MoE" : "dense";
    std::printf("%-*s  %-4s  %-18s  %8s  %9s  %-7s  %-24s  %-3s %s\n", int(w), it.id.c_str(), f.format.c_str(),
                f.quant.substr(0, 18).c_str(), models::gb(it.disk).c_str(), models::gb(v.need.total()).c_str(), fits.c_str(),
                kind.c_str(), f.has_mtp ? "yes" : "no", f.has_vision ? "yes" : "no");
  }
  if (items.empty()) std::printf("(no supported model)\n");
  std::printf("\nIn %s and %s (ESPETERO_MODELS_DIR). NEEDS: GPU memory at --context %d; FITS: on this Mac (%s for the "
              "GPU)%s.\n", models::tilde(cache).c_str(), models::tilde(dir).c_str(), plan.max_ctx, models::gb(mac.usable()).c_str(),
              other ? (", " + std::to_string(other) + " models of other families or formats not shown").c_str() : "");
  return 0;
}

// espetero rm ID...: remove from the Hugging Face cache after a confirmation
// (--yes: none).
int cmd_rm(const Args& a) {
  const std::string cache = hub::cache_dir();
  for (const hub::Ref& r : repo_ids(a, "rm")) {
    const std::string id = r.repo + (r.quant.empty() ? "" : ":" + r.quant);
    const hub::Removal rm = hub::plan_removal(cache, r);
    std::fprintf(stderr, "%s in %s: %s, %s\n", id.c_str(), models::tilde(rm.repo_dir).c_str(),
                 rm.whole ? "the whole repository" : (std::to_string(rm.files.size()) + " file(s)").c_str(),
                 models::size_text(rm.bytes).c_str());
    for (size_t i = 0; i < rm.files.size() && i < 12; ++i) std::fprintf(stderr, "  %s\n", rm.files[i].c_str());
    if (rm.files.size() > 12) std::fprintf(stderr, "  ... and %zu more\n", rm.files.size() - 12);
    if (!a.has("yes")) {
      if (!::isatty(0)) config_error("not removed: confirm with --yes (no terminal to ask on)");
      std::fprintf(stderr, "Remove it? [y/N] ");
      std::string answer;
      if (!std::getline(std::cin, answer) || (answer != "y" && answer != "Y" && answer != "yes")) {
        std::fprintf(stderr, "Nothing was removed.\n");
        return 1;
      }
    }
    hub::remove(cache, rm);
    std::printf("Removed %s: %s freed.\n", id.c_str(), models::size_text(rm.bytes).c_str());
  }
  return 0;
}

int cmd_recommend(const Args& a) {
  std::fputs(recommendations(a, a.has("all")).c_str(), stdout);
  std::printf("Run one: espetero serve --model ID (downloaded into the Hugging Face cache when missing).\n");
  return 0;
}

json::Value g_config;  // resolved settings: --print-config, GET /v1/config

// Shutdown. The first SIGINT / SIGTERM wakes a watcher thread (a signal
// handler may only do async-signal-safe work) that stops the HTTP server, the
// batch runners and the engine: the engine ends what is queued or running
// once its current pass completes, so handlers return at once instead of
// after their generations (shutdown took minutes). A second signal exits at
// once, whatever is still running.
std::atomic<int> g_signals{0};
int g_stop_pipe[2] = {-1, -1};

void on_signal(int) {
  if (g_signals.fetch_add(1) > 0) {
    static const char msg[] = "\n[espetero] second signal: exiting now\n";
    (void)!::write(2, msg, sizeof msg - 1);
    ::_exit(130);
  }
  const char c = 's';
  if (g_stop_pipe[1] >= 0) (void)!::write(g_stop_pipe[1], &c, 1);
}

int cmd_info(const Args& a) {
  EngineOptions o = engine_options(a);
  o.load.tune_max_m = a.i("tune-max-m", 0);
  Engine e(o);
  std::printf("%s\n", json::dump(e.info().to_json(), json::DumpOptions{", ", ": ", false, 2, false}).c_str());
  return 0;
}

// `serve` opens its port before the model is resolved (downloaded) or
// loaded. A port in use fails at once, not after a 27B's ~50 s load, and
// until the model is ready every route answers 503 {"status": "loading"}
// (as llama-server does) except the web chat page, which waits for it.
struct Listener {
  std::unique_ptr<http::Server> server;
  std::thread thread;
  http::Options opts;
  ~Listener() { stop(); }
  // Stops the server and waits for its handlers (before what they use goes).
  void stop() {
    if (!thread.joinable()) return;
    server->stop();
    thread.join();
  }
};

void open_port(Listener& L, const Args& a) {
  http::Options& ho = L.opts;
  ho.host = a.get("host", "127.0.0.1");
  ho.port = a.i("port", 8080);
  ho.api_key = a.get("api-key");
  if (ho.api_key.empty())  // the environment keeps the key out of `ps`
    if (const char* k = std::getenv("ESPETERO_API_KEY")) ho.api_key = k;
  ho.cors = a.has("cors");
  ho.log_requests = a.has("verbose");
  try {
    L.server = std::make_unique<http::Server>(ho);
  } catch (const http::ListenError& e) {
    if (e.error == EADDRINUSE)
      throw std::runtime_error("port " + std::to_string(ho.port) + " is in use (another program listens on it): pick "
                               "another with --port");
    throw;
  }
  install_loading_routes(*L.server, !a.has("no-webui"));
  L.thread = std::thread([&L] { L.server->run(); });
  std::fprintf(stderr, "[espetero] listening on http://%s:%d (requests get 503 until the model is ready)\n",
               ho.host.find(':') != std::string::npos ? ("[" + ho.host + "]").c_str() : ho.host.c_str(), L.server->port());
  if (ho.api_key.empty() && !http::is_loopback_host(ho.host))
    std::fprintf(stderr,
                 "[espetero] warning: listening on %s without --api-key: any client that can reach this port can use "
                 "the server (only web pages are held back: requests must be addressed to localhost, 127.0.0.1, [::1] "
                 "or %s); set --api-key or ESPETERO_API_KEY\n",
                 ho.host.c_str(), ho.host.c_str());
  if (ho.api_key.empty() && ho.cors)
    std::fprintf(stderr,
                 "[espetero] warning: --cors without --api-key: any web page open in a browser on this machine can use "
                 "the server\n");
}

int cmd_serve(const Args& a, Listener& L) {
  EngineOptions o = engine_options(a);
  std::fprintf(stderr, "[espetero] loading %s ...\n", o.model_path.c_str());
  Engine engine(o);
  const ModelInfo& I = engine.info();
  std::fprintf(stderr, "[espetero] %s on %s: %d layers, vocab %d, %d slots, %d KV tokens, vision %s, mtp %s (%.1f s)\n",
               engine.model_name().c_str(), I.device.c_str(), I.layers, I.vocab, I.slots, I.kv_blocks * I.block_tokens,
               I.vision ? "yes" : "no", I.mtp ? "yes" : "no", I.load_ms / 1000);
  const std::string dir = a.get("data-dir", paths::user_cache_dir() + "/batches");
  paths::make_private_dirs(dir);
  BatchService batches(engine, dir, a.i("batch-concurrency", o.max_seqs), a.d("batch-disk-gb", 4));
  struct StopFirst {  // however this returns, the handlers end before the engine and the batches they use
    Listener& L;
    ~StopFirst() { L.stop(); }
  } stop_first{L};
  http::Server& server = *L.server;
  install_routes(server, engine, &batches, &g_config);
  if (::pipe(g_stop_pipe) != 0) throw std::runtime_error("cannot create the shutdown pipe");
  ::fcntl(g_stop_pipe[1], F_SETFL, O_NONBLOCK);
  std::thread watcher([&] {
    char c = 0;
    while (::read(g_stop_pipe[0], &c, 1) < 0 && errno == EINTR) {
    }
    if (c != 's') return;  // the server stopped on its own
    std::fprintf(stderr, "[espetero] stopping: queued and running requests end after the current pass "
                         "(a second signal exits at once)\n");
    server.stop();
    batches.shutdown();
    engine.shutdown();
  });
  struct Join {  // the watcher, if no signal woke it (also when run() throws)
    std::thread& t;
    ~Join() {
      const char q = 'q';
      (void)!::write(g_stop_pipe[1], &q, 1);
      t.join();
    }
  } join{watcher};
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  server.open();
  std::fprintf(stderr, "[espetero] ready: http://%s:%d\n",
               L.opts.host.find(':') != std::string::npos ? ("[" + L.opts.host + "]").c_str() : L.opts.host.c_str(),
               server.port());
  L.thread.join();  // run() returns once the watcher stopped the server
  std::fprintf(stderr, "[espetero] stopped\n");
  return 0;
}

int cmd_batch(const Args& a) {
  EngineOptions o = engine_options(a);
  const std::string in = a.get("input"), out = a.get("output");
  if (in.empty() || out.empty()) {
    usage();
    return 2;
  }
  Engine engine(o);
  BatchService svc(engine, paths::user_cache_dir() + "/batches",
                   a.i("batch-concurrency", o.max_seqs));
  const auto t0 = std::chrono::steady_clock::now();
  auto [ok, bad] = svc.run_file(in, out, a.get("errors"), a.get("endpoint", "/v1/chat/completions"));
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  json::Value st = engine.stats();
  std::fprintf(stderr, "[espetero] %d completed, %d failed in %.1f s; %lld tokens generated (%.1f tokens/s)\n", ok, bad, s,
               (long long)st.get_int("generated_tokens", 0), double(st.get_int("generated_tokens", 0)) / s);
  return bad ? 1 : 0;
}

int cmd_chat(const Args& a) {
  EngineOptions o = engine_options(a);
  Engine engine(o);
  std::fprintf(stderr, "[espetero] %s ready. Empty line = new conversation, Ctrl-D = quit.\n", engine.model_name().c_str());
  json::Value messages = json::Value::array();
  if (a.has("system")) {
    json::Value m = json::Value::object();
    m.set("role", "system");
    m.set("content", a.get("system"));
    messages.push_back(m);
  }
  std::string line;
  while (true) {
    std::fprintf(stdout, "\n> ");
    std::fflush(stdout);
    if (!std::getline(std::cin, line)) break;
    if (line.empty()) {
      messages = json::Value::array();
      std::fprintf(stdout, "(new conversation)\n");
      continue;
    }
    json::Value m = json::Value::object();
    m.set("role", "user");
    m.set("content", line);
    messages.push_back(m);
    json::Value body = json::Value::object();
    body.set("messages", messages);
    body.set("max_tokens", a.i("max-tokens", 2048));
    // The model's sampling defaults (generation_config.json) unless given
    // here; without them (or with --no-generation-config) 0.7 / 0.95 / 20.
    const SamplingDefaults& sd = engine.sampling_defaults();
    if (a.has("temperature") || !sd.temperature) body.set("temperature", a.d("temperature", 0.7));
    if (a.has("top-p") || !sd.top_p) body.set("top_p", a.d("top-p", 0.95));
    if (a.has("top-k") || !sd.top_k) body.set("top_k", a.i("top-k", 20));
    if (a.has("think")) body.set("enable_thinking", true);
    try {
      ChatJob job = prepare_chat(engine, body);
      std::string content, reasoning;
      ChatEvent last;
      bool in_reasoning = false;
      run_chat(engine, job, [&](const ChatEvent& ev) {
        if (!ev.reasoning.empty()) {
          if (!in_reasoning) std::fprintf(stdout, "\x1b[2m");
          in_reasoning = true;
          std::fwrite(ev.reasoning.data(), 1, ev.reasoning.size(), stdout);
        }
        if (!ev.content.empty()) {
          if (in_reasoning) std::fprintf(stdout, "\x1b[0m\n");
          in_reasoning = false;
          std::fwrite(ev.content.data(), 1, ev.content.size(), stdout);
        }
        std::fflush(stdout);
        content += ev.content;
        reasoning += ev.reasoning;
        if (ev.final) last = ev;
        return true;
      });
      if (in_reasoning) std::fprintf(stdout, "\x1b[0m");
      const json::Value& t = last.timings;
      std::fprintf(stderr, "\n\x1b[2m[%lld tokens, prefill %.0f ms, %.1f tokens/s%s]\x1b[0m\n",
                   (long long)last.usage.get_int("completion_tokens", 0), t.get_double("prefill_ms", 0),
                   t.get_double("decode_tokens_per_s", 0),
                   t.get("draft_acceptance") ? (", drafts " + std::to_string(int(t.get_double("draft_acceptance", 0) * 100)) + "%").c_str() : "");
      json::Value reply = json::Value::object();
      reply.set("role", "assistant");
      reply.set("content", content);
      messages.push_back(reply);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      messages.as_array().pop_back();
    }
  }
  return 0;
}

// Espetero service install|uninstall|status|logs (service.h).
void usage_service() {
  std::fprintf(stderr,
               "espetero service: run the server in the background (a LaunchAgent of this user)\n\n"
               "  espetero service install [--config FILE] [serve options, e.g. --model ORG/NAME --port 8080]\n"
               "                        save the options in the config file (default ~/.config/espetero/config.json),\n"
               "                        then run `espetero serve --config FILE` now, at every login and again if it\n"
               "                        fails (installing again restarts it with the file's settings)\n"
               "  espetero service status      installed, running, and what GET /health answers\n"
               "  espetero service logs [--follow] [--lines N]   ~/Library/Logs/espetero.log\n"
               "  espetero service uninstall   stop it and remove the LaunchAgent (the config file and the log stay)\n"
               "  --dry-run             (install, uninstall) show what would be written and run, change nothing\n");
}

// The path this binary was started by (a symlink such as Homebrew's bin/espetero
// is kept, so the agent survives upgrades), made absolute.
std::string executable_path() {
  uint32_t n = 0;
  _NSGetExecutablePath(nullptr, &n);
  std::string p(n, '\0');
  if (_NSGetExecutablePath(p.data(), &n) != 0) return "espetero";
  p.resize(std::strlen(p.c_str()));
  const size_t slash = p.rfind('/');
  const std::string dir = slash == std::string::npos ? "." : p.substr(0, slash);
  char real[PATH_MAX];
  if (::realpath(dir.c_str(), real)) p = std::string(real) + "/" + p.substr(slash == std::string::npos ? 0 : slash + 1);
  return p;
}

int cmd_service(const Args& a) {
  const std::string action = a.positional.empty() ? "" : a.positional[0];
  if (action != "install" && action != "uninstall" && action != "status" && action != "logs") {
    usage_service();
    return action.empty() || action == "help" ? 0 : 2;
  }
  service::Setup s;
  s.home = paths::home();
  s.uid = unsigned(::getuid());
  s.dry_run = a.has("dry-run");
  s.binary = executable_path();
  s.config = a.has("config") ? a.get("config") : service::default_config(s.home);
  if (s.config == "none" || s.config.empty()) config_error("service: the service runs with a config file (--config FILE)");
  if (s.config[0] != '/') {
    char cwd[PATH_MAX];
    if (::getcwd(cwd, sizeof cwd)) s.config = std::string(cwd) + "/" + s.config;
  }
  for (const char* k : {"PATH", "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE"})
    if (const char* v = std::getenv(k); v && *v) s.env.push_back({k, v});
  if (action == "install") {
    // The agent runs in the home directory: relative paths that exist here
    // are saved as absolute ones.
    char cwd[PATH_MAX];
    const bool have_cwd = ::getcwd(cwd, sizeof cwd) != nullptr;
    std::vector<std::pair<std::string, std::string>> settings;
    for (auto [k, v] : a.kv) {
      if (k == "config" || k == "dry-run") continue;
      const bool path_option = k == "model" || k == "mmproj" || k == "mtp" || k == "data-dir" || k == "prefix-dir" ||
                               k == "expert-store";
      if (path_option && have_cwd && !v.empty() && v[0] != '/' && ::access(v.c_str(), F_OK) == 0)
        v = std::string(cwd) + "/" + v;
      settings.push_back({k, v});
    }
    return service::install(s, settings, a.bare, stdout);
  }
  if (action == "uninstall") return service::uninstall(s, stdout);
  if (action == "status") return service::status(s, stdout);
  bool follow = a.has("follow");
  for (const std::string& x : a.positional) follow |= x == "-f";
  return service::logs(s, follow, a.i("lines", 50));
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);
  Args cli = parse(argc, argv);
  if (cli.has("version") || cli.cmd == "--version" || cli.cmd == "version") {
    std::printf("%s %s\n", kProduct, kVersion);
    return 0;
  }
  if (cli.cmd == "service" && cli.has("help")) {
    usage_service();
    return 0;
  }
  if (cli.has("help-all") || cli.cmd == "--help-all" || cli.cmd == "help-all") {
    usage_all();
    return 0;
  }
  if (cli.has("help") || cli.cmd == "--help" || cli.cmd == "help") {
    usage();
    return 0;
  }
  try {
    if (cli.cmd == "service") return cmd_service(cli);
    if (cli.cmd == "recommend" || (cli.cmd == "models" && cli.has("recommend"))) return cmd_recommend(resolve_settings(cli).args);
    if (cli.cmd == "pull" || cli.cmd == "list" || cli.cmd == "models" || cli.cmd == "rm") {
      Args a = resolve_settings(cli).args;
      switches(a, {"yes", "skip-memory-check", "offline"});
      if (a.cmd == "pull") return cmd_pull(a);
      if (a.cmd == "rm") return cmd_rm(a);
      return cmd_list(a);
    }
    if (cli.cmd == "serve" || cli.cmd == "chat" || cli.cmd == "batch" || cli.cmd == "info") {
      Resolved r = resolve_settings(cli);
      Args& a = r.args;
      Listener listener;  // serve's port, open before the model is downloaded or loaded
      if (a.cmd == "serve" && !cli.has("print-config")) {
        (void)engine_options(a);  // invalid options exit here, before a server thread runs
        open_port(listener, a);
      }
      const std::string repo = resolve_model(a, cli.has("print-config"));
      g_config = r.report;
      g_config.set("effective", effective_json(engine_options(a), repo));
      if (cli.has("print-config")) {
        std::printf("%s\n", json::dump(g_config, json::DumpOptions{", ", ": ", false, 2, false}).c_str());
        return 0;
      }
      check_memory(a, repo);
      if (a.cmd == "serve") return cmd_serve(a, listener);
      if (a.cmd == "chat") return cmd_chat(a);
      if (a.cmd == "batch") return cmd_batch(a);
      if (a.cmd == "info") return cmd_info(a);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  usage();
  return cli.cmd.empty() ? 0 : 2;
}
