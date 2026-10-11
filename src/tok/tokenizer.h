// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Byte-level BPE tokenizer, token-for-token identical to Hugging Face
// `tokenizers` for the Qwen family (Qwen2/Qwen3/Qwen3.5/Qwen3.8 tokenizer.json)
// and for the same vocabularies stored in GGUF files.
//
// Pipeline (as in `tokenizers`): split out added tokens (leftmost-longest) ->
// NFC on the remaining text -> pre-tokenizer regex split (hand-written, see
// pretok.h) -> byte-level mapping -> BPE merges by rank.
//
// Loading throws std::runtime_error with a readable message on unsupported or
// malformed input. After loading, every method is const and thread-safe (the
// internal word cache is sharded and locked).
//
// Deliberate differences from the reference, none of which affect valid input:
//  * Input need not be valid UTF-8: each invalid byte is treated like a symbol
//    character and encoded as its byte token, so decode() gives it back.
//  * decode() returns raw bytes; the reference returns a Python str where
//    invalid UTF-8 became U+FFFD. Apply a lossy conversion when showing text.
//  * The post-processor is ignored (for these models it adds nothing), so
//    encode() never inserts BOS/EOS; use bos_id()/eos_id() explicitly.
//  * GGUF files carry token types, not the HF "special" flags: llama.cpp's
//    converter marks tokens that look special (<|fim_prefix|>, <|repo_name|>,
//    <|file_sep|>, ...) as control tokens, so for those is_special(), skip_special
//    and parse_special=false follow the GGUF, not tokenizer.json. Default
//    encode() output is identical.
//  * single_word added tokens (unused by Qwen) approximate the reference's \w
//    test with the letter/mark/number classes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace espetero {

class Tokenizer {
 public:
  // HF directory: tokenizer.json (+ tokenizer_config.json when present: added-token flags,
  // eos/bos/pad names, chat_template string or chat_template.jinja file next to it).
  static std::unique_ptr<Tokenizer> from_hf_dir(const std::string& dir);
  // GGUF (llama.cpp) vocabularies: tokenizer.ggml.model must be "gpt2"; `pre` is tokenizer.ggml.pre
  // ("qwen35", "qwen2", ...); token types as in GGUF (1 normal, 3 control, 4 user-defined, ...).
  // Control tokens become special added tokens, user-defined ones non-special added tokens;
  // "[PAD..." slots of other types keep their ids but are not tokens.
  static std::unique_ptr<Tokenizer> from_vocab(std::vector<std::string> tokens, std::vector<int32_t> types,
                                               const std::vector<std::string>& merges /* "a b" */,
                                               const std::string& pre);
  // Gemma 4 GGUF vocabularies (tokenizer.ggml.model "gemma4"): SentencePiece-style BPE with
  // byte fallback; control tokens and `eos` become special added tokens, user-defined ones
  // non-special added tokens.
  static std::unique_ptr<Tokenizer> from_spm_vocab(std::vector<std::string> tokens, std::vector<int32_t> types,
                                                   const std::vector<std::string>& merges, int64_t eos);
  // Reads the tokenizer.* keys of a .gguf file (mmap; only metadata, never tensors) and calls
  // from_vocab (from_spm_vocab for Gemma 4); also fills chat_template(), eos/bos/pad ids.
  static std::unique_ptr<Tokenizer> from_gguf(const std::string& path);

  ~Tokenizer();
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  // parse_special=true (the reference default) splits every added token found in the text;
  // false treats special added tokens as plain text (non-special ones are still split), like
  // `encode_special_tokens=True` in `tokenizers`.
  std::vector<int32_t> encode(std::string_view text, bool parse_special = true) const;
  void encode_append(std::string_view text, std::vector<int32_t>& out, bool parse_special = true) const;

  // Concatenated raw bytes of the tokens (unknown ids contribute nothing).
  std::string decode(const int32_t* ids, size_t n, bool skip_special = false) const;
  std::string decode(const std::vector<int32_t>& ids, bool skip_special = false) const;
  // Raw bytes a token contributes to decoded text (added tokens: their content). For
  // streaming; may be an incomplete UTF-8 sequence. Empty for unknown ids.
  const std::string& piece(int32_t id) const;

  int32_t token_id(std::string_view token_text) const;  // exact vocab/added-token lookup; -1 if missing
  bool is_special(int32_t id) const;                   // added token with special=true
  bool is_added(int32_t id) const;                     // any added token (special or not)
  // Token text as stored in the vocabulary (byte-level form, e.g. "Ġthe"; added tokens:
  // their content). Empty for unknown ids.
  std::string_view token_text(int32_t id) const;
  int32_t vocab_size() const;                          // max id + 1
  int32_t eos_id() const;
  int32_t bos_id() const;
  int32_t pad_id() const;  // -1 if unknown
  // The model's files ask for BOS before raw text (llama.cpp's add_bos): see
  // with_bos(). Chat templates write their own.
  bool add_bos() const;
  // ids with BOS in front when add_bos() and it is not there yet.
  std::vector<int32_t> with_bos(std::vector<int32_t> ids) const;
  const std::string& chat_template() const;            // "" if none
  std::vector<int32_t> special_ids() const;
  std::string_view pre_tokenizer_name() const;         // "qwen35", "qwen2" or "llama3"

  struct Impl;

 private:
  explicit Tokenizer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace espetero
