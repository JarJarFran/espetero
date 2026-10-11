// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Tokenizer::from_vocab (llama.cpp-style vocabularies) and Tokenizer::from_gguf,
// with a minimal GGUF metadata reader: the file is mmapped and only the header
// and key/value section are touched, never tensor data.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

#include "tok_internal.h"
#include "tokenizer.h"

namespace espetero {

namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("tokenizer: " + msg); }

// GGUF value types.
enum GgufType : uint32_t {
  kU8 = 0, kI8 = 1, kU16 = 2, kI16 = 3, kU32 = 4, kI32 = 5, kF32 = 6, kBool = 7,
  kString = 8, kArray = 9, kU64 = 10, kI64 = 11, kF64 = 12,
};

size_t scalar_size(uint32_t t) {
  switch (t) {
    case kU8: case kI8: case kBool: return 1;
    case kU16: case kI16: return 2;
    case kU32: case kI32: case kF32: return 4;
    case kU64: case kI64: case kF64: return 8;
    default: return 0;
  }
}

// Bounds-checked little-endian reader over the mapped file.
class Reader {
 public:
  Reader(const unsigned char* p, size_t n, const std::string& path) : p_(p), e_(p + n), path_(path) {}
  template <class T>
  T get() {
    need(sizeof(T));
    T v;
    std::memcpy(&v, p_, sizeof(T));
    p_ += sizeof(T);
    return v;
  }
  std::string_view str() {
    const uint64_t n = get<uint64_t>();
    need(n);
    std::string_view s(reinterpret_cast<const char*>(p_), size_t(n));
    p_ += n;
    return s;
  }
  int64_t integer(uint32_t t) {
    switch (t) {
      case kU8: return get<uint8_t>();
      case kI8: return get<int8_t>();
      case kU16: return get<uint16_t>();
      case kI16: return get<int16_t>();
      case kU32: return get<uint32_t>();
      case kI32: return get<int32_t>();
      case kU64: return int64_t(get<uint64_t>());
      case kI64: return get<int64_t>();
      default: fail(path_ + ": expected an integer GGUF value");
    }
  }
  void skip(uint32_t t) {
    if (t == kString) {
      str();
    } else if (t == kArray) {
      const uint32_t et = get<uint32_t>();
      const uint64_t n = get<uint64_t>();
      if (const size_t sz = scalar_size(et)) {
        if (n > uint64_t(e_ - p_) / sz) fail(path_ + ": truncated GGUF array");
        p_ += n * sz;
      } else {
        for (uint64_t i = 0; i < n; ++i) skip(et);
      }
    } else if (const size_t sz = scalar_size(t)) {
      need(sz);
      p_ += sz;
    } else {
      fail(path_ + ": unknown GGUF value type " + std::to_string(t));
    }
  }
  // Array header: element type and count.
  uint64_t array(uint32_t t, uint32_t& elem) {
    if (t != kArray) fail(path_ + ": expected a GGUF array");
    elem = get<uint32_t>();
    return get<uint64_t>();
  }

 private:
  void need(uint64_t n) {
    if (n > uint64_t(e_ - p_)) fail(path_ + ": truncated GGUF metadata");
  }
  const unsigned char* p_;
  const unsigned char* e_;
  const std::string& path_;
};

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::from_vocab(std::vector<std::string> tokens, std::vector<int32_t> types,
                                                 const std::vector<std::string>& merges, const std::string& pre) {
  const auto pattern = tok::split_pattern_from_gguf_pre(pre);
  if (!pattern)
    fail("unsupported tokenizer.ggml.pre '" + pre + "' (supported: qwen35, qwen2, deepseek-r1-qwen, llama3, llama-bpe)");
  if (!types.empty() && types.size() != tokens.size()) fail("token_type count does not match the token count");

  tok::BuildInput in;
  in.pattern = *pattern;
  // Match the reference tokenizer.json of these families: Qwen normalizes to NFC
  // and merges strictly; Llama-3 has no normalizer and uses ignore_merges.
  in.nfc = *pattern != tok::SplitPattern::kLlama3;
  in.ignore_merges = *pattern == tok::SplitPattern::kLlama3;
  size_t total = 0;
  for (const auto& t : tokens) total += t.size();
  in.arena.reserve(total);
  in.vocab.reserve(tokens.size());
  for (size_t id = 0; id < tokens.size(); ++id) {
    const std::string& text = tokens[id];
    const int32_t type = types.empty() ? 1 : types[id];
    // Rows added to round the embedding matrix up: they keep their ids, but are not tokens.
    if (type != 1 && text.rfind("[PAD", 0) == 0) continue;
    if (text.empty()) continue;
    if (type == 3 || type == 4 || type == 2) {  // control / user-defined / unknown -> added tokens
      tok::AddedToken a;
      a.content = text;
      a.id = int32_t(id);
      a.special = type != 4;
      in.added.push_back(std::move(a));
      continue;
    }
    in.vocab.push_back({{uint32_t(in.arena.size()), uint32_t(text.size())}, int32_t(id)});
    in.arena += text;
  }
  tokens.clear();
  tokens.shrink_to_fit();

  size_t mtotal = 0;
  for (const auto& m : merges) mtotal += m.size();
  in.merge_arena.reserve(mtotal);
  in.merges.reserve(merges.size());
  for (size_t r = 0; r < merges.size(); ++r) {
    const std::string& m = merges[r];
    const size_t sp = m.find(' ');
    if (sp == std::string::npos || m.find(' ', sp + 1) != std::string::npos)
      fail("bad merge #" + std::to_string(r) + ": '" + m + "'");
    const uint32_t off = uint32_t(in.merge_arena.size());
    in.merge_arena += m;
    in.merges.push_back({{off, uint32_t(sp)}, {off + uint32_t(sp) + 1, uint32_t(m.size() - sp - 1)}});
  }
  return std::unique_ptr<Tokenizer>(new Tokenizer(tok::build(std::move(in))));
}

// Gemma 4's GGUF vocabulary: SentencePiece-style BPE with byte
// fallback (tokenizer.ggml.model "gemma4": the pipeline of its tokenizer.json,
// see Impl::spm). Control tokens and the end-of-sequence token become special added
// tokens (<eos> too: a normal token in these files), user-defined ones
// non-special added tokens; byte tokens (type 6)
// are the byte-fallback symbols.
std::unique_ptr<Tokenizer> Tokenizer::from_spm_vocab(std::vector<std::string> tokens, std::vector<int32_t> types,
                                                     const std::vector<std::string>& merges, int64_t eos) {
  if (!types.empty() && types.size() != tokens.size()) fail("token_type count does not match the token count");
  tok::BuildInput in;
  in.spm = true;
  in.nfc = false;
  in.unk_token = "<unk>";
  size_t total = 0;
  for (const auto& t : tokens) total += t.size();
  in.arena.reserve(total);
  in.vocab.reserve(tokens.size());
  for (size_t id = 0; id < tokens.size(); ++id) {
    const std::string& text = tokens[id];
    const int32_t type = types.empty() ? 1 : types[id];
    if (text.empty()) continue;
    // (<eos> is a normal token in these files: special as in tokenizer.json.)
    if (type == 3 || type == 4 || type == 2 || int64_t(id) == eos || text == "<eos>") {
      tok::AddedToken a;
      a.content = text;
      a.id = int32_t(id);
      a.special = type != 4;
      in.added.push_back(std::move(a));
      continue;
    }
    in.vocab.push_back({{uint32_t(in.arena.size()), uint32_t(text.size())}, int32_t(id)});
    in.arena += text;
  }
  tokens.clear();
  tokens.shrink_to_fit();
  size_t mtotal = 0;
  for (const auto& m : merges) mtotal += m.size();
  in.merge_arena.reserve(mtotal);
  in.merges.reserve(merges.size());
  for (size_t r = 0; r < merges.size(); ++r) {
    const std::string& m = merges[r];
    const size_t sp = m.find(' ');
    if (sp == std::string::npos || m.find(' ', sp + 1) != std::string::npos)
      fail("bad merge #" + std::to_string(r) + ": '" + m + "'");
    const uint32_t off = uint32_t(in.merge_arena.size());
    in.merge_arena += m;
    in.merges.push_back({{off, uint32_t(sp)}, {off + uint32_t(sp) + 1, uint32_t(m.size() - sp - 1)}});
  }
  return std::unique_ptr<Tokenizer>(new Tokenizer(tok::build(std::move(in))));
}

std::unique_ptr<Tokenizer> Tokenizer::from_gguf(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) fail("cannot open " + path);
  struct stat st {};
  if (::fstat(fd, &st) != 0 || st.st_size < 24) {
    ::close(fd);
    fail(path + ": not a GGUF file");
  }
  const size_t size = size_t(st.st_size);
  void* base = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (base == MAP_FAILED) fail("cannot mmap " + path);
  struct Unmap {
    void* p;
    size_t n;
    ~Unmap() { ::munmap(p, n); }
  } unmap{base, size};

  Reader r(static_cast<const unsigned char*>(base), size, path);
  if (std::memcmp(base, "GGUF", 4) != 0) fail(path + ": not a GGUF file");
  r.get<uint32_t>();  // magic
  const uint32_t version = r.get<uint32_t>();
  if (version < 2) fail(path + ": GGUF version " + std::to_string(version) + " is not supported");
  r.get<uint64_t>();  // tensor count
  const uint64_t n_kv = r.get<uint64_t>();

  std::string model, pre, chat_template;
  std::vector<std::string> tokens, merges;
  std::vector<int32_t> types;
  int64_t eos = -1, bos = -1, pad = -1;
  int add_bos = -1;  // -1: the file does not say
  for (uint64_t k = 0; k < n_kv; ++k) {
    const std::string_view key = r.str();
    const uint32_t t = r.get<uint32_t>();
    uint32_t et = 0;
    if (key == "tokenizer.ggml.model" && t == kString) {
      model = r.str();
    } else if (key == "tokenizer.ggml.pre" && t == kString) {
      pre = r.str();
    } else if (key == "tokenizer.chat_template" && t == kString) {
      chat_template = r.str();
    } else if (key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges") {
      const uint64_t n = r.array(t, et);
      if (et != kString) fail(path + ": " + std::string(key) + " must be an array of strings");
      auto& dst = key == "tokenizer.ggml.tokens" ? tokens : merges;
      dst.reserve(size_t(n));
      for (uint64_t i = 0; i < n; ++i) dst.emplace_back(r.str());
    } else if (key == "tokenizer.ggml.token_type") {
      const uint64_t n = r.array(t, et);
      types.reserve(size_t(n));
      for (uint64_t i = 0; i < n; ++i) types.push_back(int32_t(r.integer(et)));
    } else if (key == "tokenizer.ggml.eos_token_id" && scalar_size(t)) {
      eos = r.integer(t);
    } else if (key == "tokenizer.ggml.bos_token_id" && scalar_size(t)) {
      bos = r.integer(t);
    } else if (key == "tokenizer.ggml.padding_token_id" && scalar_size(t)) {
      pad = r.integer(t);
    } else if (key == "tokenizer.ggml.add_bos_token" && t == kBool) {
      add_bos = r.get<uint8_t>() != 0;
    } else {
      r.skip(t);
    }
  }
  if (model != "gpt2" && model != "gemma4")
    fail(path + ": tokenizer.ggml.model is '" + model + "', only 'gpt2' (byte-level BPE) and 'gemma4' are supported");
  if (tokens.empty()) fail(path + ": no tokenizer.ggml.tokens");

  auto tk = model == "gemma4" ? from_spm_vocab(std::move(tokens), std::move(types), merges, eos)
                              : from_vocab(std::move(tokens), std::move(types), merges, pre);
  Impl& im = *tk->impl_;
  auto valid = [&](int64_t id) { return id >= 0 && id < im.n_vocab && im.kind[size_t(id)] != Impl::kNone ? int32_t(id) : -1; };
  im.eos = valid(eos);
  im.bos = valid(bos);
  im.pad = valid(pad);
  im.add_bos = add_bos >= 0 ? add_bos == 1 : model == "gemma4";  // llama.cpp's default for this vocabulary
  im.chat_template = std::move(chat_template);
  return tk;
}

}  // namespace espetero
