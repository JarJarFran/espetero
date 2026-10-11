// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Format-neutral view of checkpoint tensors. Loaders ask a WeightSource for
// tensors by canonical (MLX-style) names; each source maps those to its own
// layout (MLX safetensors, GGUF, GGUF mmproj) and exposes the raw storage.
#pragma once

#include "../../util/json.h"
#include "common.h"

namespace espetero::native {

struct TensorView {
  enum Kind { Dense, MLXQuant, GGUFQuant };
  std::string name;              // source-side name, for messages
  Kind kind = Dense;
  std::vector<int64_t> shape;    // logical, row-major: matrices are [rows = out, cols = in]
  std::string dtype;             // Dense: F32, F16 or BF16 (MLXQuant: scale dtype)
  const uint8_t* data = nullptr;
  size_t nbytes = 0;
  int bits = 0, group = 0;       // MLXQuant: code width and group size along cols
  const uint8_t* scales = nullptr;  // MLXQuant: [rows, cols / group]
  const uint8_t* biases = nullptr;
  int gguf_type = -1;            // GGUFQuant: ggml type id

  int64_t rows() const { return shape.empty() ? 0 : (shape.size() == 1 ? 1 : shape[0]); }
  int64_t cols() const { return shape.empty() ? 0 : shape.back(); }
  int64_t numel() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  }
};

struct WeightSource {
  virtual ~WeightSource() = default;
  virtual std::string describe() const = 0;
  virtual bool has(const std::string& name) const = 0;
  virtual TensorView tensor(const std::string& name) const = 0;

  // A dense tensor as fp32 (norm weights, gates, biases, conv taps, ...).
  std::vector<float> floats(const std::string& name, size_t expect) const {
    const TensorView t = tensor(name);
    if (t.kind != TensorView::Dense) fail(name + ": expected a dense tensor");
    const size_t n = size_t(t.numel());
    if (expect && n != expect)
      fail(name + " has " + std::to_string(n) + " elements, expected " + std::to_string(expect));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = dense_at(t.data, t.dtype, i);
    return out;
  }
};

// ---------------------------------------------------------------- safetensors
// MLX-converted checkpoints (mlx-lm / mlx-vlm layout): affine quantized
// matrices (weight U32 + scales/biases, per-layer widths from config.json)
// and dense bf16/fp16/fp32 tensors.
class SafetensorsSource : public WeightSource {
 public:
  explicit SafetensorsSource(const std::string& dir) : dir_(dir) {
    const std::string index = dir + "/model.safetensors.index.json";
    if (file_exists(index)) {
      json::Value idx = json::parse(read_file(index));
      std::vector<std::string> files;
      if (const json::Value* wm = idx.get("weight_map")) {
        for (const json::Value& f : wm->as_object().vals) {
          const std::string& name = f.as_string();
          if (std::find(files.begin(), files.end(), name) == files.end()) files.push_back(name);
        }
      }
      for (const auto& f : files) add_file(dir + "/" + f);
    } else {
      add_file(dir + "/model.safetensors");
    }
    config_ = json::parse(read_file(dir + "/config.json"));
    const json::Value* qz = config_.get("quantization");
    if (!qz || !qz->is_object()) qz = config_.get("quantization_config");
    if (qz && qz->is_object()) {
      default_bits_ = int(qz->get_int("bits", 4));
      default_group_ = int(qz->get_int("group_size", 64));
      const json::Object& o = qz->as_object();
      for (size_t i = 0; i < o.size(); ++i) {
        const json::Value& v = o.vals[i];
        if (!v.is_object()) continue;
        overrides_[o.keys[i]] = {int(v.get_int("bits", default_bits_)), int(v.get_int("group_size", default_group_))};
      }
    }
  }

  const json::Value& config() const { return config_; }
  const std::string& dir() const { return dir_; }
  std::string describe() const override { return "mlx:" + dir_; }
  bool has(const std::string& name) const override { return entries_.count(name) != 0; }

  TensorView tensor(const std::string& name) const override {
    auto it = entries_.find(name);
    if (it == entries_.end()) fail("missing tensor " + name);
    const Entry& e = it->second;
    TensorView t;
    t.name = name;
    t.shape = e.shape;
    t.data = files_[size_t(e.file)]->at(e.offset);
    t.nbytes = e.nbytes;
    t.dtype = e.dtype;
    if (e.dtype == "U32" && ends_with(name, ".weight")) {
      const std::string base = name.substr(0, name.size() - 7);
      auto s = entries_.find(base + ".scales");
      auto z = entries_.find(base + ".biases");
      if (s == entries_.end() || z == entries_.end()) fail(base + ": quantized weight without scales/biases");
      auto ov = overrides_.find(base);
      const int bits = ov != overrides_.end() ? ov->second.first : default_bits_;
      const int group = ov != overrides_.end() ? ov->second.second : default_group_;
      // Rank 2, or rank 3 for stacked experts ([E, out, in] read as [E * out, in]).
      if (e.shape.size() != 2 && e.shape.size() != 3) fail(base + ": unexpected quantized rank");
      int64_t rows = 1;
      for (size_t i = 0; i + 1 < e.shape.size(); ++i) rows *= e.shape[i];
      const int64_t in = e.shape.back() * 32 / bits;
      if (e.shape.back() * 32 % bits || in % group || s->second.shape.back() * group != in)
        fail(base + ": quantized shape does not match its width and group");
      if (s->second.dtype != "BF16" && s->second.dtype != "F16")
        fail(base + ": unsupported scale dtype " + s->second.dtype);
      t.kind = TensorView::MLXQuant;
      t.shape = {rows, in};
      t.bits = bits;
      t.group = group;
      t.scales = files_[size_t(s->second.file)]->at(s->second.offset);
      t.biases = files_[size_t(z->second.file)]->at(z->second.offset);
      t.dtype = s->second.dtype;
    } else if (e.dtype != "F32" && e.dtype != "F16" && e.dtype != "BF16") {
      fail(name + ": unsupported dtype " + e.dtype);
    }
    return t;
  }

  // Where a tensor's bytes are, for data read later with pread (the n-gram
  // table and the routed experts of models larger than memory) and for raw
  // tensors of other dtypes (I64 hashing constants).
  struct Location {
    std::string file, dtype;
    std::vector<int64_t> shape;
    uint64_t offset = 0, nbytes = 0;
    const uint8_t* data = nullptr;  // mapped view
  };
  Location locate(const std::string& name) const {
    auto it = entries_.find(name);
    if (it == entries_.end()) fail("missing tensor " + name);
    const Entry& e = it->second;
    Location l;
    l.file = files_[size_t(e.file)]->path;
    l.dtype = e.dtype;
    l.shape = e.shape;
    l.offset = e.offset;
    l.nbytes = e.nbytes;
    l.data = files_[size_t(e.file)]->at(e.offset);
    return l;
  }
  // Quantization (bits, group) the config gives a matrix (`base` without .weight).
  std::pair<int, int> quant_of(const std::string& base) const {
    auto ov = overrides_.find(base);
    return ov != overrides_.end() ? ov->second : std::make_pair(default_bits_, default_group_);
  }

 private:
  struct Entry {
    std::string dtype;
    std::vector<int64_t> shape;
    size_t offset = 0, nbytes = 0;
    int file = 0;
  };

  // Bytes per element of the safetensors dtypes (0: unknown, not checked).
  static size_t dtype_bytes(const std::string& t) {
    if (t == "F64" || t == "I64" || t == "U64") return 8;
    if (t == "F32" || t == "I32" || t == "U32") return 4;
    if (t == "F16" || t == "BF16" || t == "I16" || t == "U16") return 2;
    if (t == "I8" || t == "U8" || t == "BOOL" || t == "F8_E4M3" || t == "F8_E5M2") return 1;
    return 0;
  }

  // The header is checked as the safetensors library checks it (offsets inside
  // the data, elements x dtype size = bytes) with overflow-safe arithmetic: a
  // crafted file fails to load instead of pointing reads outside the mapping.
  void add_file(const std::string& path) {
    auto mf = std::make_unique<MappedFile>(path);
    if (mf->size < 8) fail(path + ": not a safetensors file");
    uint64_t hlen = 0;
    std::memcpy(&hlen, mf->at(0), 8);
    if (hlen > mf->size - 8) fail(path + ": truncated safetensors header");
    json::Value header = json::parse(std::string_view(reinterpret_cast<const char*>(mf->at(8)), size_t(hlen)));
    if (!header.is_object()) fail(path + ": the safetensors header is not an object");
    const size_t data0 = 8 + size_t(hlen), data_bytes = mf->size - data0;
    const int fi = int(files_.size());
    const json::Object& o = header.as_object();
    for (size_t i = 0; i < o.size(); ++i) {
      if (o.keys[i] == "__metadata__") continue;
      const std::string where = path + ": tensor " + o.keys[i];
      const json::Value& t = o.vals[i];
      const json::Value* shape = t.get("shape");
      const json::Value* offs = t.get("data_offsets");
      if (!shape || !shape->is_array() || !offs || !offs->is_array() || offs->size() != 2)
        fail(where + " has no shape or data_offsets");
      Entry e;
      e.dtype = t.get_string("dtype", "");
      uint64_t count = 1;
      for (const json::Value& s : shape->as_array()) {
        const int64_t d = s.as_int();
        if (d < 0 || (d > 0 && count > UINT64_MAX / uint64_t(d))) fail(where + " has a bad shape");
        count *= uint64_t(d);
        e.shape.push_back(d);
      }
      const int64_t begin = (*offs)[0].as_int(), end = (*offs)[1].as_int();
      if (begin < 0 || end < begin || uint64_t(end) > data_bytes) fail(where + " lies past the end of the file");
      e.offset = data0 + size_t(begin);
      e.nbytes = size_t(end - begin);
      e.file = fi;
      if (const size_t eb = dtype_bytes(e.dtype); eb && (count > SIZE_MAX / eb || count * eb != e.nbytes))
        fail(where + ": " + std::to_string(e.nbytes) + " bytes do not hold its shape");
      entries_[o.keys[i]] = std::move(e);
    }
    files_.push_back(std::move(mf));
  }

  std::string dir_;
  std::vector<std::unique_ptr<MappedFile>> files_;
  std::unordered_map<std::string, Entry> entries_;
  json::Value config_;
  int default_bits_ = 4, default_group_ = 64;
  std::unordered_map<std::string, std::pair<int, int>> overrides_;
};

}  // namespace espetero::native
