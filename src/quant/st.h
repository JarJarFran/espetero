// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Safetensors for espetero quantize: a mapped reader for single files
// (calibration statistics) and a writer whose layout is planned before any
// tensor is encoded, so each tensor is written at its offset as soon as it
// exists and the whole checkpoint never sits in memory.
#pragma once

#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "../model/impl/common.h"
#include "../util/json.h"

namespace espetero::quant {

using native::fail;

size_t dtype_size(const std::string& dtype);

// One safetensors file, mapped.
class StFile {
 public:
  explicit StFile(const std::string& path);
  struct Entry {
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
  };
  bool has(const std::string& k) const { return entries_.count(k) != 0; }
  const Entry& at(const std::string& k) const;
  const std::map<std::string, Entry>& entries() const { return entries_; }
  const std::map<std::string, std::string>& metadata() const { return meta_; }

 private:
  std::unique_ptr<native::MappedFile> f_;
  std::map<std::string, Entry> entries_;
  std::map<std::string, std::string> meta_;
};

// A planned tensor of a checkpoint being written.
struct StTensor {
  std::string name, dtype;
  std::vector<int64_t> shape;
  size_t nbytes() const;
};

// Shards of at most `shard_bytes` (tensors sorted by name, a new shard when
// the next tensor would pass the limit: eq_convert.py's split), created with
// their headers; write() puts a tensor's bytes in place.
class StWriter {
 public:
  // file_name(i, n): the name of shard i of n.
  StWriter(const std::string& dir, std::vector<StTensor> tensors, size_t shard_bytes,
           const std::map<std::string, std::string>& metadata,
           const std::function<std::string(int, int)>& file_name);
  ~StWriter();
  void write(const std::string& name, const void* data, size_t nbytes);
  // Every planned tensor written? (fails naming the first that was not)
  void finish();
  // name -> shard file (model.safetensors.index.json) and the total bytes.
  const std::map<std::string, std::string>& weight_map() const { return wmap_; }
  size_t total_bytes() const { return total_; }
  bool planned(const std::string& name) const { return slot_.count(name) != 0; }

 private:
  struct Slot {
    int file = 0;
    size_t off = 0, nbytes = 0;
    bool done = false;
  };
  std::vector<int> fds_;
  std::map<std::string, Slot> slot_;
  std::map<std::string, std::string> wmap_;
  size_t total_ = 0;
  std::mutex mu_;
};

// A whole small file at once (calibration statistics).
void st_save(const std::string& path, const std::vector<std::pair<StTensor, const void*>>& tensors,
             const std::map<std::string, std::string>& metadata);

}  // namespace espetero::quant
