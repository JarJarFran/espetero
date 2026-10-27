// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "st.h"

#include <fcntl.h>
#include <unistd.h>

namespace espetero::quant {

size_t dtype_size(const std::string& t) {
  if (t == "F64" || t == "I64" || t == "U64") return 8;
  if (t == "F32" || t == "I32" || t == "U32") return 4;
  if (t == "F16" || t == "BF16" || t == "I16" || t == "U16") return 2;
  if (t == "I8" || t == "U8" || t == "BOOL") return 1;
  fail("safetensors: unsupported dtype " + t);
}

size_t StTensor::nbytes() const {
  size_t n = dtype_size(dtype);
  for (int64_t d : shape) n *= size_t(d);
  return n;
}

StFile::StFile(const std::string& path) : f_(std::make_unique<native::MappedFile>(path)) {
  if (f_->size < 8) fail(path + ": not a safetensors file");
  uint64_t hlen = 0;
  std::memcpy(&hlen, f_->at(0), 8);
  if (hlen > f_->size - 8) fail(path + ": truncated safetensors header");
  const json::Value h = json::parse(std::string_view(reinterpret_cast<const char*>(f_->at(8)), size_t(hlen)));
  const size_t data0 = 8 + size_t(hlen);
  const json::Object& o = h.as_object();
  for (size_t i = 0; i < o.size(); ++i) {
    const json::Value& v = o.vals[i];
    if (o.keys[i] == "__metadata__") {
      const json::Object& m = v.as_object();
      for (size_t j = 0; j < m.size(); ++j) meta_[m.keys[j]] = m.vals[j].as_string();
      continue;
    }
    Entry e;
    e.dtype = v.get_string("dtype", "");
    size_t n = dtype_size(e.dtype);
    for (const json::Value& d : v.get("shape")->as_array()) {
      e.shape.push_back(d.as_int());
      n *= size_t(d.as_int());
    }
    const auto& off = v.get("data_offsets")->as_array();
    const size_t b = size_t(off[0].as_int()), en = size_t(off[1].as_int());
    if (en < b || en - b != n || data0 + en > f_->size) fail(path + ": bad tensor " + o.keys[i]);
    e.data = f_->at(data0 + b);
    e.nbytes = n;
    entries_[o.keys[i]] = std::move(e);
  }
}

const StFile::Entry& StFile::at(const std::string& k) const {
  auto it = entries_.find(k);
  if (it == entries_.end()) fail("missing tensor " + k);
  return it->second;
}

namespace {

json::Value shape_json(const std::vector<int64_t>& s) {
  json::Value a = json::Value::array();
  for (int64_t d : s) a.push_back(json::Value(d));
  return a;
}

// The header of one file: tensors in the given order, data contiguous, the
// header padded with spaces to a multiple of 8 bytes.
std::string header(const std::vector<const StTensor*>& ts, const std::map<std::string, std::string>& meta,
                   std::vector<size_t>& offsets) {
  json::Value h = json::Value::object();
  if (!meta.empty()) {
    json::Value m = json::Value::object();
    for (const auto& [k, v] : meta) m.set(k, json::Value(v));
    h.set("__metadata__", std::move(m));
  }
  size_t off = 0;
  offsets.clear();
  for (const StTensor* t : ts) {
    json::Value e = json::Value::object();
    e.set("dtype", json::Value(t->dtype));
    e.set("shape", shape_json(t->shape));
    json::Value o = json::Value::array();
    o.push_back(json::Value(off));
    o.push_back(json::Value(off + t->nbytes()));
    e.set("data_offsets", std::move(o));
    h.set(t->name, std::move(e));
    offsets.push_back(off);
    off += t->nbytes();
  }
  std::string s = json::dump(h);
  while (s.size() % 8) s.push_back(' ');
  return s;
}

void pwrite_all(int fd, const void* p, size_t n, size_t off, const std::string& what) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  while (n) {
    const ssize_t w = ::pwrite(fd, b, std::min<size_t>(n, size_t(1) << 30), off_t(off));
    if (w <= 0) fail("cannot write " + what + ": " + std::strerror(errno));
    b += w;
    n -= size_t(w);
    off += size_t(w);
  }
}

}  // namespace

StWriter::StWriter(const std::string& dir, std::vector<StTensor> tensors, size_t shard_bytes,
                   const std::map<std::string, std::string>& metadata,
                   const std::function<std::string(int, int)>& file_name) {
  std::sort(tensors.begin(), tensors.end(), [](const StTensor& a, const StTensor& b) { return a.name < b.name; });
  std::vector<std::vector<const StTensor*>> shards(1);
  size_t size = 0;
  for (const StTensor& t : tensors) {
    const size_t nb = t.nbytes();
    if (!shards.back().empty() && size + nb > shard_bytes) {
      shards.emplace_back();
      size = 0;
    }
    shards.back().push_back(&t);
    size += nb;
  }
  const int n = int(shards.size());
  for (int si = 0; si < n; ++si) {
    std::vector<size_t> offs;
    const std::string h = header(shards[size_t(si)], metadata, offs);
    const std::string fn = file_name(si, n);
    const std::string path = dir + "/" + fn;
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) fail("cannot create " + path);
    fds_.push_back(fd);
    const uint64_t hl = h.size();
    pwrite_all(fd, &hl, 8, 0, path);
    pwrite_all(fd, h.data(), h.size(), 8, path);
    size_t end = 8 + h.size();
    for (size_t i = 0; i < shards[size_t(si)].size(); ++i) {
      const StTensor* t = shards[size_t(si)][i];
      slot_[t->name] = {si, 8 + h.size() + offs[i], t->nbytes(), false};
      wmap_[t->name] = fn;
      total_ += t->nbytes();
      end = 8 + h.size() + offs[i] + t->nbytes();
    }
    if (::ftruncate(fd, off_t(end)) != 0) fail("cannot size " + path);
  }
}

StWriter::~StWriter() {
  for (int fd : fds_) ::close(fd);
}

void StWriter::write(const std::string& name, const void* data, size_t nbytes) {
  Slot* s = nullptr;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = slot_.find(name);
    if (it == slot_.end()) fail("writer: " + name + " was not planned");
    s = &it->second;
    if (s->done) fail("writer: " + name + " written twice");
    if (nbytes != s->nbytes)
      fail("writer: " + name + " has " + std::to_string(nbytes) + " bytes, planned " + std::to_string(s->nbytes));
    s->done = true;
  }
  pwrite_all(fds_[size_t(s->file)], data, nbytes, s->off, name);
}

void StWriter::finish() {
  for (const auto& [k, s] : slot_)
    if (!s.done) fail("writer: " + k + " was planned but never written");
  for (int fd : fds_) ::fsync(fd);
}

void st_save(const std::string& path, const std::vector<std::pair<StTensor, const void*>>& tensors,
             const std::map<std::string, std::string>& metadata) {
  std::vector<const StTensor*> ts;
  for (const auto& [t, _] : tensors) ts.push_back(&t);
  std::vector<size_t> offs;
  const std::string h = header(ts, metadata, offs);
  const std::string tmp = path + ".partial";
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) fail("cannot create " + tmp);
  const uint64_t hl = h.size();
  pwrite_all(fd, &hl, 8, 0, tmp);
  pwrite_all(fd, h.data(), h.size(), 8, tmp);
  for (size_t i = 0; i < tensors.size(); ++i)
    pwrite_all(fd, tensors[i].second, tensors[i].first.nbytes(), 8 + h.size() + offs[i], tmp);
  ::close(fd);
  if (::rename(tmp.c_str(), path.c_str()) != 0) fail("cannot rename " + tmp);
}

}  // namespace espetero::quant
