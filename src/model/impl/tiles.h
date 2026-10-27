// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#pragma once

// Tile images: a checkpoint's matrices stored exactly as the kernels read
// them. A load repacks every matrix into the GPU's fragment order (codes,
// high-bit planes, lane-ordered scales) and fuses some (q/k/v, gate/up,
// expert stacks); the image keeps those finished buffers and each matrix's
// metadata, page-aligned in one file beside the checkpoint:
//
//   espetero-tiles.json  version, build id, source files (name, size, mtime),
//                        load options; per matrix (keyed by its load site,
//                        e.g. "language_model.model.layers.3.gate_up") the
//                        QW / HW fields and its buffers (segment, offset, size)
//   espetero-tiles.bin   segments of at most 1 GB, each page-aligned
//
// ESPETERO_TILES=write loads normally and writes the image; a later load
// whose build, files and options match maps the segments without copies
// (newBufferWithBytesNoCopy over the file's pages: the page cache is the
// model's memory) and skips reading, fusing and repacking those matrices.
// ESPETERO_TILES=0 ignores images. Qwen3.5 / Qwen3.6 MLX-layout checkpoints
// (EQ, MLX affine) for now.

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "model.h"
#include "../../util/json.h"

namespace espetero::native {

struct TileImage {
  static constexpr int kVersion = 1;
  static constexpr size_t kSegment = size_t(1) << 30;
  bool reading = false, writing = false;
  std::string json_path, bin_path;
  json::Value fingerprint;  // build, source files, options
  // reading: the mapped file and its segments as GPU buffers
  int fd = -1;
  void* base = nullptr;
  size_t size = 0;
  std::vector<id<MTLBuffer>> segs;
  std::map<std::string, Mat> mats;
  size_t bytes = 0;
  // writing: the finished matrices of this load, in order
  std::vector<std::pair<std::string, Mat>> rec;

  TileImage() = default;
  TileImage(const TileImage&) = delete;
  TileImage& operator=(const TileImage&) = delete;
  ~TileImage() {
    for (id<MTLBuffer> b : segs) Residency::get().remove(b);
    segs.clear();
    if (base) ::munmap(base, size);
    if (fd >= 0) ::close(fd);
  }
  bool has(const std::string& k) const { return mats.count(k) != 0; }
  void record(const std::string& k, const Mat& m) {
    if (writing && !k.empty()) rec.push_back({k, m});
  }
};

inline bool tiles_reading(const Model& m) { return m.tiles && m.tiles->reading; }

// The matrix `key` of the image, counted as finish() counts it.
inline Mat tile_mat(Model& m, const std::string& key) {
  auto it = m.tiles->mats.find(key);
  if (it == m.tiles->mats.end()) fail("tile image " + m.tiles->json_path + ": no matrix " + key);
  const Mat& out = it->second;
  if (out.dense) ++m.n_dense;
  else ++(out.q.bits == 4 ? m.n_q4 : out.q.bits == 5 ? m.n_q5 : out.q.bits == 6 ? m.n_q6 : out.q.bits == 3 ? m.n_q3 : m.n_q8);
  return out;
}

// What an image is valid for: this build, these files, these load options.
inline json::Value tiles_fingerprint(const Model& m, const std::string& dir) {
  json::Value fp = json::Value::object();
  fp.set("version", TileImage::kVersion);
  fp.set("build", m.build_id);
  json::Value files = json::Value::array();
  std::vector<std::string> names;
  if (DIR* d = ::opendir(dir.c_str())) {
    while (dirent* e = ::readdir(d)) {
      const std::string n = e->d_name;
      if (ends_with(n, ".safetensors") || n == "config.json" || n == "model.safetensors.index.json") names.push_back(n);
    }
    ::closedir(d);
  }
  std::sort(names.begin(), names.end());
  for (const std::string& n : names) {
    struct stat st;
    if (::stat((dir + "/" + n).c_str(), &st) != 0) continue;
    json::Value f = json::Value::array();
    f.push_back(n);
    f.push_back(int64_t(st.st_size));
    f.push_back(int64_t(st.st_mtimespec.tv_sec));
    files.push_back(std::move(f));
  }
  fp.set("files", std::move(files));
  json::Value opt = json::Value::object();
  opt.set("mtp_head4", m.mtp.head4 ? 1 : 0);
  opt.set("mtp_coarse", m.mtp.coarse_c);
  opt.set("mtp_q8", m.mtp.q8 ? 1 : 0);
  std::string dense;
  for (const std::string& k : m.dense_mats) dense += k + ",";
  opt.set("dense_mats", dense);
  for (const char* v : {"ESPETERO_ROUTER_Q8", "ESPETERO_GGUF_AFFINE"}) {
    const char* x = std::getenv(v);
    opt.set(v, x ? x : "");
  }
  fp.set("options", std::move(opt));
  return fp;
}

// Opens dir's image for reading when it matches; otherwise prepares one for
// writing when ESPETERO_TILES=write. No image (or ESPETERO_TILES=0): nullptr.
inline std::shared_ptr<TileImage> open_tiles(const Model& m, const std::string& dir) {
  const char* mode = std::getenv("ESPETERO_TILES");
  const std::string md = mode ? mode : "";
  if (md == "0") return nullptr;
  auto t = std::make_shared<TileImage>();
  t->json_path = dir + "/espetero-tiles.json";
  t->bin_path = dir + "/espetero-tiles.bin";
  t->fingerprint = tiles_fingerprint(m, dir);
  if (md == "write") {
    t->writing = true;
    return t;
  }
  if (!file_exists(t->json_path) || !file_exists(t->bin_path)) return nullptr;
  json::Value idx;
  try {
    idx = json::parse(read_file(t->json_path));
  } catch (const std::exception&) {
    return nullptr;
  }
  const json::Value* fp = idx.get("fingerprint");
  if (!fp || json::dump(*fp) != json::dump(t->fingerprint)) {
    std::fprintf(stderr, "[espetero] %s is for another build, files or options: loading normally "
                         "(ESPETERO_TILES=write rewrites it)\n", t->json_path.c_str());
    return nullptr;
  }
  t->fd = ::open(t->bin_path.c_str(), O_RDONLY);
  if (t->fd < 0) return nullptr;
  struct stat st;
  ::fstat(t->fd, &st);
  t->size = size_t(st.st_size);
  t->base = ::mmap(nullptr, t->size, PROT_READ, MAP_SHARED, t->fd, 0);
  if (t->base == MAP_FAILED) {
    t->base = nullptr;
    return nullptr;
  }
  for (const json::Value& s : idx.get("segments")->as_array()) {
    const size_t off = size_t(s[0].as_int()), len = size_t(s[1].as_int());
    if (off % size_t(getpagesize()) || len % size_t(getpagesize()) || off + len > t->size)
      fail(t->bin_path + ": segment not page-aligned or past the file");
    id<MTLBuffer> b = [m.dev newBufferWithBytesNoCopy:static_cast<uint8_t*>(t->base) + off
                                               length:len
                                              options:kShared
                                          deallocator:nil];
    if (!b) fail(t->bin_path + ": cannot map a segment as a GPU buffer");
    Residency::get().add(b);
    t->segs.push_back(b);
    t->bytes += len;
  }
  auto buf = [&](const json::Value* v, size_t& n) -> Buf {
    n = 0;
    if (!v || !v->is_array()) return Buf{};
    const size_t seg = size_t((*v)[0].as_int());
    if (seg >= t->segs.size()) fail(t->json_path + ": bad segment index");
    n = size_t((*v)[2].as_int());
    return Buf(t->segs[seg], size_t((*v)[1].as_int()));
  };
  const json::Object& mats = idx.get("mats")->as_object();
  for (size_t i = 0; i < mats.size(); ++i) {
    const json::Value& e = mats.vals[i];
    Mat out;
    out.dense = e.get_int("dense", 0) != 0;
    if (out.dense) {
      HW& d = out.d;
      d.out = int(e.get_int("out", 0));
      d.in = int(e.get_int("in", 0));
      d.kp = int(e.get_int("kp", 0));
      d.act = int(e.get_int("act", 0));
      d.has_bias = e.get_int("has_bias", 0) != 0;
      d.wr = buf(e.get("wr"), d.wr_n);
      d.bias = buf(e.get("bias"), d.bias_n);
    } else {
      QW& q = out.q;
      q.out = int(e.get_int("out", 0));
      q.in = int(e.get_int("in", 0));
      q.bits = int(e.get_int("bits", 4));
      q.src_bits = int(e.get_int("src_bits", 4));
      q.group = int(e.get_int("group", 64));
      q.half_scales = e.get_int("half_scales", 0) != 0;
      q.scale_mode = int(e.get_int("scale_mode", 0));
      q.act = int(e.get_int("act", 0));
      q.wr = buf(e.get("wr"), q.wr_n);
      q.sb = buf(e.get("sb"), q.sb_n);
      q.wh = buf(e.get("wh"), q.wh_n);
    }
    t->mats[mats.keys[i]] = out;
  }
  t->reading = true;
  return t;
}

// Writes the image of this load (ESPETERO_TILES=write), after its GPU work.
inline void save_tiles(Model& m) {
  if (!m.tiles || !m.tiles->writing) return;
  TileImage& t = *m.tiles;
  const size_t page = size_t(getpagesize());
  struct Piece {
    const Buf* b;
    size_t n, seg, off;
  };
  std::vector<Piece> pieces;
  json::Value segs = json::Value::array();
  std::vector<size_t> seg_off;  // file offset per segment
  size_t cur = 0, file_end = 0;
  auto new_seg = [&]() {
    if (!seg_off.empty()) {
      const size_t len = (cur + page - 1) / page * page;
      json::Value s = json::Value::array();
      s.push_back(int64_t(seg_off.back()));
      s.push_back(int64_t(len));
      segs.push_back(std::move(s));
      file_end = seg_off.back() + len;
    }
    seg_off.push_back(file_end);
    cur = 0;
  };
  new_seg();
  json::Value mats = json::Value::object();
  auto place = [&](const Buf& b, size_t n) -> json::Value {
    if (!b || n == 0) return json::Value();
    const size_t a = (n + 255) / 256 * 256;
    if (cur > 0 && cur + a > TileImage::kSegment) new_seg();
    pieces.push_back({&b, n, seg_off.size() - 1, cur});
    json::Value v = json::Value::array();
    v.push_back(int64_t(seg_off.size() - 1));
    v.push_back(int64_t(cur));
    v.push_back(int64_t(n));
    cur += a;
    return v;
  };
  for (const auto& [key, mt] : t.rec) {
    json::Value e = json::Value::object();
    e.set("dense", mt.dense ? 1 : 0);
    if (mt.dense) {
      const HW& d = mt.d;
      e.set("out", d.out);
      e.set("in", d.in);
      e.set("kp", d.kp);
      e.set("act", d.act);
      e.set("has_bias", d.has_bias ? 1 : 0);
      if (d.wr && !d.wr_n) fail("tile image: " + key + " has a buffer of unknown size");
      e.set("wr", place(d.wr, d.wr_n));
      if (d.has_bias) e.set("bias", place(d.bias, d.bias_n));
    } else {
      const QW& q = mt.q;
      e.set("out", q.out);
      e.set("in", q.in);
      e.set("bits", q.bits);
      e.set("src_bits", q.src_bits);
      e.set("group", q.group);
      e.set("half_scales", q.half_scales ? 1 : 0);
      e.set("scale_mode", q.scale_mode);
      e.set("act", q.act);
      if ((q.wr && !q.wr_n) || (q.sb && !q.sb_n) || (q.wh && !q.wh_n))
        fail("tile image: " + key + " has a buffer of unknown size");
      e.set("wr", place(q.wr, q.wr_n));
      e.set("sb", place(q.sb, q.sb_n));
      if (q.wh) e.set("wh", place(q.wh, q.wh_n));
    }
    mats.set(key, std::move(e));
  }
  new_seg();  // closes the last segment
  seg_off.pop_back();
  const std::string tmp_bin = t.bin_path + ".partial", tmp_json = t.json_path + ".partial";
  const int fd = ::open(tmp_bin.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd < 0) fail("cannot write " + tmp_bin);
  for (const Piece& p : pieces) {
    const uint8_t* src = static_cast<const uint8_t*>(ptr(*p.b));
    size_t done = 0;
    while (done < p.n) {
      const ssize_t w = ::pwrite(fd, src + done, p.n - done, off_t(seg_off[p.seg] + p.off + done));
      if (w <= 0) {
        ::close(fd);
        fail("cannot write " + tmp_bin);
      }
      done += size_t(w);
    }
  }
  if (::ftruncate(fd, off_t(file_end)) != 0) {
    ::close(fd);
    fail("cannot size " + tmp_bin);
  }
  ::close(fd);
  json::Value idx = json::Value::object();
  idx.set("fingerprint", t.fingerprint);
  idx.set("segments", std::move(segs));
  idx.set("mats", std::move(mats));
  {
    std::ofstream f(tmp_json);
    f << json::dump(idx);
    if (!f) fail("cannot write " + tmp_json);
  }
  if (::rename(tmp_bin.c_str(), t.bin_path.c_str()) != 0 || ::rename(tmp_json.c_str(), t.json_path.c_str()) != 0)
    fail("cannot move the tile image into place (" + t.json_path + ")");
  std::fprintf(stderr, "[espetero] tile image: %zu matrices, %.2f GB -> %s\n", t.rec.size(), double(file_end) / 1e9,
               t.bin_path.c_str());
  t.writing = false;
  t.rec.clear();
}

}  // namespace espetero::native
