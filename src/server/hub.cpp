// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "hub.h"

#include <dirent.h>
#include <fcntl.h>
#include <fts.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <climits>
#include <map>
#include <set>
#include <sstream>
#include <tuple>

#include "../util/json.h"
#include "../util/model_family.h"
#include "../util/paths.h"
#include "download.h"

namespace espetero::hub {

namespace {

std::string lower(std::string s) {
  for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}
bool starts_with(const std::string& s, const std::string& x) { return s.compare(0, x.size(), x) == 0; }
bool ends_with(const std::string& s, const std::string& x) {
  return s.size() >= x.size() && s.compare(s.size() - x.size(), x.size(), x) == 0;
}
bool separator(char c) { return c == '-' || c == '_' || c == '.'; }

bool is_file(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string read_text(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  if (f) ss << f.rdbuf();
  return ss.str();
}

std::string dirname_of(const std::string& p) {
  const size_t s = p.find_last_of('/');
  return s == std::string::npos ? std::string() : p.substr(0, s);
}
std::string basename_of(const std::string& p) {
  const size_t s = p.find_last_of('/');
  return s == std::string::npos ? p : p.substr(s + 1);
}
std::string join(const std::string& dir, const std::string& rel) { return dir.empty() ? rel : dir + "/" + rel; }

std::vector<std::string> entries(const std::string& dir) {
  std::vector<std::string> out;
  if (DIR* d = ::opendir(dir.c_str())) {
    while (dirent* e = ::readdir(d)) {
      const std::string n = e->d_name;
      if (n != "." && n != "..") out.push_back(n);
    }
    ::closedir(d);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// Regular files under root/rel as paths relative to root. Symlinks are
// followed (snapshot files link to blobs); one whose blob is missing (a
// download cut short) is skipped.
void walk(const std::string& root, const std::string& rel, int depth, std::vector<std::string>& out) {
  for (const std::string& n : entries(join(root, rel))) {
    const std::string r = join(rel, n);
    struct stat st;
    if (::stat((root + "/" + r).c_str(), &st) != 0) continue;
    if (S_ISREG(st.st_mode)) out.push_back(r);
    else if (S_ISDIR(st.st_mode) && depth > 0) walk(root, r, depth - 1, out);
  }
}

// ---------------------------------------------------------------- GGUF files
enum Kind { kModel, kProjector, kMtp, kOther };

Kind kind_of(const std::string& rel) {
  const std::string base = lower(basename_of(rel));
  if (!ends_with(base, ".gguf")) return kOther;
  if (base.find("mmproj") != std::string::npos) return kProjector;
  if (starts_with(base, "mtp") || ("/" + lower(dirname_of(rel)) + "/").find("/mtp/") != std::string::npos) return kMtp;
  if (base.find("imatrix") != std::string::npos) return kOther;
  return kModel;
}

// One model: a .gguf file, or the parts of a split one (X-00001-of-00003.gguf).
struct Group {
  std::string stem;                // file name without .gguf and the part suffix
  std::vector<std::string> files;  // relative paths, sorted
  int parts = 0;                   // > 0: split in this many parts
  int rank = -1;                   // selector match: 0 the whole stem, 1 its end after a separator
};

std::vector<Group> model_groups(const std::vector<std::string>& files) {
  std::map<std::string, Group> by;
  for (const std::string& f : files) {
    if (kind_of(f) != kModel) continue;
    std::string stem = basename_of(f);
    stem.resize(stem.size() - 5);  // ".gguf"
    int parts = 0;
    if (stem.size() > 15) {  // llama.cpp's part suffix: -%05d-of-%05d
      const std::string t = stem.substr(stem.size() - 15);
      bool ok = t[0] == '-' && t.compare(6, 4, "-of-") == 0;
      for (size_t i : {1, 2, 3, 4, 5, 10, 11, 12, 13, 14}) ok = ok && std::isdigit(static_cast<unsigned char>(t[i]));
      if (ok) {
        parts = std::max(1, std::atoi(t.c_str() + 10));
        stem.resize(stem.size() - 15);
      }
    }
    Group& g = by[join(dirname_of(f), stem)];
    g.stem = stem;
    g.files.push_back(f);
    g.parts = std::max(g.parts, parts);
  }
  std::vector<Group> out;
  for (auto& [key, g] : by) {
    std::sort(g.files.begin(), g.files.end());
    out.push_back(std::move(g));
  }
  return out;
}

// A model's quantization in its name: the end that starts after a separator
// with UD-, Q<digit>, IQ<digit>, TQ<digit>, F16, BF16, F32 or MXFP
// ("Qwen3.8-27B-UD-Q6_K" -> "UD-Q6_K"), else the whole name.
std::string quant_selector(const std::string& stem) {
  for (size_t i = 0; i + 1 < stem.size(); ++i) {
    if (!separator(stem[i])) continue;
    std::string u = stem.substr(i + 1);
    for (char& c : u) c = char(std::toupper(static_cast<unsigned char>(c)));
    auto digit = [&](size_t k) { return k < u.size() && std::isdigit(static_cast<unsigned char>(u[k])); };
    if (starts_with(u, "UD-") || (u[0] == 'Q' && digit(1)) || ((starts_with(u, "IQ") || starts_with(u, "TQ")) && digit(2)) ||
        starts_with(u, "F16") || starts_with(u, "BF16") || starts_with(u, "F32") || starts_with(u, "MXFP"))
      return stem.substr(i + 1);
  }
  return stem;
}

// The selectors of the models: their names after the prefix they all share,
// cut at a separator ("Qwen3.8-27B-UD-Q6_K" -> "UD-Q6_K"); a lone model's
// quantization (quant_selector).
std::vector<std::string> selectors(const std::vector<Group>& gs) {
  if (gs.size() == 1) return {quant_selector(gs[0].stem)};
  std::string prefix = gs.empty() ? std::string() : gs[0].stem;
  for (const Group& g : gs) {
    size_t n = 0;
    while (n < prefix.size() && n < g.stem.size() && prefix[n] == g.stem[n]) ++n;
    prefix.resize(n);
  }
  size_t keep = 0;
  for (size_t i = 0; i < prefix.size(); ++i)
    if (separator(prefix[i])) keep = i + 1;
  std::vector<std::string> out;
  for (const Group& g : gs) out.push_back(g.stem.size() > keep ? g.stem.substr(keep) : g.stem);
  return out;
}

std::string choices(const Ref& ref, const std::vector<Group>& gs) {
  const std::vector<std::string> sel = selectors(gs);
  std::string out;
  for (size_t i = 0; i < gs.size(); ++i) {
    out += "\n  --model " + ref.repo + ":" + sel[i];
    if (gs[i].parts) out += "   (split in " + std::to_string(gs[i].parts) + " files, not supported)";
  }
  return out;
}

// ---------------------------------------------------------------- snapshots
std::string repo_dir(const std::string& cache, const Ref& ref) {
  const size_t slash = ref.repo.find('/');
  return cache + "/models--" + ref.repo.substr(0, slash) + "--" + ref.repo.substr(slash + 1);
}

// Snapshot directories, newest first: the revision refs/main names, then by
// modification time.
std::vector<std::string> snapshots(const std::string& repo) {
  std::string main = read_text(repo + "/refs/main");
  while (!main.empty() && std::isspace(static_cast<unsigned char>(main.back()))) main.pop_back();
  struct Snap {
    std::string name;
    bool main;
    timespec mtime;
  };
  std::vector<Snap> v;
  for (const std::string& n : entries(repo + "/snapshots")) {
    struct stat st;
    if (::stat((repo + "/snapshots/" + n).c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) continue;
    v.push_back({n, n == main, st.st_mtimespec});
  }
  std::sort(v.begin(), v.end(), [](const Snap& a, const Snap& b) {
    if (a.main != b.main) return a.main;
    if (a.mtime.tv_sec != b.mtime.tv_sec) return a.mtime.tv_sec > b.mtime.tv_sec;
    if (a.mtime.tv_nsec != b.mtime.tv_nsec) return a.mtime.tv_nsec > b.mtime.tv_nsec;
    return a.name < b.name;
  });
  std::vector<std::string> out;
  for (const Snap& s : v) out.push_back(repo + "/snapshots/" + s.name);
  return out;
}

// An MLX model the engine can load: config.json, tokenizer.json and every
// weight file (the index's shards, else model.safetensors).
bool mlx_complete(const std::string& dir) {
  if (!is_file(dir + "/config.json") || !is_file(dir + "/tokenizer.json")) return false;
  const std::string index = dir + "/model.safetensors.index.json";
  if (!is_file(index)) return is_file(dir + "/model.safetensors");
  try {
    const json::Value idx = json::parse(read_text(index));
    const json::Value* wm = idx.get("weight_map");
    if (!wm || !wm->is_object() || !wm->size()) return false;
    for (const json::Value& f : wm->as_object().vals)
      if (!f.is_string() || !is_file(dir + "/" + f.as_string())) return false;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// The engine finds a projector next to the model (mmproj*.gguf) and an MTP
// head there or in its MTP directory (mtp*.gguf); others are passed to it.
bool engine_finds(const std::string& dir, const char* prefix) {
  for (const std::string& n : entries(dir)) {
    const std::string l = lower(n);
    if (starts_with(l, prefix) && ends_with(l, ".gguf") && is_file(dir + "/" + n)) return true;
  }
  return false;
}

Model from_pick(const std::string& snap, const GgufPick& p) {
  Model m;
  m.path = snap + "/" + p.model;
  const std::string dir = dirname_of(m.path);
  if (!p.mmproj.empty() && !engine_finds(dir, "mmproj")) m.mmproj = snap + "/" + p.mmproj;
  if (!p.mtp.empty() && !engine_finds(dir, "mtp") && !engine_finds(dir + "/MTP", "mtp")) m.mtp = snap + "/" + p.mtp;
  return m;
}

// ---------------------------------------------------------------- memory
bool has_mlx_files(const std::vector<std::string>& files) {
  return std::count(files.begin(), files.end(), "config.json") &&
         std::any_of(files.begin(), files.end(), [](const std::string& f) { return ends_with(f, ".safetensors"); });
}

// Other files of the model in `info` (the GGUF repository's other
// quantizations; for MLX, repositories named with other bit widths), with
// footprints scaled from `fp`: the same geometry, projector and MTP head.
std::vector<models::Alternative> alternatives_from(const Ref& ref, const RepoInfo& info, const models::Footprint& fp,
                                                   const std::string& gguf_model) {
  std::vector<models::Alternative> out;
  auto scaled = [&](const std::string& model, int64_t disk, int64_t gpu, int64_t own_disk, const std::string& quant) {
    models::Alternative a;
    a.model = model;
    a.footprint = fp;
    a.footprint.weights = fp.weights - fp.base + gpu;
    a.footprint.base = gpu;
    a.footprint.disk = fp.disk - own_disk + disk;
    a.footprint.quant = quant;
    out.push_back(std::move(a));
  };
  if (!gguf_model.empty()) {
    std::vector<std::string> paths;
    for (const RepoFile& f : info.files) paths.push_back(f.path);
    const std::vector<Group> gs = model_groups(paths);
    const std::vector<std::string> sel = selectors(gs);
    const RepoFile* own = info.find(gguf_model);
    for (size_t i = 0; i < gs.size(); ++i) {
      if (gs[i].parts || gs[i].files[0] == gguf_model) continue;
      const RepoFile* f = info.find(gs[i].files[0]);
      const double factor = models::gguf_quant_factor(gs[i].stem);
      if (!f || factor <= 0) continue;
      scaled(ref.repo + ":" + sel[i], f->size, int64_t(double(f->size) * factor), own ? own->size : 0, sel[i]);
    }
    for (models::Alternative& a : models::catalog_alternatives(ref.repo + (ref.quant.empty() ? "" : ":" + ref.quant)))
      out.push_back(std::move(a));  // the model's files in other repositories
    return out;
  }
  // The candidate names, looked up at once.
  const size_t slash = ref.repo.find('/');
  std::vector<std::future<std::pair<std::string, int64_t>>> found;
  for (const std::string& name : models::bit_siblings(ref.repo.substr(slash + 1)))
    found.push_back(std::async(std::launch::async, [repo = ref.repo.substr(0, slash + 1) + name] {
      int64_t disk = 0;
      try {
        for (const RepoFile& f : repo_info(repo).files)
          if (ends_with(f.path, ".safetensors") && f.path.find('/') == std::string::npos) disk += f.size;
      } catch (const std::exception&) {
      }
      return std::make_pair(repo, disk);
    }));
  for (auto& f : found)
    if (const auto [repo, disk] = f.get(); disk > 0) scaled(repo, disk, disk, fp.disk, "");
  for (models::Alternative& a : models::catalog_alternatives(ref.repo)) out.push_back(std::move(a));
  return out;
}

// The first `bytes` of each file at once (the safetensors headers of a
// sharded model), for a ReadFn that answers from them.
std::map<std::string, std::string> prefetch(const RepoInfo& info, const std::vector<std::string>& paths, int64_t bytes) {
  std::map<std::string, std::future<std::string>> pending;
  for (const std::string& p : paths)
    if (const RepoFile* f = info.find(p))
      pending[p] = std::async(std::launch::async, [&info, p, n = std::min(bytes, f->size)] {
        try {
          return read_range(info, p, 0, n);
        } catch (const std::exception&) {
          return std::string();  // read again, with its error, when needed
        }
      });
  std::map<std::string, std::string> out;
  for (auto& [p, f] : pending) out[p] = f.get();
  return out;
}

// ---------------------------------------------------------------- downloads
// The files of `ref` the engine reads, from its repository's listing. GGUF
// (a selector, or a repository without an MLX model): the selected model,
// one projector and the MTP head. MLX: the configuration, tokenizer and
// weight files at the repository's top level (*.json, *.safetensors,
// *.jinja, *.txt, tokenizer.model, *.tiktoken: what mlx-lm fetches), not
// its README, images or files of other formats.
struct Selection {
  bool gguf = false;
  GgufPick pick;
  std::vector<std::string> files;
};

Selection select_files(const Ref& ref, const RepoInfo& info) {
  std::vector<std::string> paths;
  for (const RepoFile& f : info.files) paths.push_back(f.path);
  const bool has_mlx = has_mlx_files(paths);
  const std::vector<Group> gs = model_groups(paths);
  Selection sel;
  if (!ref.quant.empty()) {
    if (gs.empty())
      throw Error(ref.repo + " has no GGUF model files" +
                  (has_mlx ? "; it is an MLX repository: pass --model " + ref.repo + " (without \":" + ref.quant + "\")" : ""));
    if (!pick_gguf(paths, ref, sel.pick))
      throw Error("no GGUF model of " + ref.repo + " matches \"" + ref.quant + "\"; name one of:" + choices(ref, gs));
    sel.gguf = true;
  } else if (!has_mlx) {
    if (!pick_gguf(paths, ref, sel.pick))  // throws when there are several
      throw Error(ref.repo + " holds neither an MLX model (config.json and *.safetensors) nor GGUF files");
    sel.gguf = true;
  }
  if (sel.gguf) {
    for (const std::string* f : {&sel.pick.model, &sel.pick.mmproj, &sel.pick.mtp})
      if (!f->empty()) sel.files.push_back(*f);
    return sel;
  }
  for (const std::string& p : paths) {
    if (p.find('/') != std::string::npos) continue;
    const std::string l = lower(p);
    for (const char* ext : {".json", ".safetensors", ".jinja", ".txt", ".tiktoken"})
      if (ends_with(l, ext) && l != "requirements.txt") {
        sel.files.push_back(p);
        break;
      }
    if (l == "tokenizer.model") sel.files.push_back(p);
  }
  return sel;
}

// The memory check before a download: the footprint from the listing
// and the headers of the files the engine would load (a few MB of ranges,
// the shards' headers at once), against this Mac. MLX: `config` is the
// repository's config.json, downloaded first.
void check_before_download(const Ref& ref, const ResolveOptions& opt, const RepoInfo& info, const Selection& sel,
                           const json::Value& config) {
  if (opt.memory == MemoryCheck::Skip) return;
  const std::string label = ref.repo + (ref.quant.empty() ? "" : ":" + ref.quant);
  const models::ReadFn read = [&](const std::string& p, int64_t off, int64_t len) { return read_range(info, p, off, len); };
  auto file = [&](const std::string& p) {
    const RepoFile* f = info.find(p);
    return models::FileRef{p, f ? f->size : 0};
  };
  models::Footprint fp;
  try {
    if (sel.gguf) {
      const models::FileRef m = file(sel.pick.model), pj = file(sel.pick.mmproj), mt = file(sel.pick.mtp);
      fp = models::footprint_gguf(m, sel.pick.mmproj.empty() ? nullptr : &pj, sel.pick.mtp.empty() ? nullptr : &mt, read);
    } else {
      if (const models::Geometry g = models::geometry_from_config(config); g.known && !g.unsupported.empty())
        throw Error(label + ": " + g.unsupported + ", which Espetero " + kVersion + " does not run");
      std::vector<models::FileRef> files;
      std::vector<std::string> shards;
      for (const RepoFile& f : info.files) {
        files.push_back({f.path, f.size});
        if (ends_with(f.path, ".safetensors") && f.path.find('/') == std::string::npos) shards.push_back(f.path);
      }
      const std::map<std::string, std::string> heads = prefetch(info, shards, 256 << 10);
      fp = models::footprint_mlx(config, files, [&](const std::string& p, int64_t off, int64_t len) {
        auto it = heads.find(p);
        if (it != heads.end() && off == 0 && len <= int64_t(it->second.size())) return it->second.substr(0, size_t(len));
        return read_range(info, p, off, len);
      });
    }
  } catch (const Error&) {
    throw;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[espetero] memory check skipped: %s: %s\n", label.c_str(), e.what());
    return;
  }
  if (fp.geo.known && !fp.geo.unsupported.empty())
    throw Error(label + ": " + fp.geo.unsupported + ", which Espetero " + kVersion + " does not run (its weights were not downloaded)");
  if (!sel.gguf && !fp.mlx_layout)
    throw Error(label + " is not an MLX conversion (its weights have no language_model.model.* tensors), which Espetero " +
                kVersion + " reads; its weights were not downloaded. `espetero recommend` lists MLX and GGUF files of the "
                "supported models");
  const models::Mac mac = models::this_mac();
  const models::Verdict v = models::judge(fp, opt.plan, mac);
  if (v.fits) return;
  const std::vector<models::Alternative> alts = alternatives_from(ref, info, fp, sel.gguf ? sel.pick.model : "");
  if (opt.memory == MemoryCheck::Refuse)
    throw Error(models::refusal(label, fp, opt.plan, mac, v, alts,
                                "Its weights were not downloaded. `espetero recommend` lists the models that fit this Mac; "
                                "--skip-memory-check downloads and runs it anyway."));
  std::fprintf(stderr, "[espetero] warning: %s\n",
               models::refusal(label, fp, opt.plan, mac, v, alts, "Downloading it anyway.").c_str());
}

}  // namespace

// ---------------------------------------------------------------- public
bool parse_ref(const std::string& model, Ref& ref) {
  if (model.empty() || std::count(model.begin(), model.end(), '/') != 1) return false;
  struct stat st;
  if (::stat(model.c_str(), &st) == 0) return false;  // an existing path
  const size_t colon = model.find(':');
  const std::string repo = model.substr(0, colon);
  const std::string quant = colon == std::string::npos ? std::string() : model.substr(colon + 1);
  auto ok = [](const std::string& s, bool name) {
    if (s.empty() || s.size() > 96 || (name && (s[0] == '.' || s[0] == '-'))) return false;
    for (char c : s)
      if (!std::isalnum(static_cast<unsigned char>(c)) && !separator(c)) return false;
    return true;
  };
  const size_t slash = repo.find('/');
  if (slash == std::string::npos || !ok(repo.substr(0, slash), true) || !ok(repo.substr(slash + 1), true)) return false;
  if (colon != std::string::npos && !ok(quant, false)) return false;
  ref.repo = repo;
  ref.quant = quant;
  return true;
}

std::string cache_dir() {
  auto env = [](const char* k) {
    const char* v = std::getenv(k);
    std::string s = v ? v : "";
    if (s == "~" || starts_with(s, "~/")) s = paths::home() + s.substr(1);
    while (s.size() > 1 && s.back() == '/') s.pop_back();
    return s;
  };
  if (std::string v = env("HF_HUB_CACHE"); !v.empty()) return v;
  if (std::string v = env("HUGGINGFACE_HUB_CACHE"); !v.empty()) return v;
  if (std::string v = env("HF_HOME"); !v.empty()) return v + "/hub";
  if (std::string v = env("XDG_CACHE_HOME"); !v.empty()) return v + "/huggingface/hub";
  return paths::home() + "/.cache/huggingface/hub";
}

bool offline_env() {
  const char* v = std::getenv("HF_HUB_OFFLINE");
  const std::string s = lower(v ? v : "");
  return s == "1" || s == "true" || s == "yes" || s == "on";
}

bool pick_gguf(const std::vector<std::string>& files, const Ref& ref, GgufPick& out) {
  std::vector<Group> gs = model_groups(files);
  if (gs.empty()) return false;
  const Group* g = nullptr;
  if (ref.quant.empty()) {
    if (gs.size() > 1)
      throw Error(ref.repo + " holds " + std::to_string(gs.size()) + " GGUF models; name one after a colon:" + choices(ref, gs));
    g = &gs[0];
  } else {
    const std::string q = lower(ref.quant);
    std::vector<const Group*> m;
    for (Group& x : gs) {
      const std::string s = lower(x.stem);
      x.rank = s == q ? 0 : (s.size() > q.size() && ends_with(s, q) && separator(s[s.size() - q.size() - 1])) ? 1 : -1;
      if (x.rank >= 0) m.push_back(&x);
    }
    if (m.empty()) return false;
    std::sort(m.begin(), m.end(), [](const Group* a, const Group* b) {
      return std::make_tuple(a->rank, a->stem.size(), a->files[0]) < std::make_tuple(b->rank, b->stem.size(), b->files[0]);
    });
    if (m.size() > 1 && m[0]->rank == m[1]->rank && m[0]->stem.size() == m[1]->stem.size()) {
      std::string tied;
      for (const Group* x : m)
        if (x->rank == m[0]->rank && x->stem.size() == m[0]->stem.size()) tied += "\n  " + x->files[0];
      throw Error(ref.repo + ":" + ref.quant + " matches several GGUF models:" + tied + "\nname one of:" + choices(ref, gs));
    }
    g = m[0];
  }
  if (g->parts) {
    std::string list;
    for (const std::string& f : g->files) list += "\n  " + f;
    throw Error(ref.repo + ":" + (ref.quant.empty() ? g->stem : ref.quant) + " is split in " + std::to_string(g->parts) +
                " files, which Espetero " + kVersion + " does not read:" + list +
                "\nmerge them with llama.cpp's `llama-gguf-split --merge FIRST_PART.gguf MERGED.gguf` and pass --model "
                "MERGED.gguf, or choose another quantization");
  }
  out = GgufPick{};
  out.model = g->files[0];
  const std::string dir = dirname_of(out.model);
  // One projector: next to the model first, then F16, BF16, F32, any.
  std::tuple<int, int, std::string> best_p{9, 9, ""}, best_m{9, 0, ""};
  for (const std::string& f : files) {
    const Kind k = kind_of(f);
    if (k == kProjector) {
      const std::string b = lower(basename_of(f));
      const bool bf16 = b.find("bf16") != std::string::npos;
      const int type = !bf16 && b.find("f16") != std::string::npos ? 0 : bf16 ? 1 : b.find("f32") != std::string::npos ? 2 : 3;
      best_p = std::min(best_p, std::make_tuple(dirname_of(f) == dir ? 0 : 1, type, f));
    } else if (k == kMtp) {
      // where the engine looks: next to the model, then its MTP directory
      const std::string d = lower(dirname_of(f));
      const int place = dirname_of(f) == dir ? 0 : d == lower(join(dir, "mtp")) ? 1 : 2;
      best_m = std::min(best_m, std::make_tuple(place, 0, f));
    }
  }
  out.mmproj = std::get<2>(best_p);
  out.mtp = std::get<2>(best_m);
  return true;
}

bool find_cached(const std::string& cache, const Ref& ref, Model& out) {
  for (const std::string& snap : snapshots(repo_dir(cache, ref))) {
    if (ref.quant.empty() && mlx_complete(snap)) {
      out = Model{};
      out.path = snap;
      out.revision = basename_of(snap);
      return true;
    }
    std::vector<std::string> files;
    walk(snap, "", 3, files);
    GgufPick p;
    if (pick_gguf(files, ref, p)) {
      out = from_pick(snap, p);
      out.revision = basename_of(snap);
      return true;
    }
  }
  return false;
}

Model resolve(const Ref& ref, bool offline) {
  ResolveOptions opt;
  opt.offline = offline;
  opt.memory = MemoryCheck::Skip;
  return resolve(ref, opt);
}

std::vector<models::Alternative> alternatives(const Ref& ref, const models::Footprint& fp, bool offline) {
  if (offline) return {};
  try {
    const RepoInfo info = repo_info(ref.repo);
    std::vector<std::string> paths;
    for (const RepoFile& f : info.files) paths.push_back(f.path);
    GgufPick pick;
    const bool gguf = !ref.quant.empty() || !has_mlx_files(paths);
    if (gguf && !pick_gguf(paths, ref, pick)) return {};
    return alternatives_from(ref, info, fp, gguf ? pick.model : "");
  } catch (const std::exception&) {
    return {};
  }
}

Model resolve(const Ref& ref, const ResolveOptions& opt) {
  const std::string cache = cache_dir();
  Model m;
  if (find_cached(cache, ref, m)) return m;
  const std::string id = ref.repo + (ref.quant.empty() ? "" : ":" + ref.quant);
  if (opt.offline)
    throw Error(id + " is not in the Hugging Face cache (" + cache + "), and downloads are off (--offline or "
                "HF_HUB_OFFLINE). Download it first, with a network: espetero pull " + id);
  const RepoInfo info = repo_info(ref.repo);
  const Selection sel = select_files(ref, info);
  json::Value config;
  if (!sel.gguf) {
    // config.json first: a model of another family is refused before its weights.
    if (!info.find("config.json")) throw Error(ref.repo + " has no config.json");
    download(cache, info, {"config.json"}, id, true);
    const std::string text = read_text(repo_dir(cache, ref) + "/snapshots/" + info.commit + "/config.json");
    try {
      config = json::parse(text);
    } catch (const json::Error& e) {
      throw Error(ref.repo + ": config.json: " + e.what());
    }
    if (const std::string msg = family::mlx_check(ref.repo, config); !msg.empty()) throw Error(msg);
  }
  check_before_download(ref, opt, info, sel, config);
  download(cache, info, sel.files, id);
  if (find_cached(cache, ref, m)) {
    std::fprintf(stderr, "[espetero] %s: %s\n", id.c_str(), m.path.c_str());
    return m;
  }
  throw Error(id + " was downloaded into " + cache +
              ", but no complete model is there (MLX: config.json, tokenizer.json and the *.safetensors weights; GGUF: the "
              "selected .gguf file)");
}

// ---------------------------------------------------------------- pull, list, rm
Model pull(const Ref& ref, const ResolveOptions& opt) {
  const std::string id = ref.repo + (ref.quant.empty() ? "" : ":" + ref.quant);
  if (opt.offline) throw Error("espetero pull " + id + " needs the network, and downloads are off (--offline or HF_HUB_OFFLINE)");
  const std::string cache = cache_dir();
  const RepoInfo info = repo_info(ref.repo);
  const Selection sel = select_files(ref, info);
  // Up to date: every file of the newest revision is there already.
  bool all = true;
  for (const std::string& f : sel.files) {
    const RepoFile* rf = info.find(f);
    struct stat st;
    all = all && rf && ::stat((repo_dir(cache, ref) + "/snapshots/" + info.commit + "/" + f).c_str(), &st) == 0 &&
          S_ISREG(st.st_mode) && int64_t(st.st_size) == rf->size;
  }
  Model m;
  if (all && find_cached(cache, ref, m) && m.revision == info.commit) {
    std::fprintf(stderr, "[espetero] %s is up to date (revision %s)\n", id.c_str(), info.commit.substr(0, 12).c_str());
    return m;
  }
  json::Value config;
  if (!sel.gguf) {
    if (!info.find("config.json")) throw Error(ref.repo + " has no config.json");
    download(cache, info, {"config.json"}, id, true);
    try {
      config = json::parse(read_text(repo_dir(cache, ref) + "/snapshots/" + info.commit + "/config.json"));
    } catch (const json::Error& e) {
      throw Error(ref.repo + ": config.json: " + e.what());
    }
    if (const std::string msg = family::mlx_check(ref.repo, config); !msg.empty()) throw Error(msg);
  }
  check_before_download(ref, opt, info, sel, config);
  download(cache, info, sel.files, id);
  if (find_cached(cache, ref, m)) return m;
  throw Error(id + " was downloaded into " + cache + ", but no complete model is there");
}

namespace {

bool is_dir(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Bytes of `files` (paths), each file once (links followed: blobs).
int64_t bytes_of(const std::vector<std::string>& files) {
  std::set<std::pair<dev_t, ino_t>> seen;
  int64_t n = 0;
  for (const std::string& f : files) {
    struct stat st;
    if (::stat(f.c_str(), &st) == 0 && S_ISREG(st.st_mode) && seen.insert({st.st_dev, st.st_ino}).second) n += st.st_size;
  }
  return n;
}

std::string real(const std::string& p) {
  char b[PATH_MAX];
  return ::realpath(p.c_str(), b) ? std::string(b) : std::string();
}

// The entries of a snapshot as relative paths, without following links:
// links (to blobs) and plain files.
void walk_links(const std::string& root, const std::string& rel, int depth, std::vector<std::string>& out) {
  for (const std::string& n : entries(join(root, rel))) {
    const std::string r = join(rel, n);
    struct stat st;
    if (::lstat((root + "/" + r).c_str(), &st) != 0) continue;
    if (S_ISLNK(st.st_mode) || S_ISREG(st.st_mode)) out.push_back(r);
    else if (S_ISDIR(st.st_mode) && depth > 0) walk_links(root, r, depth - 1, out);
  }
}

// The blob a snapshot entry links to ("" for a plain file or a link elsewhere).
std::string blob_of(const std::string& link) {
  char b[PATH_MAX];
  const ssize_t n = ::readlink(link.c_str(), b, sizeof b - 1);
  if (n <= 0) return "";
  const std::string t(b, size_t(n));
  const size_t at = t.rfind("/blobs/");
  if (at == std::string::npos || t.find("/", at + 7) != std::string::npos) return "";
  for (size_t i = 0; i < at; i += 3)  // only ../../blobs/<name> (and deeper ../), as huggingface_hub writes them
    if (t.compare(i, 3, "../") != 0) return "";
  return t.substr(at + 7);
}

// Removes a directory tree without following links (fts, physical).
void remove_tree(const std::string& dir) {
  char* argv[] = {const_cast<char*>(dir.c_str()), nullptr};
  FTS* fts = ::fts_open(argv, FTS_PHYSICAL | FTS_NOCHDIR | FTS_XDEV, nullptr);
  if (!fts) throw Error("cannot remove " + dir + ": " + std::strerror(errno));
  std::string err;
  while (FTSENT* e = ::fts_read(fts)) {
    switch (e->fts_info) {
      case FTS_D: break;
      case FTS_DP:
        if (::rmdir(e->fts_path) != 0 && err.empty()) err = std::string(e->fts_path) + ": " + std::strerror(errno);
        break;
      case FTS_DNR: case FTS_ERR: case FTS_NS:
        if (err.empty()) err = std::string(e->fts_path) + ": " + std::strerror(e->fts_errno);
        break;
      default:
        if (::unlink(e->fts_path) != 0 && err.empty()) err = std::string(e->fts_path) + ": " + std::strerror(errno);
    }
  }
  ::fts_close(fts);
  if (!err.empty()) throw Error("cannot remove " + err);
}

int64_t tree_bytes(const std::string& dir) {
  char* argv[] = {const_cast<char*>(dir.c_str()), nullptr};
  FTS* fts = ::fts_open(argv, FTS_PHYSICAL | FTS_NOCHDIR | FTS_XDEV, nullptr);
  int64_t n = 0;
  if (!fts) return 0;
  while (FTSENT* e = ::fts_read(fts))
    if (e->fts_info == FTS_F && e->fts_statp) n += e->fts_statp->st_size;
  ::fts_close(fts);
  return n;
}

}  // namespace

std::vector<Installed> installed(const std::string& cache, const std::string& models_dir, int* other) {
  std::vector<Installed> out;
  int skipped = 0;
  // A model the engine runs: its family (and geometry, when the files say it); else counted as another.
  auto keep = [&](Installed it, const std::string& path, const std::string& mmproj, const std::string& mtp) {
    try {
      it.footprint = models::footprint_local(path, mmproj, mtp);
    } catch (const std::exception&) {
      ++skipped;
      return;
    }
    const models::Footprint& f = it.footprint;
    const bool family = f.format == "GGUF" ? family::gguf_arch_supported(f.arch) : family::mlx_type_supported(f.arch);
    // MTP heads (qwen3_5_mtp, qwen4_exp_mtp drafters) are not models; MLX
    // directories must be MLX conversions (the loader's language_model.model.* names).
    if (!family || ends_with(f.arch, "_mtp") || (f.format == "MLX" && !f.mlx_layout) ||
        (f.geo.known && !f.geo.unsupported.empty())) {
      ++skipped;
      return;
    }
    out.push_back(std::move(it));
  };
  for (const std::string& n : entries(cache)) {
    if (!starts_with(n, "models--")) continue;
    const std::string rest = n.substr(8);
    const size_t sep = rest.find("--");
    if (sep == std::string::npos) continue;
    const std::string repo_id = rest.substr(0, sep) + "/" + rest.substr(sep + 2);
    for (const std::string& snap : snapshots(cache + "/" + n)) {  // the newest snapshot with a model
      if (mlx_complete(snap)) {
        Installed it;
        it.id = repo_id;
        it.path = snap;
        it.in_cache = true;
        std::vector<std::string> files;
        walk(snap, "", 3, files);
        for (std::string& f : files) f = snap + "/" + f;
        it.disk = bytes_of(files);
        keep(std::move(it), snap, "", "");
        break;
      }
      std::vector<std::string> files;
      walk(snap, "", 3, files);
      const std::vector<Group> gs = model_groups(files);
      if (gs.empty()) continue;
      const std::vector<std::string> sel = selectors(gs);
      for (size_t i = 0; i < gs.size(); ++i) {
        if (gs[i].parts) {  // split models are not read
          ++skipped;
          continue;
        }
        Ref r;
        r.repo = repo_id;
        r.quant = sel[i];
        GgufPick p;
        try {
          if (!pick_gguf(files, r, p) || p.model != gs[i].files[0]) continue;
        } catch (const Error&) {
          continue;
        }
        const Model m = from_pick(snap, p);
        Installed it;
        it.id = repo_id + ":" + sel[i];
        it.path = m.path;
        it.in_cache = true;
        std::vector<std::string> used = {m.path};
        for (const std::string* x : {&p.mmproj, &p.mtp})
          if (!x->empty()) used.push_back(snap + "/" + *x);
        it.disk = bytes_of(used);
        keep(std::move(it), m.path, m.mmproj, m.mtp);
      }
      break;
    }
  }
  // ~/models/<family>/<name>: MLX directories and .gguf files (in <name>/ or as <name> itself).
  for (const std::string& fam : entries(models_dir)) {
    const std::string fd = models_dir + "/" + fam;
    if (fam[0] == '.' || !is_dir(fd)) continue;
    for (const std::string& name : entries(fd)) {
      const std::string p = fd + "/" + name;
      std::vector<std::string> ggufs;
      if (is_dir(p) && is_file(p + "/config.json")) {
        if (!mlx_complete(p)) continue;
        Installed it;
        it.id = models::tilde(p);
        it.path = p;
        std::vector<std::string> files;
        walk(p, "", 0, files);
        for (std::string& f : files) f = p + "/" + f;
        it.disk = bytes_of(files);
        keep(std::move(it), p, "", "");
        continue;
      }
      if (is_dir(p)) {
        for (const std::string& f : entries(p))
          if (kind_of(f) == kModel) ggufs.push_back(p + "/" + f);
      } else if (kind_of(name) == kModel) {
        ggufs.push_back(p);
      }
      for (const std::string& g : ggufs) {
        const std::vector<Group> one = model_groups({g.substr(g.find_last_of('/') + 1)});
        if (one.empty() || one[0].parts) {
          ++skipped;
          continue;
        }
        Installed it;
        it.id = models::tilde(g);
        it.path = g;
        keep(std::move(it), g, "", "");
        if (!out.empty() && out.back().path == g) {
          std::vector<std::string> used = {g};
          const std::string dir = dirname_of(g);
          for (const std::string& f : entries(dir)) {
            const std::string l = lower(f);
            if ((starts_with(l, "mmproj") || starts_with(l, "mtp")) && ends_with(l, ".gguf")) used.push_back(dir + "/" + f);
          }
          out.back().disk = bytes_of(used);
        }
      }
    }
  }
  if (other) *other = skipped;
  return out;
}

Removal plan_removal(const std::string& cache, const Ref& ref) {
  const std::string id = ref.repo + (ref.quant.empty() ? "" : ":" + ref.quant);
  Removal r;
  r.repo_dir = repo_dir(cache, ref);
  struct stat st;
  if (::lstat(r.repo_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode))
    throw Error(id + " is not in the Hugging Face cache (" + cache + ")");
  const std::string rc = real(cache), rr = real(r.repo_dir);
  if (rc.empty() || rr != rc + "/" + r.repo_dir.substr(r.repo_dir.find_last_of('/') + 1))
    throw Error(r.repo_dir + " is not a directory of the Hugging Face cache (" + cache + "); nothing was removed");
  const std::vector<std::string> snaps = snapshots(r.repo_dir);
  if (ref.quant.empty()) {
    r.whole = true;
    r.bytes = tree_bytes(r.repo_dir);
    for (const std::string& snap : snaps) {
      std::vector<std::string> files;
      walk_links(snap, "", 3, files);
      for (const std::string& f : files) r.files.push_back(snap.substr(snap.find_last_of('/') + 1, 12) + "/" + f);
    }
    return r;
  }
  // The GGUF model's files in every snapshot; a snapshot left without a model loses everything.
  std::map<std::string, int> links_to;  // blob -> links to it, every snapshot
  std::map<std::string, int> removed;   // blob -> removed links to it
  size_t total = 0;
  std::set<std::string> shown;
  for (const std::string& snap : snaps) {
    std::vector<std::string> files;
    walk_links(snap, "", 3, files);
    total += files.size();
    for (const std::string& f : files)
      if (const std::string b = blob_of(snap + "/" + f); !b.empty()) ++links_to[b];
    GgufPick p;
    if (!pick_gguf(files, ref, p)) continue;  // throws when the selector names several models
    const std::vector<std::string> go = model_groups(files).size() == 1 ? files : std::vector<std::string>{p.model};
    for (const std::string& f : go) {
      r.links.push_back(snap + "/" + f);
      if (const std::string b = blob_of(snap + "/" + f); !b.empty()) ++removed[b];
      if (shown.insert(f).second) r.files.push_back(f);
    }
  }
  if (r.links.empty()) throw Error("no GGUF model of " + ref.repo + " matching \"" + ref.quant + "\" is in the Hugging Face cache");
  if (r.links.size() == total) {  // nothing is left: the repository goes
    r.whole = true;
    r.links.clear();
    r.bytes = tree_bytes(r.repo_dir);
    return r;
  }
  for (const auto& [b, n] : removed)
    if (links_to[b] == n) {
      const std::string path = r.repo_dir + "/blobs/" + b;
      if (::lstat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
        r.blobs.push_back(path);
        r.bytes += st.st_size;
      }
    }
  for (const std::string& l : r.links)  // plain files in a snapshot (no blob)
    if (::lstat(l.c_str(), &st) == 0 && S_ISREG(st.st_mode)) r.bytes += st.st_size;
  return r;
}

void remove(const std::string& cache, const Removal& r) {
  const std::string rc = real(cache), name = r.repo_dir.substr(r.repo_dir.find_last_of('/') + 1);
  if (rc.empty() || name.rfind("models--", 0) != 0 || real(r.repo_dir) != rc + "/" + name)
    throw Error(r.repo_dir + " is not a directory of the Hugging Face cache (" + cache + "); nothing was removed");
  if (r.whole) {
    remove_tree(r.repo_dir);
    if (is_dir(cache + "/.locks/" + name)) remove_tree(cache + "/.locks/" + name);
    return;
  }
  const std::string prefix = r.repo_dir + "/snapshots/";
  for (const std::string& l : r.links) {
    struct stat st;
    if (l.compare(0, prefix.size(), prefix) != 0 || l.find("/../") != std::string::npos) continue;
    if (::lstat(l.c_str(), &st) == 0 && (S_ISLNK(st.st_mode) || S_ISREG(st.st_mode)) && ::unlink(l.c_str()) != 0)
      throw Error("cannot remove " + l + ": " + std::strerror(errno));
    // Directories left empty go, up to snapshots/.
    for (std::string d = dirname_of(l); d.size() > prefix.size() - 1 && ::rmdir(d.c_str()) == 0; d = dirname_of(d)) {
    }
  }
  for (const std::string& b : r.blobs) {
    struct stat st;
    if (b.compare(0, r.repo_dir.size() + 7, r.repo_dir + "/blobs/") != 0) continue;
    if (::lstat(b.c_str(), &st) == 0 && S_ISREG(st.st_mode) && ::unlink(b.c_str()) != 0)
      throw Error("cannot remove " + b + ": " + std::strerror(errno));
  }
  // Refs to revisions whose snapshot is gone go too.
  for (const std::string& n : entries(r.repo_dir + "/refs")) {
    std::string commit = read_text(r.repo_dir + "/refs/" + n);
    while (!commit.empty() && std::isspace(static_cast<unsigned char>(commit.back()))) commit.pop_back();
    if (!commit.empty() && !is_dir(r.repo_dir + "/snapshots/" + commit)) ::unlink((r.repo_dir + "/refs/" + n).c_str());
  }
}

}  // namespace espetero::hub
