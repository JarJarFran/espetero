// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "batch.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <system_error>

#include "../util/paths.h"
#include "chat.h"
#include "embed.h"

namespace espetero {

namespace {

int64_t now_s() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string random_id(const char* prefix) {
  static std::mutex mu;
  static std::mt19937_64 rng{std::random_device{}()};
  static const char* h = "0123456789abcdef";
  std::lock_guard<std::mutex> lk(mu);
  std::string s = prefix;
  for (int i = 0; i < 24; ++i) s.push_back(h[rng() & 15]);
  return s;
}

bool read_all(const std::string& path, std::string& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::stringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}

// Writes a file of this user only (0600) through a temporary file; false when
// it could not be written whole (the old contents then stay).
bool write_all(const std::string& path, const std::string& data) {
  const std::string tmp = path + ".tmp";
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (fd < 0) return false;
  ::fchmod(fd, 0600);
  size_t done = 0;
  while (done < data.size()) {
    const ssize_t n = ::write(fd, data.data() + done, data.size() - done);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) break;
    done += size_t(n);
  }
  if (::close(fd) != 0 || done != data.size()) {
    ::unlink(tmp.c_str());
    return false;
  }
  return std::rename(tmp.c_str(), path.c_str()) == 0;
}

bool valid_id(const std::string& id) {
  if (id.empty() || id.size() > 64) return false;
  for (char c : id)
    if (!(isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')) return false;
  return true;
}

json::Value error_body(const std::string& msg, const char* type = "invalid_request_error") {
  json::Value e = json::Value::object();
  e.set("message", msg);
  e.set("type", type);
  json::Value o = json::Value::object();
  o.set("error", std::move(e));
  return o;
}

}  // namespace

struct BatchService::Batch {
  json::Value meta;  // the OpenAI batch object
  // Result files. They fill up while the batch runs but are published in the
  // batch object (output_file_id / error_file_id) only once it is final.
  std::string out_id, err_id;
  std::atomic<bool> cancel{false};
  std::atomic<int> completed{0}, failed{0};
  int total = 0;
};

BatchService::BatchService(Engine& engine, std::string data_dir, int concurrency, double disk_gb)
    : engine_(engine),
      dir_(std::move(data_dir)),
      concurrency_(std::max(1, concurrency)),
      disk_quota_(uint64_t(std::max(0.0, disk_gb) * 1e9)) {
  paths::make_private_dirs(dir_);
  // Reload files and batches; resume unfinished batches.
  if (DIR* d = ::opendir(dir_.c_str())) {
    while (dirent* e = ::readdir(d)) {
      const std::string name = e->d_name;
      if (name.size() < 6 || name.compare(name.size() - 5, 5, ".json") != 0) continue;
      std::string txt;
      if (!read_all(dir_ + "/" + name, txt)) continue;
      try {
        json::Value v = json::parse(txt);
        const std::string id = v.get_string("id", "");
        if (v.get_string("object", "") == "file") files_[id] = v;
        else if (v.get_string("object", "") == "batch") {
          auto b = std::make_shared<Batch>();
          b->out_id = v.get_string("x_output_file", "");
          b->err_id = v.get_string("x_error_file", "");
          v.as_object().erase("x_output_file");
          v.as_object().erase("x_error_file");
          b->meta = v;
          if (const json::Value* rc = v.get("request_counts")) {
            b->total = int(rc->get_int("total", 0));
            b->completed = int(rc->get_int("completed", 0));
            b->failed = int(rc->get_int("failed", 0));
          }
          batches_[id] = b;
        }
      } catch (...) {
      }
    }
    ::closedir(d);
  }
  // Unfinished batches resume in the order they were created.
  std::vector<std::shared_ptr<Batch>> resume;
  for (auto& [id, b] : batches_) {
    const std::string st = b->meta.get_string("status", "");
    if (st == "validating" || st == "in_progress" || st == "finalizing" || st == "cancelling") {
      b->cancel = st == "cancelling";  // a cancelled batch just publishes what it has
      resume.push_back(b);
    }
  }
  std::stable_sort(resume.begin(), resume.end(), [](const auto& a, const auto& b) {
    return a->meta.get_int("created_at", 0) < b->meta.get_int("created_at", 0);
  });
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& b : resume) enqueue(b);
}

BatchService::~BatchService() {
  shutdown();
  for (auto& t : runners_)
    if (t.joinable()) t.join();
}

void BatchService::shutdown() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    stop_ = true;
    for (auto& [id, b] : batches_) b->cancel = true;
  }
  queue_cv_.notify_all();
}

// ---------------------------------------------------------------- runners
// Queues a batch for the runners, starting one when none is idle (at most
// kRunners). A runner that cannot be started leaves the batch to the others,
// or fails it when there are none.
void BatchService::enqueue(std::shared_ptr<Batch> b) {
  queue_.push_back(b);
  if (idle_runners_ == 0 && int(runners_.size()) < kRunners) {
    try {
      runners_.emplace_back([this] { runner(); });
    } catch (const std::system_error& e) {
      if (runners_.empty()) {
        queue_.pop_back();
        fail(*b, "server_error", std::string("cannot start a batch runner: ") + e.what());
        return;
      }
    }
  }
  queue_cv_.notify_one();
}

void BatchService::runner() {
  std::unique_lock<std::mutex> lk(mu_);
  for (;;) {
    ++idle_runners_;
    queue_cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
    --idle_runners_;
    if (stop_) return;
    std::shared_ptr<Batch> b = queue_.front();
    queue_.pop_front();
    lk.unlock();
    std::string error;
    try {
      run(b);
    } catch (const std::exception& e) {  // out of memory, threads or disk: the batch fails, not the server
      error = e.what();
    } catch (...) {
      error = "unknown error";
    }
    lk.lock();
    if (!error.empty()) {
      std::fprintf(stderr, "[espetero] batch %s failed: %s\n", b->meta.get_string("id", "").c_str(), error.c_str());
      fail(*b, "server_error", "the batch failed: " + error);
    }
  }
}

void BatchService::fail(Batch& b, const std::string& code, const std::string& message) {
  json::Value e = json::Value::object();
  e.set("code", code);
  e.set("message", message);
  json::Value data = json::Value::array();
  data.push_back(std::move(e));
  json::Value errs = json::Value::object();
  errs.set("object", "list");
  errs.set("data", std::move(data));
  b.meta.set("errors", std::move(errs));
  b.meta.set("status", "failed");
  b.meta.set("failed_at", now_s());
  save_batch(b);
}

uint64_t BatchService::disk_used() const {
  uint64_t total = 0;
  if (DIR* d = ::opendir(dir_.c_str())) {
    while (dirent* e = ::readdir(d)) {
      struct stat st;
      if (::stat((dir_ + "/" + e->d_name).c_str(), &st) == 0 && S_ISREG(st.st_mode)) total += uint64_t(st.st_size);
    }
    ::closedir(d);
  }
  return total;
}

std::string BatchService::path_of(const std::string& id, const char* ext) const { return dir_ + "/" + id + ext; }

// ---------------------------------------------------------------- files
json::Value BatchService::upload(const std::string& filename, const std::string& purpose, const std::string& bytes) {
  if (purpose.empty()) throw RequestError(422, "purpose: required");
  // One upload at a time, so that two cannot both pass the quota check.
  static std::mutex upload_mu;
  std::lock_guard<std::mutex> ul(upload_mu);
  const uint64_t used = disk_used();
  if (used + bytes.size() > disk_quota_) {
    char msg[256];
    std::snprintf(msg, sizeof msg,
                  "Batch storage is full: %.2f GB used of %.2f GB (--batch-disk-gb), this file needs %.2f GB more; "
                  "delete files you no longer need (DELETE /v1/files/{id})",
                  double(used) / 1e9, double(disk_quota_) / 1e9, double(bytes.size()) / 1e9);
    throw RequestError(413, msg);
  }
  const std::string id = random_id("file-");
  if (!write_all(path_of(id, ".jsonl"), bytes)) throw RequestError(500, "cannot store the file");
  json::Value f = json::Value::object();
  f.set("id", id);
  f.set("object", "file");
  f.set("bytes", int64_t(bytes.size()));
  f.set("created_at", now_s());
  f.set("filename", filename.empty() ? "upload.jsonl" : filename);
  f.set("purpose", purpose);
  f.set("status", "processed");
  if (!write_all(path_of(id, ".json"), json::dump(f))) {
    ::unlink(path_of(id, ".jsonl").c_str());
    throw RequestError(500, "cannot store the file");
  }
  std::lock_guard<std::mutex> lk(mu_);
  files_[id] = f;
  return f;
}

json::Value BatchService::file_info(const std::string& id) const {
  if (!valid_id(id)) throw RequestError(404, "No such file: " + id);  // ids name files on disk
  std::lock_guard<std::mutex> lk(mu_);
  auto it = files_.find(id);
  if (it == files_.end()) throw RequestError(404, "No such file: " + id);
  return it->second;
}

std::string BatchService::file_content(const std::string& id) const {
  (void)file_info(id);
  std::string out;
  if (!read_all(path_of(id, ".jsonl"), out)) throw RequestError(404, "file content missing: " + id);
  return out;
}

json::Value BatchService::list_files() const {
  std::lock_guard<std::mutex> lk(mu_);
  json::Value data = json::Value::array();
  for (auto& [id, f] : files_) data.push_back(f);
  json::Value o = json::Value::object();
  o.set("object", "list");
  o.set("data", std::move(data));
  return o;
}

json::Value BatchService::delete_file(const std::string& id) {
  (void)file_info(id);
  {
    std::lock_guard<std::mutex> lk(mu_);
    files_.erase(id);
  }
  std::remove(path_of(id, ".jsonl").c_str());
  std::remove(path_of(id, ".json").c_str());
  json::Value o = json::Value::object();
  o.set("id", id);
  o.set("object", "file");
  o.set("deleted", true);
  return o;
}

// ---------------------------------------------------------------- execution
std::pair<int, json::Value> BatchService::execute(Engine& engine, const std::string& endpoint, const json::Value& body) {
  try {
    if (endpoint == "/v1/decisions" || endpoint == "/v1/systemone")
      return {200, engine.is_clef() && !body.get("context") ? engine.systemone(body) : engine.decide(body)};
    if (endpoint == "/v1/embeddings") return {200, embeddings_response(engine, body)};
    if (endpoint == "/v1/chat/completions" || endpoint == "/v1/completions") {
      if (engine.is_clef())
        throw RequestError(422, "this is a Clef decision model: it answers /v1/systemone and /v1/decisions, not chat");
      ChatJob job = endpoint == "/v1/chat/completions" ? prepare_chat(engine, body) : prepare_completion(engine, body);
      job.stream = false;
      job.gen.priority = std::max(job.gen.priority, 1);  // below interactive traffic
      return {200, respond(engine, job)};  // n choices too
    }
    return {404, error_body("unsupported endpoint " + endpoint)};
  } catch (const RequestError& e) {
    return {e.status, error_body(e.what(), e.status >= 500 ? "server_error" : "invalid_request_error")};
  } catch (const std::exception& e) {
    return {500, error_body(e.what(), "server_error")};
  }
}

namespace {

// Requests of a JSONL batch file: (custom_id, method, url, body) or a parse error.
struct Line {
  std::string custom_id, url, error;
  json::Value body;
};

std::vector<Line> parse_lines(const std::string& text, const std::string& default_url) {
  std::vector<Line> out;
  size_t pos = 0, n = 0;
  while (pos < text.size()) {
    size_t e = text.find('\n', pos);
    if (e == std::string::npos) e = text.size();
    std::string line = text.substr(pos, e - pos);
    pos = e + 1;
    ++n;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    bool blank = true;
    for (char c : line) blank = blank && (c == ' ' || c == '\t');
    if (blank) continue;
    Line l;
    try {
      json::Value v = json::parse(line);
      l.custom_id = v.get_string("custom_id", "line-" + std::to_string(n));
      l.url = v.get_string("url", default_url);
      const std::string method = v.get_string("method", "POST");
      if (method != "POST") l.error = "only POST requests are supported";
      if (const json::Value* b = v.get("body")) l.body = *b;
      else l.error = "missing body";
    } catch (const std::exception& ex) {
      l.custom_id = "line-" + std::to_string(n);
      l.error = std::string("invalid JSON: ") + ex.what();
    }
    out.push_back(std::move(l));
  }
  return out;
}

// Runs lines with a pool of workers; results go through `emit` (thread-safe).
// When the system has no threads left the pool is smaller (or the caller's
// thread does the work); a line whose result cannot be written is logged and
// skipped (it runs again when the batch resumes).
void run_lines(Engine& engine, std::vector<Line>& lines, int concurrency, const std::atomic<bool>& cancel,
               const std::function<void(const Line&, int, const json::Value&)>& emit) {
  std::atomic<size_t> next{0};
  auto work = [&] {
    while (!cancel) {
      const size_t i = next++;
      if (i >= lines.size()) break;
      Line& l = lines[i];
      try {
        if (!l.error.empty()) {
          emit(l, 400, error_body(l.error));
          continue;
        }
        auto [status, body] = BatchService::execute(engine, l.url, l.body);
        emit(l, status, body);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "[espetero] batch request %s: %s\n", l.custom_id.c_str(), e.what());
      }
    }
  };
  std::vector<std::thread> pool;
  const int n = std::max(1, std::min<int>(concurrency, int(lines.size())));
  for (int w = 0; w < n; ++w) {
    try {
      pool.emplace_back(work);
    } catch (const std::system_error&) {
      break;
    }
  }
  if (pool.empty()) work();
  for (auto& t : pool) t.join();
}

json::Value result_line(const Line& l, int status, const json::Value& body) {
  json::Value resp = json::Value::object();
  resp.set("status_code", status);
  resp.set("request_id", random_id("req_"));
  resp.set("body", body);
  json::Value o = json::Value::object();
  o.set("id", random_id("batch_req_"));
  o.set("custom_id", l.custom_id);
  o.set("response", std::move(resp));
  o.set("error", json::Value());
  return o;
}

}  // namespace

std::pair<int, int> BatchService::run_file(const std::string& input_path, const std::string& output_path,
                                           const std::string& errors_path, const std::string& default_endpoint) {
  std::string text;
  if (!read_all(input_path, text)) throw std::runtime_error("cannot read " + input_path);
  std::vector<Line> lines = parse_lines(text, default_endpoint);
  std::ofstream out(output_path, std::ios::binary | std::ios::trunc), err;
  if (!errors_path.empty()) err.open(errors_path, std::ios::binary | std::ios::trunc);
  std::mutex mu;
  int ok = 0, bad = 0;
  std::atomic<bool> cancel{false};
  run_lines(engine_, lines, concurrency_, cancel, [&](const Line& l, int status, const json::Value& body) {
    const std::string s = json::dump(result_line(l, status, body)) + "\n";
    std::lock_guard<std::mutex> lk(mu);
    if (status == 200) {
      out << s;
      ++ok;
    } else {
      (err.is_open() ? err : out) << s;
      ++bad;
    }
    out.flush();
  });
  return {ok, bad};
}

// ---------------------------------------------------------------- batches
json::Value BatchService::to_json(const Batch& b) const {
  json::Value v = b.meta;
  json::Value rc = json::Value::object();
  rc.set("total", b.total);
  rc.set("completed", b.completed.load());
  rc.set("failed", b.failed.load());
  v.set("request_counts", std::move(rc));
  return v;
}

void BatchService::save_batch(const Batch& b) const {
  json::Value v = to_json(b);
  v.set("x_output_file", b.out_id);
  v.set("x_error_file", b.err_id);
  write_all(path_of(b.meta.get_string("id", ""), ".json"), json::dump(v));
}

json::Value BatchService::create(const json::Value& body) {
  if (!body.is_object()) throw RequestError(422, "request body must be a JSON object");
  const std::string input = body.get_string("input_file_id", "");
  const std::string endpoint = body.get_string("endpoint", "");
  if (endpoint != "/v1/chat/completions" && endpoint != "/v1/completions" && endpoint != "/v1/decisions" &&
      endpoint != "/v1/systemone" && endpoint != "/v1/embeddings")
    throw RequestError(422, "endpoint: /v1/chat/completions, /v1/completions, /v1/embeddings, /v1/decisions or /v1/systemone");
  (void)file_info(input);
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (queue_.size() >= size_t(kMaxQueued))
      throw RequestError(429, "Too many batches waiting to run (" + std::to_string(kMaxQueued) +
                                  "); create this one when some have finished");
  }
  if (disk_used() >= disk_quota_)
    throw RequestError(413, "Batch storage is full (--batch-disk-gb): delete files you no longer need "
                            "(DELETE /v1/files/{id}) before creating a batch");
  auto b = std::make_shared<Batch>();
  const std::string id = random_id("batch_");
  json::Value m = json::Value::object();
  m.set("id", id);
  m.set("object", "batch");
  m.set("endpoint", endpoint);
  m.set("errors", json::Value());
  m.set("input_file_id", input);
  m.set("completion_window", body.get_string("completion_window", "24h"));
  m.set("status", "validating");
  m.set("output_file_id", json::Value());
  m.set("error_file_id", json::Value());
  m.set("created_at", now_s());
  m.set("in_progress_at", json::Value());
  m.set("expires_at", now_s() + 24 * 3600);
  m.set("finalizing_at", json::Value());
  m.set("completed_at", json::Value());
  m.set("failed_at", json::Value());
  m.set("expired_at", json::Value());
  m.set("cancelling_at", json::Value());
  m.set("cancelled_at", json::Value());
  m.set("metadata", body.get("metadata") ? *body.get("metadata") : json::Value());
  b->meta = std::move(m);
  std::lock_guard<std::mutex> lk(mu_);
  save_batch(*b);
  batches_[id] = b;
  enqueue(b);
  return to_json(*b);
}

void BatchService::run(std::shared_ptr<Batch> b) {
  const std::string id = b->meta.get_string("id", "");
  const std::string endpoint = b->meta.get_string("endpoint", "");
  std::string text;
  const std::string input = b->meta.get_string("input_file_id", "");
  if (!read_all(path_of(input, ".jsonl"), text)) {
    std::lock_guard<std::mutex> lk(mu_);
    fail(*b, "missing_input", "input file not found");
    return;
  }
  std::vector<Line> lines = parse_lines(text, endpoint);
  // Output files: continue existing ones (resume) or create them.
  if (b->out_id.empty()) b->out_id = random_id("file-");
  if (b->err_id.empty()) b->err_id = random_id("file-");
  const std::string out_id = b->out_id, err_id = b->err_id;
  // Counts come from the result files themselves (the saved counts may lag).
  std::set<std::string> done_ids;
  int prev[2] = {0, 0};
  for (int k = 0; k < 2; ++k) {
    std::string txt;
    if (!read_all(path_of(k ? err_id : out_id, ".jsonl"), txt)) continue;
    for (const Line& l : parse_lines(txt, endpoint))
      if (!l.custom_id.empty() && done_ids.insert(l.custom_id).second) ++prev[k];
  }
  b->completed = prev[0];
  b->failed = prev[1];
  std::vector<Line> todo;
  for (auto& l : lines)
    if (!done_ids.count(l.custom_id)) todo.push_back(std::move(l));
  {
    std::lock_guard<std::mutex> lk(mu_);
    b->total = int(lines.size());
    if (!b->cancel) b->meta.set("status", "in_progress");
    if (b->meta.get("in_progress_at")->is_null()) b->meta.set("in_progress_at", now_s());
    save_batch(*b);
  }
  if (!paths::make_private_file(path_of(out_id, ".jsonl")) || !paths::make_private_file(path_of(err_id, ".jsonl")))
    throw std::runtime_error("cannot create its result files");
  std::ofstream out(path_of(out_id, ".jsonl"), std::ios::binary | std::ios::app);
  std::ofstream err(path_of(err_id, ".jsonl"), std::ios::binary | std::ios::app);
  std::mutex wmu;
  auto last_save = std::chrono::steady_clock::now();
  run_lines(engine_, todo, concurrency_, b->cancel, [&](const Line& l, int status, const json::Value& body) {
    // A request that failed while the server stops (the engine ended
    // it) is not its result: unrecorded, it runs again when the batch resumes.
    if (status != 200 && stop_) return;
    const std::string s = json::dump(result_line(l, status, body)) + "\n";
    std::lock_guard<std::mutex> lk(wmu);
    if (status == 200) {
      out << s;
      out.flush();
      ++b->completed;
    } else {
      err << s;
      err.flush();
      ++b->failed;
    }
    if (std::chrono::steady_clock::now() - last_save > std::chrono::seconds(2)) {
      std::lock_guard<std::mutex> mk(mu_);
      save_batch(*b);
      last_save = std::chrono::steady_clock::now();
    }
  });
  out.close();
  err.close();
  std::lock_guard<std::mutex> lk(mu_);
  auto file_meta = [&](const std::string& fid, const char* name) {
    struct stat st{};
    ::stat(path_of(fid, ".jsonl").c_str(), &st);
    json::Value f = json::Value::object();
    f.set("id", fid);
    f.set("object", "file");
    f.set("bytes", int64_t(st.st_size));
    f.set("created_at", now_s());
    f.set("filename", std::string(name) + "-" + id + ".jsonl");
    f.set("purpose", "batch_output");
    f.set("status", "processed");
    write_all(path_of(fid, ".json"), json::dump(f));
    files_[fid] = f;
  };
  if (stop_) {  // shutting down: resume later
    save_batch(*b);
    return;
  }
  // Publish the result files: outputs always, errors only when some failed.
  file_meta(out_id, "output");
  b->meta.set("output_file_id", out_id);
  if (b->failed > 0) {
    file_meta(err_id, "errors");
    b->meta.set("error_file_id", err_id);
  } else {
    ::unlink(path_of(err_id, ".jsonl").c_str());
  }
  if (b->cancel) {
    b->meta.set("status", "cancelled");
    b->meta.set("cancelled_at", now_s());
  } else {
    b->meta.set("finalizing_at", now_s());
    b->meta.set("status", "completed");
    b->meta.set("completed_at", now_s());
  }
  save_batch(*b);
}

json::Value BatchService::get(const std::string& id) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = batches_.find(id);
  if (it == batches_.end()) throw RequestError(404, "No such batch: " + id);
  return to_json(*it->second);
}

json::Value BatchService::list(int limit, const std::string& after) const {
  std::lock_guard<std::mutex> lk(mu_);
  std::vector<std::shared_ptr<Batch>> all;
  for (auto& [id, b] : batches_) all.push_back(b);
  std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) {
    return a->meta.get_int("created_at", 0) > b->meta.get_int("created_at", 0);
  });
  json::Value data = json::Value::array();
  bool started = after.empty();
  bool more = false;
  for (auto& b : all) {
    const std::string id = b->meta.get_string("id", "");
    if (!started) {
      started = id == after;
      continue;
    }
    if (int(data.size()) >= limit) {
      more = true;
      break;
    }
    data.push_back(to_json(*b));
  }
  json::Value o = json::Value::object();
  o.set("object", "list");
  o.set("first_id", data.size() ? *data[0].get("id") : json::Value());
  o.set("last_id", data.size() ? *data[data.size() - 1].get("id") : json::Value());
  o.set("has_more", more);
  o.set("data", std::move(data));
  return o;
}

json::Value BatchService::cancel(const std::string& id) {
  std::shared_ptr<Batch> b;
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = batches_.find(id);
    if (it == batches_.end()) throw RequestError(404, "No such batch: " + id);
    b = it->second;
    const std::string st = b->meta.get_string("status", "");
    if (st == "validating" || st == "in_progress") {
      b->cancel = true;
      const auto q = std::find(queue_.begin(), queue_.end(), b);
      if (q != queue_.end() && st == "validating") {  // never started: nothing to publish
        queue_.erase(q);
        b->meta.set("status", "cancelled");
        b->meta.set("cancelled_at", now_s());
      } else {
        b->meta.set("status", "cancelling");
        b->meta.set("cancelling_at", now_s());
      }
      save_batch(*b);
    }
  }
  return get(id);
}

}  // namespace espetero
