// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// OpenAI-compatible Batch API: uploaded JSONL files of requests, batch jobs
// that run them through the engine at low priority (continuous batching with
// interactive traffic), results as JSONL output/error files. State lives in a
// data directory, so finished batches survive restarts and unfinished ones
// resume (requests whose custom_id already has a result are skipped).
//
// Bounded for untrusted clients: at most kRunners batches run at once (each
// with up to `concurrency` request threads), the rest wait in a queue of at
// most kMaxQueued; the data directory holds at most disk_gb (uploads beyond
// it get a 413); a batch that fails to run is marked failed, never the server.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../util/json.h"
#include "engine.h"

namespace espetero {

class BatchService {
 public:
  static constexpr int kRunners = 2;      // batches running at once
  static constexpr int kMaxQueued = 100;  // batches waiting for a runner: 429 beyond

  BatchService(Engine& engine, std::string data_dir, int concurrency, double disk_gb = 4);
  ~BatchService();  // shutdown(), then joins the runners

  // Stops the runners taking lines: a running batch stops once its
  // requests in flight end (the engine's shutdown ends them) and resumes at
  // the next start; a line that fails because the server is stopping is not
  // recorded, so it runs again then. Thread-safe.
  void shutdown();

  // Files.
  json::Value upload(const std::string& filename, const std::string& purpose, const std::string& bytes);
  json::Value file_info(const std::string& id) const;   // throws RequestError(404)
  std::string file_content(const std::string& id) const;
  json::Value list_files() const;
  json::Value delete_file(const std::string& id);

  // Batches.
  json::Value create(const json::Value& body);
  json::Value get(const std::string& id) const;
  json::Value list(int limit, const std::string& after) const;
  json::Value cancel(const std::string& id);

  // Run one JSONL file synchronously (CLI): returns (completed, failed).
  std::pair<int, int> run_file(const std::string& input_path, const std::string& output_path,
                               const std::string& errors_path, const std::string& default_endpoint);

  // One request of a batch: (status, response body).
  static std::pair<int, json::Value> execute(Engine& engine, const std::string& endpoint, const json::Value& body);

 private:
  struct Batch;
  void save_batch(const Batch& b) const;
  void run(std::shared_ptr<Batch> b);
  void fail(Batch& b, const std::string& code, const std::string& message);  // with mu_ held
  void enqueue(std::shared_ptr<Batch> b);                                    // with mu_ held
  void runner();
  uint64_t disk_used() const;  // bytes in the data directory
  json::Value to_json(const Batch& b) const;
  std::string path_of(const std::string& id, const char* ext) const;

  Engine& engine_;
  std::string dir_;
  int concurrency_;
  uint64_t disk_quota_;
  mutable std::mutex mu_;
  std::map<std::string, json::Value> files_;  // id -> metadata
  std::map<std::string, std::shared_ptr<Batch>> batches_;
  std::deque<std::shared_ptr<Batch>> queue_;  // waiting for a runner
  std::condition_variable queue_cv_;
  int idle_runners_ = 0;
  std::vector<std::thread> runners_;          // at most kRunners
  std::atomic<bool> stop_{false};
};

}  // namespace espetero
