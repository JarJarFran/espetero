// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The Hugging Face Hub over HTTPS, natively: a repository's files at a
// revision with their sizes and blob ids (the API), byte ranges of its files
// (headers for the memory check) and downloads into the hub cache,
// with Foundation's NSURLSession: an ephemeral session per request, no
// cookies, cache or stored credentials. No Python and no `hf` CLI.
//
//   HF_ENDPOINT   the Hub (default https://huggingface.co)
//   HF_TOKEN      a token for gated and private repositories (also
//                 HUGGING_FACE_HUB_TOKEN, or the token file: HF_TOKEN_PATH,
//                 else $HF_HOME/token, else ~/.cache/huggingface/token),
//                 sent only to the endpoint's host and never printed
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "hub.h"

namespace espetero::hub {

// HF_ENDPOINT without a trailing '/', else https://huggingface.co.
std::string endpoint();

// The user's Hugging Face token ("" when none).
std::string token();

struct RepoFile {
  std::string path;  // relative path in the repository
  int64_t size = 0;
  std::string blob;  // its name in the cache's blobs/: an LFS file's sha256, else its git blob id
  bool lfs = false;
};

struct RepoInfo {
  std::string repo, revision, commit;
  std::string gated;  // "" (open), "auto" or "manual"
  std::vector<RepoFile> files;
  const RepoFile* find(const std::string& path) const;
};

// GET /api/models/{repo}/revision/{revision}?blobs=true. Throws Error with
// the reason (not found or private, gated, no network, ...).
RepoInfo repo_info(const std::string& repo, const std::string& revision = "main");

// `length` bytes at `offset` of a repository file at info.commit (an HTTP
// range of /{repo}/resolve/{commit}/{path}). Throws Error.
std::string read_range(const RepoInfo& info, const std::string& path, int64_t offset, int64_t length);

// Downloads `paths` of info.commit into the hub cache `cache` exactly
// as huggingface_hub lays it out, so hf, transformers, mlx-lm and Espetero
// share one copy:
//   models--ORG--NAME/blobs/<blob>             the content: an LFS file's sha256, else its git blob id
//   models--ORG--NAME/snapshots/<commit>/<path> a relative link to ../../blobs/<blob>
//   models--ORG--NAME/refs/<revision>          the commit (when the revision is a name)
// A blob is written as blobs/<blob>.incomplete, resumed with an HTTP range
// after an interruption, verified (sha256 for LFS files, the git blob id
// otherwise) and renamed; .locks/models--ORG--NAME/<blob>.lock (flock, as
// huggingface_hub's filelock) keeps two processes from writing the same blob.
// Files whose blob is there already only get their link. Up to 4 files at a
// time, a progress line on stderr. Refused before any file (or ref) is
// written when the disk lacks the space. Throws Error.
void download(const std::string& cache, const RepoInfo& info, const std::vector<std::string>& paths,
              const std::string& label, bool quiet = false);

// ---------------------------------------------------------------- HTTP
struct HttpResponse {
  int status = 0;
  std::map<std::string, std::string> headers;  // lower-case names
  std::string body;
  std::string error;  // transport failure ("" when a response arrived)
  std::string header(const std::string& lower_name) const;
};

using Headers = std::vector<std::pair<std::string, std::string>>;

// GET `url` with `headers` (and the token when the host is the endpoint's),
// following up to 10 redirects: never from https to http, and the token is
// dropped when a redirect leaves the endpoint's host. `on_response` sees the
// final status and headers and `on_data` each piece of the body; either
// returns false to stop the transfer (not an error). `idle_s`: the longest
// wait for data.
HttpResponse http_stream(const std::string& url, const Headers& headers,
                         const std::function<bool(const HttpResponse&)>& on_response,
                         const std::function<bool(const char*, size_t)>& on_data, double idle_s = 60);

// GET with the body kept up to `max_body` bytes (the transfer stops there).
HttpResponse http_get(const std::string& url, const Headers& headers, size_t max_body, double idle_s = 60);

// The URL of a repository file: {endpoint}/{repo}/resolve/{revision}/{path}.
std::string resolve_url(const std::string& repo, const std::string& revision, const std::string& path);

}  // namespace espetero::hub
