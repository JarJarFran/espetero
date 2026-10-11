// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Where the engine keeps what it writes outside the model directory.
#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <string>

namespace espetero::paths {

inline std::string home() {
  const char* h = std::getenv("HOME");
  return h && *h ? std::string(h) : std::string("/tmp");
}

inline bool is_dir(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// What the engine writes for the user (Batch API requests and results, chat
// checkpoints, caches) is readable by this user only, whatever the umask:
// directories it creates are 0700 (existing ones keep their mode) and files
// it writes 0600.
//
// mkdir -p: every missing component of `dir` is created 0700. True when `dir`
// is a directory afterwards.
inline bool make_private_dirs(const std::string& dir) {
  for (size_t i = 1; i <= dir.size(); ++i) {
    if (i < dir.size() && dir[i] != '/') continue;
    const std::string part = dir.substr(0, i);
    if (::mkdir(part.c_str(), 0700) == 0) ::chmod(part.c_str(), 0700);  // a umask can only clear bits; chmod sets them all
  }
  return is_dir(dir);
}

// Creates `path` (or keeps it) with mode 0600, for writers that then open it
// themselves (std::ofstream and fopen keep the mode of an existing file).
// Returns false when the file cannot be created.
inline bool make_private_file(const std::string& path) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0600);
  if (fd < 0) return false;
  ::fchmod(fd, 0600);
  ::close(fd);
  return true;
}

// Small files: GEMM autotune results, Batch API storage, persisted chat checkpoints.
inline std::string user_cache_dir() {
  return home() + "/.cache/espetero";
}

// Large stores (expert records): ~/Library/Caches is outside Time Machine backups.
inline std::string user_library_cache_dir() {
  return home() + "/Library/Caches/espetero";
}

}  // namespace espetero::paths
