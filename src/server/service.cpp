// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The configuration file and the LaunchAgent; see service.h.
#include "service.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "../util/json.h"
#include "../util/paths.h"

extern char** environ;

namespace espetero::service {

namespace {

std::string read_file(const std::string& path, bool& found) {
  std::ifstream f(path, std::ios::binary);
  found = bool(f);
  std::stringstream ss;
  if (found) ss << f.rdbuf();
  return ss.str();
}

// Written whole or not at all (a temporary file renamed over the old one),
// with `mode` for a new file (an existing one keeps its own).
void write_file(const std::string& path, const std::string& data, mode_t mode) {
  struct stat st;
  const mode_t m = ::stat(path.c_str(), &st) == 0 ? (st.st_mode & 07777) : mode;
  const std::string tmp = path + ".tmp";
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, m);
  if (fd < 0) throw std::runtime_error("cannot write " + tmp + ": " + std::strerror(errno));
  ::fchmod(fd, m);
  size_t done = 0;
  while (done < data.size()) {
    const ssize_t n = ::write(fd, data.data() + done, data.size() - done);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) {
      ::close(fd);
      ::unlink(tmp.c_str());
      throw std::runtime_error("cannot write " + tmp + ": " + std::strerror(errno));
    }
    done += size_t(n);
  }
  ::close(fd);
  if (::rename(tmp.c_str(), path.c_str()) != 0) {
    ::unlink(tmp.c_str());
    throw std::runtime_error("cannot replace " + path + ": " + std::strerror(errno));
  }
}

std::string dir_of(const std::string& path) {
  const size_t slash = path.rfind('/');
  return slash == std::string::npos ? "." : slash == 0 ? "/" : path.substr(0, slash);
}

// mkdir -p with `mode` for the directories it creates.
void make_dirs(const std::string& dir, mode_t mode) {
  for (size_t i = 1; i <= dir.size(); ++i)
    if (i == dir.size() || dir[i] == '/') ::mkdir(dir.substr(0, i).c_str(), mode);
  if (!paths::is_dir(dir)) throw std::runtime_error("cannot create " + dir);
}

std::string xml(const std::string& s) {
  std::string o;
  for (char c : s) {
    switch (c) {
      case '&': o += "&amp;"; break;
      case '<': o += "&lt;"; break;
      case '>': o += "&gt;"; break;
      case '"': o += "&quot;"; break;
      case '\'': o += "&apos;"; break;
      default: o += c;
    }
  }
  return o;
}

struct Ran {
  int code = -1;
  std::string out;
};

// Runs argv (no shell); `capture`: its stdout and stderr, else inherited.
Ran run(const std::vector<std::string>& argv, bool capture) {
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  int p[2] = {-1, -1};
  if (capture) {
    if (::pipe(p) != 0) throw std::runtime_error("pipe failed");
    posix_spawn_file_actions_adddup2(&fa, p[1], 1);
    posix_spawn_file_actions_adddup2(&fa, p[1], 2);
    posix_spawn_file_actions_addclose(&fa, p[0]);
    posix_spawn_file_actions_addclose(&fa, p[1]);
  }
  std::vector<char*> args;
  for (const std::string& a : argv) args.push_back(const_cast<char*>(a.c_str()));
  args.push_back(nullptr);
  pid_t pid = 0;
  const int rc = ::posix_spawn(&pid, argv[0].c_str(), &fa, nullptr, args.data(), environ);
  posix_spawn_file_actions_destroy(&fa);
  Ran r;
  if (capture) ::close(p[1]);
  if (rc != 0) {
    if (capture) ::close(p[0]);
    r.out = "cannot run " + argv[0] + ": " + std::strerror(rc);
    return r;
  }
  if (capture) {
    char buf[4096];
    for (ssize_t n; (n = ::read(p[0], buf, sizeof buf)) != 0;) {
      if (n < 0) {
        if (errno == EINTR) continue;
        break;
      }
      r.out.append(buf, size_t(n));
    }
    ::close(p[0]);
  }
  int st = 0;
  while (::waitpid(pid, &st, 0) < 0 && errno == EINTR) {
  }
  r.code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
  return r;
}

std::string domain(const Setup& s) { return "gui/" + std::to_string(s.uid); }

// A command line as text (for --dry-run and messages).
std::string shown(const std::vector<std::string>& argv) {
  std::string o;
  for (const std::string& a : argv) {
    if (!o.empty()) o += ' ';
    o += a.find_first_of(" \t'\"") == std::string::npos ? a : "'" + a + "'";
  }
  return o;
}

json::Value read_config(const std::string& path, bool& found) {
  const std::string text = read_file(path, found);
  if (!found) return json::Value::object();
  json::Value v;
  try {
    v = json::parse(text);
  } catch (const std::exception& e) {
    throw std::runtime_error(path + ": " + e.what());
  }
  if (!v.is_object()) throw std::runtime_error(path + ": expected a JSON object of options");
  return v;
}

// GET /health at the configured address: "200 ...", "503 ...", or why not.
std::string health(const json::Value& cfg) {
  auto str = [&](const char* k, const char* def) {
    const json::Value* v = cfg.get(k);
    return v && v->is_string() ? v->as_string() : v && v->is_number() ? json::dump(*v) : std::string(def);
  };
  std::string host = str("host", "127.0.0.1");
  const std::string port = str("port", "8080");
  if (host == "0.0.0.0") host = "127.0.0.1";
  if (host == "::") host = "::1";
  addrinfo hints{};
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* ai = nullptr;
  if (::getaddrinfo(host.c_str(), port.c_str(), &hints, &ai) != 0 || !ai) return "cannot resolve " + host;
  const int fd = ::socket(ai->ai_family, SOCK_STREAM, 0);
  std::string out = "no answer at " + host + ":" + port;
  if (fd >= 0) {
    ::fcntl(fd, F_SETFL, O_NONBLOCK);
    bool up = ::connect(fd, ai->ai_addr, ai->ai_addrlen) == 0;
    pollfd p{fd, POLLOUT, 0};
    int so = 0;
    socklen_t sl = sizeof so;
    if (!up && errno == EINPROGRESS)
      up = ::poll(&p, 1, 1000) == 1 && ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &so, &sl) == 0 && so == 0;
    if (up) {
      const std::string rq = "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
      (void)!::write(fd, rq.data(), rq.size());
      std::string got;
      char buf[2048];
      pollfd r{fd, POLLIN, 0};
      while (got.find("\r\n\r\n") == std::string::npos && ::poll(&r, 1, 3000) == 1) {
        const ssize_t n = ::read(fd, buf, sizeof buf);
        if (n <= 0) break;
        got.append(buf, size_t(n));
      }
      const size_t sp = got.find(' ');
      const int code = sp == std::string::npos ? 0 : std::atoi(got.c_str() + sp + 1);
      out = code == 200   ? "ready (GET /health: 200)"
            : code == 503 ? "loading the model (GET /health: 503)"
            : code == 401 ? "answering (GET /health needs the API key: 401)"
            : code         ? "answering (GET /health: " + std::to_string(code) + ")"
                           : "connected, but no HTTP answer at " + host + ":" + port;
    }
    ::close(fd);
  }
  ::freeaddrinfo(ai);
  return out;
}

}  // namespace

std::string default_config(const std::string& home) {
  const char* xdg = std::getenv("XDG_CONFIG_HOME");
  const std::string base = xdg && *xdg == '/' ? std::string(xdg) : home + "/.config";
  return base + "/espetero/config.json";
}

std::string plist_path(const Setup& s) { return s.home + "/Library/LaunchAgents/" + kLabel + ".plist"; }
std::string log_path(const Setup& s) { return s.home + "/Library/Logs/espetero.log"; }

std::string plist(const Setup& s) {
  std::string o =
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
      "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
      "<plist version=\"1.0\">\n<dict>\n"
      "  <key>Label</key>\n  <string>" + std::string(kLabel) + "</string>\n"
      "  <key>ProgramArguments</key>\n  <array>\n"
      "    <string>" + xml(s.binary) + "</string>\n"
      "    <string>serve</string>\n"
      "    <string>--config</string>\n"
      "    <string>" + xml(s.config) + "</string>\n"
      "  </array>\n";
  if (!s.env.empty()) {
    o += "  <key>EnvironmentVariables</key>\n  <dict>\n";
    for (const auto& [k, v] : s.env) o += "    <key>" + xml(k) + "</key>\n    <string>" + xml(v) + "</string>\n";
    o += "  </dict>\n";
  }
  o +=
      // At login, and again whenever it exits with an error or crashes (not
      // after a clean stop), at most every 10 s.
      "  <key>RunAtLoad</key>\n  <true/>\n"
      "  <key>KeepAlive</key>\n  <dict>\n    <key>SuccessfulExit</key>\n    <false/>\n  </dict>\n"
      "  <key>ThrottleInterval</key>\n  <integer>10</integer>\n"
      // Without a ProcessType launchd applies light resource limits to an
      // agent (throttled CPU and I/O); an inference server needs none.
      "  <key>ProcessType</key>\n  <string>Interactive</string>\n"
      "  <key>WorkingDirectory</key>\n  <string>" + xml(s.home) + "</string>\n"
      "  <key>StandardOutPath</key>\n  <string>" + xml(log_path(s)) + "</string>\n"
      "  <key>StandardErrorPath</key>\n  <string>" + xml(log_path(s)) + "</string>\n"
      "</dict>\n</plist>\n";
  return o;
}

int install(const Setup& s, const std::vector<std::pair<std::string, std::string>>& settings,
            const std::vector<std::string>& switches, FILE* out) {
  bool found = false;
  json::Value cfg = read_config(s.config, found);
  for (const auto& [k, v] : settings) {
    // Switches given without a value become true; numbers that read back
    // unchanged become numbers; anything else stays the text it was.
    bool sw = false;
    for (const std::string& x : switches) sw |= x == k;
    if (sw && v == "1") {
      cfg.set(k, true);
      continue;
    }
    try {
      json::Value n = json::parse(v);
      if (n.is_number() && json::dump(n) == v) {
        cfg.set(k, std::move(n));
        continue;
      }
    } catch (...) {
    }
    cfg.set(k, v);
  }
  const json::Value* model = cfg.get("model");
  if (!model || !model->is_string() || model->as_string().empty())
    throw std::runtime_error("service install: name the model, with --model ORG/NAME (or a path), or as \"model\" in " +
                             s.config);
  const std::string config_text = json::dump(cfg, json::DumpOptions{", ", ": ", false, 2, false}) + "\n";
  const std::string list = plist(s);
  const std::vector<std::string> bootout = {s.launchctl, "bootout", domain(s) + "/" + kLabel};
  const std::vector<std::string> bootstrap = {s.launchctl, "bootstrap", domain(s), plist_path(s)};
  if (s.dry_run) {
    if (!settings.empty() || !found) std::fprintf(out, "would write %s:\n%s\n", s.config.c_str(), config_text.c_str());
    std::fprintf(out, "would write %s:\n%s\nwould run:\n  %s\n  %s\n", plist_path(s).c_str(), list.c_str(),
                 shown(bootout).c_str(), shown(bootstrap).c_str());
    return 0;
  }
  if (!settings.empty() || !found) {
    make_dirs(dir_of(s.config), 0700);
    write_file(s.config, config_text, 0600);  // it may hold the API key
    std::fprintf(out, "wrote %s\n", s.config.c_str());
  }
  make_dirs(dir_of(plist_path(s)), 0755);
  make_dirs(dir_of(log_path(s)), 0755);
  write_file(plist_path(s), list, 0644);
  run(bootout, true);  // not loaded yet: nothing to unload
  const Ran r = run(bootstrap, true);
  if (r.code != 0)
    throw std::runtime_error(shown(bootstrap) + " failed (" + std::to_string(r.code) + "): " + r.out);
  std::fprintf(out,
               "installed %s\n"
               "  runs %s serve --config %s at login, and again if it fails\n"
               "  log: %s (espetero service logs)\n"
               "  check: espetero service status; after editing %s run espetero service install again\n",
               plist_path(s).c_str(), s.binary.c_str(), s.config.c_str(), log_path(s).c_str(), s.config.c_str());
  return 0;
}

int uninstall(const Setup& s, FILE* out) {
  const std::vector<std::string> bootout = {s.launchctl, "bootout", domain(s) + "/" + kLabel};
  struct stat st;
  const bool present = ::stat(plist_path(s).c_str(), &st) == 0;
  if (s.dry_run) {
    std::fprintf(out, "would run:\n  %s\n", shown(bootout).c_str());
    if (present) std::fprintf(out, "would remove %s\n", plist_path(s).c_str());
    return 0;
  }
  const Ran r = run(bootout, true);
  if (present && ::unlink(plist_path(s).c_str()) != 0)
    throw std::runtime_error("cannot remove " + plist_path(s) + ": " + std::strerror(errno));
  if (present)
    std::fprintf(out, "uninstalled: the server is stopped and %s removed (%s and %s stay)\n", plist_path(s).c_str(),
                 s.config.c_str(), log_path(s).c_str());
  else if (r.code == 0)
    std::fprintf(out, "stopped the loaded agent (%s was already gone)\n", plist_path(s).c_str());
  else
    std::fprintf(out, "not installed (%s does not exist)\n", plist_path(s).c_str());
  return 0;
}

int status(const Setup& s, FILE* out) {
  struct stat st;
  const bool installed = ::stat(plist_path(s).c_str(), &st) == 0;
  std::fprintf(out, "service:  %s\n", installed ? plist_path(s).c_str() : "not installed (espetero service install)");
  const Ran r = run({s.launchctl, "print", domain(s) + "/" + kLabel}, true);
  if (r.code == 0) {
    std::string state = "?", pid, last;
    std::istringstream ls(r.out);
    for (std::string line; std::getline(ls, line);) {
      const size_t a = line.find_first_not_of(" \t");
      if (a == std::string::npos) continue;
      line = line.substr(a);
      if (line.rfind("state = ", 0) == 0) state = line.substr(8);
      else if (line.rfind("pid = ", 0) == 0) pid = line.substr(6);
      else if (line.rfind("last exit code = ", 0) == 0) last = line.substr(17);
    }
    std::fprintf(out, "launchd:  loaded, %s%s%s\n", state.c_str(), pid.empty() ? "" : (", pid " + pid).c_str(),
                 last.empty() ? "" : (", last exit code " + last).c_str());
  } else {
    std::fprintf(out, "launchd:  not loaded\n");
  }
  bool found = false;
  json::Value cfg;
  try {
    cfg = read_config(s.config, found);
  } catch (const std::exception& e) {
    std::fprintf(out, "config:   %s\n", e.what());
    cfg = json::Value::object();
  }
  std::fprintf(out, "config:   %s%s\n", s.config.c_str(), found ? "" : " (missing)");
  if (found) std::fprintf(out, "model:    %s\n", cfg.get_string("model", "(none: add \"model\")").c_str());
  std::fprintf(out, "server:   %s\n", health(cfg).c_str());
  std::fprintf(out, "log:      %s\n", log_path(s).c_str());
  const std::string brew = s.home + "/Library/LaunchAgents/homebrew.mxcl.espetero.plist";
  if (::stat(brew.c_str(), &st) == 0)
    std::fprintf(out, "note:     brew services has an espetero service too (%s): run only one of them, both use the "
                      "same port\n", brew.c_str());
  return 0;
}

int logs(const Setup& s, bool follow, int lines) {
  struct stat st;
  if (::stat(log_path(s).c_str(), &st) != 0) throw std::runtime_error("no log yet: " + log_path(s));
  std::vector<std::string> argv = {"/usr/bin/tail", "-n", std::to_string(std::max(0, lines))};
  if (follow) argv.push_back("-F");
  argv.push_back(log_path(s));
  return run(argv, false).code;
}

}  // namespace espetero::service
