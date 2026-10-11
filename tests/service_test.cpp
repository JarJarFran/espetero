// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The configuration file and the LaunchAgent (make test), in a
// temporary home directory with a stand-in for launchctl that only records
// its arguments: nothing is installed. The property list is checked with
// plutil -lint.
//
//   build/service_test
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "../src/server/service.h"
#include "../src/util/json.h"

using namespace espetero;

namespace {

int checks = 0, failures = 0;

void check(bool ok, const std::string& what) {
  ++checks;
  if (!ok) {
    ++failures;
    std::fprintf(stderr, "  FAIL: %s\n", what.c_str());
  }
}

std::string slurp(const std::string& path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool exists(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0;
}

mode_t mode_of(const std::string& path) {
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 ? (st.st_mode & 0777) : 0;
}

// Runs f with output to a temporary file; returns what it printed.
template <class F>
std::string captured(const std::string& dir, F&& f, std::string* error = nullptr) {
  const std::string path = dir + "/out.txt";
  FILE* out = std::fopen(path.c_str(), "w");
  try {
    f(out);
  } catch (const std::exception& e) {
    if (error) *error = e.what();
  }
  std::fclose(out);
  return slurp(path);
}

}  // namespace

int main() {
  char tmpl[] = "/tmp/espetero-service-test-XXXXXX";
  const std::string home = ::mkdtemp(tmpl);
  ::unsetenv("XDG_CONFIG_HOME");
  check(service::default_config(home) == home + "/.config/espetero/config.json", "default config");
  ::setenv("XDG_CONFIG_HOME", (home + "/xdg").c_str(), 1);
  check(service::default_config(home) == home + "/xdg/espetero/config.json", "XDG_CONFIG_HOME");
  ::unsetenv("XDG_CONFIG_HOME");

  // A stand-in launchctl: records each call, "print" fails as for a job that is not loaded.
  const std::string calls = home + "/launchctl.calls", fake = home + "/launchctl";
  {
    std::ofstream f(fake);
    // (bootout of a job whose property list is gone fails, as launchctl does for one that is not loaded)
    f << "#!/bin/sh\necho \"$@\" >> '" << calls << "'\n[ \"$1\" = print ] && exit 113\n"
      << "[ \"$1\" = bootout ] && [ ! -e '" << home << "/Library/LaunchAgents/espetero.server.plist' ] && exit 113\nexit 0\n";
  }
  ::chmod(fake.c_str(), 0755);

  service::Setup s;
  s.home = home;
  s.binary = "/opt/espetero & co/bin/espetero";  // characters XML must escape
  s.config = service::default_config(home);
  s.env = {{"PATH", "/opt/homebrew/bin:/usr/bin:/bin"}, {"HF_HOME", home + "/hf <cache>"}};
  s.launchctl = fake;
  s.uid = 501;

  // The property list: valid, with the command, the logs and the restart policy.
  const std::string pl = service::plist(s);
  {
    std::ofstream f(home + "/check.plist");
    f << pl;
  }
  check(std::system(("plutil -lint -s '" + home + "/check.plist'").c_str()) == 0, "plutil -lint");
  check(std::system(("plutil -extract ProgramArguments.0 raw -o - '" + home + "/check.plist' | grep -qx '/opt/espetero & co/bin/espetero'").c_str()) == 0,
        "binary path unescaped by plutil");
  for (const char* want : {"<string>espetero.server</string>", "<string>serve</string>", "<string>--config</string>",
                           "<key>RunAtLoad</key>", "<key>SuccessfulExit</key>", "<string>Interactive</string>",
                           "/Library/Logs/espetero.log</string>", "&lt;cache&gt;"})
    check(pl.find(want) != std::string::npos, std::string("plist has ") + want);

  // install without a model: refused, nothing written.
  std::string err;
  captured(home, [&](FILE* out) { service::install(s, {}, {}, out); }, &err);
  check(err.find("name the model") != std::string::npos && !exists(s.config) && !exists(service::plist_path(s)),
        "install without a model: " + err);

  // --dry-run: prints, writes nothing, runs nothing.
  service::Setup dry = s;
  dry.dry_run = true;
  std::string text = captured(home, [&](FILE* out) { service::install(dry, {{"model", "org/m"}}, {}, out); });
  check(text.find("would write " + s.config) != std::string::npos && text.find("bootstrap gui/501") != std::string::npos &&
            !exists(s.config) && !exists(service::plist_path(s)) && !exists(calls),
        "dry run: " + text.substr(0, 120));

  // install: the options go into the configuration file (0600), the agent is loaded.
  err.clear();
  text = captured(home, [&](FILE* out) {
    service::install(s, {{"model", "mlx-community/Qwen3.5-0.8B-4bit"}, {"port", "8099"}, {"cors", "1"}, {"name", "007"}},
                     {"cors"}, out);
  }, &err);
  check(err.empty(), "install: " + err);
  const json::Value cfg = json::parse(slurp(s.config));
  check(cfg.get_string("model", "") == "mlx-community/Qwen3.5-0.8B-4bit" && cfg.get("port") && cfg.get("port")->is_int() &&
            cfg.get_int("port", 0) == 8099 && cfg.get("cors") && cfg.get("cors")->is_bool() && cfg.get_string("name", "") == "007",
        "config written: " + slurp(s.config));
  check(mode_of(s.config) == 0600 && mode_of(home + "/.config/espetero") == 0700, "config file private");
  check(slurp(service::plist_path(s)) == pl && exists(home + "/Library/Logs"), "plist written");
  check(slurp(calls) == "bootout gui/501/espetero.server\nbootstrap gui/501 " + service::plist_path(s) + "\n",
        "launchctl calls: " + slurp(calls));

  // Installing again keeps the file's other options and takes the new ones.
  captured(home, [&](FILE* out) { service::install(s, {{"port", "8100"}}, {}, out); });
  const json::Value cfg2 = json::parse(slurp(s.config));
  check(cfg2.get_int("port", 0) == 8100 && cfg2.get_string("model", "") == "mlx-community/Qwen3.5-0.8B-4bit", "reinstall merges");

  // status: installed, not loaded (the stand-in), no server answering on 8100.
  text = captured(home, [&](FILE* out) { service::status(s, out); });
  check(text.find("service:  " + service::plist_path(s)) != std::string::npos && text.find("not loaded") != std::string::npos &&
            text.find("model:    mlx-community/Qwen3.5-0.8B-4bit") != std::string::npos &&
            text.find("no answer at 127.0.0.1:8100") != std::string::npos,
        "status: " + text);

  // logs: refused before there is one.
  err.clear();
  captured(home, [&](FILE*) { service::logs(s, false, 5); }, &err);
  check(err.find("no log yet") != std::string::npos, "logs without a log: " + err);

  // uninstall: unloaded and removed; the configuration stays.
  ::unlink(calls.c_str());
  text = captured(home, [&](FILE* out) { service::uninstall(s, out); });
  check(!exists(service::plist_path(s)) && exists(s.config) && slurp(calls) == "bootout gui/501/espetero.server\n",
        "uninstall: " + text);
  text = captured(home, [&](FILE* out) { service::uninstall(s, out); });
  check(text.find("not installed") != std::string::npos, "uninstall twice: " + text);

  std::system(("rm -rf '" + home + "'").c_str());
  std::printf("service_test: %d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
