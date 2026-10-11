// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The user's configuration file and the background service.
//
// ~/.config/espetero/config.json ($XDG_CONFIG_HOME/espetero/config.json when
// that is set) is read by serve, chat, batch and info when it exists and no
// --config names another file; the command line wins over it.
//
// `espetero service install|uninstall|status|logs` manages a user
// LaunchAgent, ~/Library/LaunchAgents/espetero.server.plist, that runs
// `espetero serve --config FILE` at login and again when it fails, logging to
// ~/Library/Logs/espetero.log. Nothing runs as root and nothing outside the
// user's home is written.
#pragma once

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace espetero::service {

constexpr const char* kLabel = "espetero.server";

// The default configuration file under `home` (or $XDG_CONFIG_HOME).
std::string default_config(const std::string& home);

struct Setup {
  std::string home;     // the user's home directory
  std::string binary;   // absolute path of the espetero executable the agent runs
  std::string config;   // absolute path of the configuration file it runs with
  // The job's environment: PATH (the hf CLI downloads missing models) and the
  // Hugging Face cache variables of the shell that installs it, so the
  // service reads the same cache. Never secrets (put an API key in the file).
  std::vector<std::pair<std::string, std::string>> env;
  std::string launchctl = "/bin/launchctl";
  unsigned uid = 0;
  bool dry_run = false;  // print what would be written and run; change nothing
};

std::string plist_path(const Setup& s);
std::string log_path(const Setup& s);
// The LaunchAgent's property list.
std::string plist(const Setup& s);

// The subcommands, writing to `out`; errors throw std::runtime_error.
// install: `settings` (serve options given to it, e.g. --model) are merged
// into the configuration file, which must then name a model; the agent is
// (re)loaded, so installing again restarts the server with the new settings.
int install(const Setup& s, const std::vector<std::pair<std::string, std::string>>& settings,
            const std::vector<std::string>& switches, FILE* out);
// uninstall: unloads the agent and removes its property list (the
// configuration file and the log stay).
int uninstall(const Setup& s, FILE* out);
// status: installed or not, loaded and running or not (launchctl), and what
// GET /health answers at the configured address.
int status(const Setup& s, FILE* out);
// logs: the log's last `lines` lines, then (follow) what is appended.
int logs(const Setup& s, bool follow, int lines);

}  // namespace espetero::service
