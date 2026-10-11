// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// The built-in web chat: one self-contained page (webui.html, embedded
// in the binary at build time) served at GET / and GET /chat. It talks to
// this server's /v1/chat/completions from the same origin, so the browser
// guard (no API key: requests addressed to this machine and from no other
// origin) lets it through; with an API key the page asks for it and keeps it
// in the browser's localStorage. A Content-Security-Policy allows the page's
// own inline script and style (by hash), images from data URLs and requests
// to this origin only.
#pragma once

#include <string>

#include "../http/http.h"

namespace espetero {

// Registers GET / and GET /chat, served without the API key (the page holds
// no data; it asks for the key). `enabled` false (--no-webui): GET / answers
// a line of text instead (the heartbeat Ollama's clients send) and /chat is
// not served.
void install_webui(http::Server& server, bool enabled);

// The page and its Content-Security-Policy (for the tests).
const std::string& webui_page();
std::string webui_csp(const std::string& page);

}  // namespace espetero
