# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

# Espetero build (Apple clang, Command Line Tools only; Metal kernels are
# embedded and compiled by the driver at load).
CXX      ?= clang++
BUILD    ?= build
# The oldest macOS the binary runs on (Metal Shading Language 3.2 needs macOS 15).
MACOS_MIN ?= 15.0
CXXFLAGS ?= -std=c++20 -O3 -mcpu=apple-m1 -mmacosx-version-min=$(MACOS_MIN) -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers
OBJCFLAGS = -fobjc-arc
# A plain `make` builds on every core: GNU Make 3.81 (the Command Line Tools'
# make) runs one job at a time unless -j is on its command line, and the
# serial build took 70 s on an M1 Max. `make all` and the other goals build
# as asked.
NPROC := $(shell sysctl -n hw.ncpu 2>/dev/null || echo 4)
.DEFAULT_GOAL := default
.PHONY: default
default:
	@$(MAKE) --no-print-directory -j$(NPROC) all
FRAMEWORKS = -framework Metal -framework Foundation -framework CoreFoundation -framework IOKit -framework ImageIO -framework CoreGraphics

KERNELS := $(sort $(wildcard kernels/*.metal))

$(BUILD)/kernels_src.h: $(KERNELS)
	@mkdir -p $(BUILD)
	@{ echo 'static const char espetero_kernels_metal[] = R"ESPETEROK('; cat $(KERNELS); echo ')ESPETEROK";'; \
	   echo 'static const unsigned espetero_kernels_metal_len = sizeof(espetero_kernels_metal) - 1;'; } > $@

$(BUILD)/json.o: src/util/json.cpp src/util/json.h
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

MODEL_HDRS := $(wildcard src/model/impl/*.h) src/model/api.h src/util/json.h src/util/model_family.h src/server/version.h
$(BUILD)/model.o: src/model/model.mm $(MODEL_HDRS) $(BUILD)/kernels_src.h
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) -I$(BUILD) -c $< -o $@

# Parity against the JEVfire reference engine (optional): needs its native sources.
JEVFIRE_NATIVE ?= ../jevfire/native
$(BUILD)/parity_test: tests/parity_test.mm $(BUILD)/model.o $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) -I$(JEVFIRE_NATIVE)/src $^ $(FRAMEWORKS) -o $@

.PHONY: clean
clean:
	rm -rf $(BUILD)

# Quality cost of a kernel option on batched decode (Fase 3): KLD / top-1 / greedy divergence vs an in-process baseline.
$(BUILD)/batch_kld: tests/batch_kld.mm $(BUILD)/model.o $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) $^ $(FRAMEWORKS) -o $@

$(BUILD)/bench_native: tests/bench_native.mm $(BUILD)/model.o $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) $^ $(FRAMEWORKS) -o $@

# Final-normed hidden states over a KLD harness reference's windows
# (kld_harness.py eval --fmt espetero:...).
$(BUILD)/kld_dump: tests/kld_dump.mm $(BUILD)/model.o $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) $^ $(FRAMEWORKS) -o $@

# A checkpoint's tile image (src/model/impl/tiles.h).
$(BUILD)/espetero_tiles: tests/tiles.mm $(BUILD)/model.o $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) $^ $(FRAMEWORKS) -o $@

# ---- modules (plain C++ unless noted)
TOK_SRC    := $(wildcard src/tok/*.cpp)
TMPL_SRC   := $(wildcard src/tmpl/*.cpp)
HTTP_SRC   := $(wildcard src/http/*.cpp)
VISION_SRC := $(wildcard src/vision/*.cpp)
VISION_MM  := $(wildcard src/vision/*.mm)
ENGINE_SRC := $(wildcard src/engine/*.cpp)
SERVER_SRC := $(wildcard src/server/*.cpp)
SERVER_MM  := $(wildcard src/server/*.mm)

OBJS := $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(TOK_SRC) $(TMPL_SRC) $(HTTP_SRC) $(VISION_SRC) $(ENGINE_SRC) $(SERVER_SRC)) \
        $(patsubst src/%.mm,$(BUILD)/obj/%.o,$(VISION_MM) $(SERVER_MM))

$(BUILD)/obj/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/obj/%.o: src/%.mm
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) -MMD -MP -c $< -o $@

-include $(OBJS:.o=.d)

# The web chat page, embedded like the kernels.
$(BUILD)/webui_html.h: src/server/webui.html
	@mkdir -p $(BUILD)
	@{ echo 'static const char webui_html[] = R"WEBUI('; cat $<; echo ')WEBUI";'; } > $@
$(BUILD)/obj/server/webui.o: src/server/webui.cpp $(BUILD)/webui_html.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -I$(BUILD) -MMD -MP -c $< -o $@

$(BUILD)/espetero: $(OBJS) $(BUILD)/model.o $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ $(FRAMEWORKS) -framework Accelerate -o $@

all: $(BUILD)/espetero

$(BUILD)/ab_native: tests/ab_native.mm $(BUILD)/model.o $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) $^ $(FRAMEWORKS) -o $@

LIB_OBJS := $(filter-out $(BUILD)/obj/server/%,$(OBJS))
$(BUILD)/decisions_test: tests/decisions_test.cpp $(LIB_OBJS) $(BUILD)/model.o $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ $(FRAMEWORKS) -framework Accelerate -o $@

# Engine-level refusals and the d1 image-run guard on real checkpoints (GPU):
#   build/engine_test --chat <Qwen3.5 dir> --d1 <d1 dir> --clef <Clef dir>
$(BUILD)/engine_test: tests/engine_test.cpp $(LIB_OBJS) $(BUILD)/model.o $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ $(FRAMEWORKS) -framework Accelerate -o $@

# Kev decision models without a GPU: the release's encoding, answers and head file
# (on Kev's tokenizer when a snapshot is cached; skipped otherwise).
$(BUILD)/kev_test: tests/kev_test.cpp $(BUILD)/obj/engine/kev.o $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(TOK_SRC)) $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ -framework Accelerate -framework CoreFoundation -o $@

# The tokenizer.json loader's family gates on synthetic files (no model, no reference data).
$(BUILD)/tok_load_test: tests/tok_load_test.cpp $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(TOK_SRC)) $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ -framework Accelerate -framework CoreFoundation -o $@

# Gemma 4 without a GPU: config parsing, its tool-call format, hidden-state heads.
$(BUILD)/gemma_test: tests/gemma_test.cpp $(BUILD)/obj/engine/gemma_tools.o $(BUILD)/obj/engine/hidden_head.o $(BUILD)/obj/engine/pplx.o $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/topk_test: tests/topk_test.mm $(BUILD)/model.o $(BUILD)/json.o $(BUILD)/obj/engine/sampling.o
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) $^ $(FRAMEWORKS) -framework Accelerate -o $@

$(BUILD)/sampling_test: tests/sampling_test.cpp $(BUILD)/obj/engine/sampling.o
	$(CXX) $(CXXFLAGS) $^ -framework Accelerate -o $@

# Structured output: schema compiler, automaton, masks (on the cached
# Qwen3.5 tokenizer when there is one; skipped otherwise).
$(BUILD)/grammar_test: tests/grammar_test.cpp $(BUILD)/obj/engine/grammar.o $(BUILD)/obj/engine/json_schema.o \
                       $(BUILD)/obj/engine/sampling.o $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(TOK_SRC)) $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ -framework Accelerate -framework CoreFoundation -o $@

# ---- make test: the tests that need no model and no GPU
$(BUILD)/json_test: tests/json_test.cpp $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/tmpl_test: tests/tmpl_test.cpp $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(TMPL_SRC)) $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/http_test: tests/http_test.cpp $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(HTTP_SRC)) $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ -o $@

VISION_OBJS := $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(VISION_SRC)) $(patsubst src/%.mm,$(BUILD)/obj/%.o,$(VISION_MM))
$(BUILD)/vision_limits_test: tests/vision_limits_test.cpp $(VISION_OBJS) $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ -framework Foundation -framework CoreFoundation -framework ImageIO -framework CoreGraphics -o $@

# Hugging Face ids, downloads into the hub cache from a fake Hub on 127.0.0.1,
# model footprints and the memory check.
HUB_OBJS := $(BUILD)/obj/server/hub.o $(BUILD)/obj/server/download.o $(BUILD)/obj/server/models.o $(BUILD)/obj/server/catalog.o \
            $(BUILD)/obj/server/mac.o
$(BUILD)/hub_test: tests/hub_test.cpp $(HUB_OBJS) $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(HTTP_SRC)) $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ -framework Foundation -framework Metal -o $@

$(BUILD)/model_meta_test: tests/model_meta_test.cpp $(BUILD)/obj/engine/model_meta.o $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# API request parsing: tool_choice, embedding inputs; images by URL
# (address checks, and fetches from a server on 127.0.0.1).
$(BUILD)/api_test: tests/api_test.cpp $(BUILD)/obj/engine/requests.o $(BUILD)/obj/vision/fetch_policy.o \
                   $(BUILD)/obj/vision/fetch.o $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(HTTP_SRC)) $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ -framework Foundation -o $@

# Ollama request translation.
$(BUILD)/ollama_test: tests/ollama_test.cpp $(BUILD)/obj/server/ollama.o $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# The web chat page: its policy, its content, the browser guard.
$(BUILD)/webui_test: tests/webui_test.cpp $(BUILD)/obj/server/webui.o $(patsubst src/%.cpp,$(BUILD)/obj/%.o,$(HTTP_SRC)) $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ -o $@

# The configuration file and the LaunchAgent, in a temporary home.
$(BUILD)/service_test: tests/service_test.cpp $(BUILD)/obj/server/service.o $(BUILD)/json.o
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: test
test: $(BUILD)/json_test $(BUILD)/sampling_test $(BUILD)/tmpl_test $(BUILD)/http_test $(BUILD)/vision_limits_test $(BUILD)/hub_test $(BUILD)/model_meta_test $(BUILD)/api_test \
      $(BUILD)/ollama_test $(BUILD)/webui_test $(BUILD)/service_test $(BUILD)/grammar_test $(BUILD)/gemma_test $(BUILD)/kev_test \
      $(BUILD)/tok_load_test
	python3 tests/ref/json_ref.py $(BUILD)/json_cases.json && $(BUILD)/json_test $(BUILD)/json_cases.json
	$(BUILD)/sampling_test
	$(BUILD)/tmpl_test tests/ref/tmpl_fixtures
	$(BUILD)/http_test
	$(BUILD)/vision_limits_test
	$(BUILD)/hub_test
	$(BUILD)/model_meta_test
	$(BUILD)/api_test
	$(BUILD)/ollama_test
	$(BUILD)/webui_test
	$(BUILD)/service_test
	$(BUILD)/grammar_test "" $(BUILD)/grammar_samples.jsonl && python3 tests/ref/grammar_check.py $(BUILD)/grammar_samples.jsonl
	$(BUILD)/gemma_test
	$(BUILD)/kev_test tests/ref/kev_fixtures.json
	$(BUILD)/tok_load_test

# ---- install / package
# make install [PREFIX=/usr/local] [DESTDIR=staging]: the espetero binary in
# $(PREFIX)/bin.
PREFIX  ?= /usr/local
VERSION := $(shell sed -n 's/.*kVersion = "\(.*\)";/\1/p' src/server/version.h)
DIST    := espetero-$(VERSION)-macos-arm64

.PHONY: install uninstall dist
install: $(BUILD)/espetero
	install -d "$(DESTDIR)$(PREFIX)/bin"
	install -m 755 $(BUILD)/espetero "$(DESTDIR)$(PREFIX)/bin/espetero"

uninstall:
	rm -f "$(DESTDIR)$(PREFIX)/bin/espetero"

# make dist: $(BUILD)/espetero-<version>-macos-arm64.tar.gz with the binary
# (it runs on the macOS version it was built on and later), README.md,
# LICENSE, THIRD_PARTY_NOTICES.md and third_party/.
dist: $(BUILD)/espetero
	rm -rf "$(BUILD)/dist"
	mkdir -p "$(BUILD)/dist/$(DIST)"
	cp $(BUILD)/espetero README.md LICENSE THIRD_PARTY_NOTICES.md "$(BUILD)/dist/$(DIST)/"
	cp -R third_party "$(BUILD)/dist/$(DIST)/"
	COPYFILE_DISABLE=1 tar --uid 0 --gid 0 --uname root --gname wheel -C "$(BUILD)/dist" -czf "$(BUILD)/$(DIST).tar.gz" "$(DIST)"
	rm -rf "$(BUILD)/dist"
	@echo "$(BUILD)/$(DIST).tar.gz"

$(BUILD)/vision_dump: tests/vision_dump.mm $(BUILD)/model.o $(BUILD)/json.o $(VISION_OBJS)
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) $^ $(FRAMEWORKS) -framework Accelerate -o $@

# Teacher-forced logits and greedy generations for given token ids (Gemma 4
# parity against mlx-lm, tests/ref/gemma4_ref.py; with images against mlx-vlm,
# tests/ref/gemma4_vision_ref.py).
$(BUILD)/logits_dump: tests/logits_dump.mm $(BUILD)/model.o $(BUILD)/json.o $(VISION_OBJS)
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) $^ $(FRAMEWORKS) -framework Accelerate -o $@

# DeepSeek-V4.1 development driver (tests/dsv4_run.mm).
$(BUILD)/dsv4_run: tests/dsv4_run.mm $(BUILD)/model.o $(BUILD)/json.o $(VISION_OBJS)
	$(CXX) $(CXXFLAGS) $(OBJCFLAGS) $^ $(FRAMEWORKS) -o $@
