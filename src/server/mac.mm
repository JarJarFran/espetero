// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// This Mac's memory for the memory check (models.h): the RAM
// (hw.memsize) and the GPU working set Metal recommends for one process
// (recommendedMaxWorkingSetSize: 2/3 or 3/4 of the RAM by default, or what
// iogpu.wired_limit_mb sets). ESPETERO_MAC_RAM_GB=N answers for a Mac with N GB
// and macOS's default working set instead (to try the check's messages).
#import <Metal/Metal.h>
#include <sys/sysctl.h>

#include <cstdlib>

#include "models.h"

namespace espetero::models {

Mac this_mac() {
  if (const char* v = std::getenv("ESPETERO_MAC_RAM_GB"); v && std::atof(v) > 0) return mac_with_ram(std::atof(v));
  Mac m;
  uint64_t ram = 0;
  size_t len = sizeof ram;
  if (::sysctlbyname("hw.memsize", &ram, &len, nullptr, 0) == 0) m.ram = int64_t(ram);
  @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (d) {
      m.gpu = int64_t(d.recommendedMaxWorkingSetSize);
      m.chip = d.name ? std::string([d.name UTF8String]) : std::string();
    }
  }
  if (!m.gpu) m.gpu = mac_with_ram(double(m.ram) / double(1ull << 30)).gpu;
  return m;
}

}  // namespace espetero::models
