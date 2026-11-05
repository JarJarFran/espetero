// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// espetero quantize (quantize.cpp): a Hugging Face BF16 checkpoint -> an EQ
// checkpoint. argv: the arguments after "quantize".
#pragma once

namespace espetero::quant {
int cmd_quantize(int argc, char** argv);
}
