# Third-party notices

Espetero is licensed under the MIT License (see [LICENSE](LICENSE)), except
`src/engine/clef.h` and `src/engine/clef.cpp`, which are licensed under the
Apache License 2.0 because they are derived from Cloudflare's Clef release.

Some files contain code derived from the projects below. Those parts remain
under their original licenses, reproduced here; each derived file names its
sources in its header comment. No third-party library is linked: the build
uses only Apple's system frameworks.

**This software is based in part on the work of the Independent JPEG Group.**

| Component | Used in | License |
|:--|:--|:--|
| libjpeg-turbo 3.1.0 (Independent JPEG Group's libjpeg) | `src/vision/jpeg.cpp` | IJG License; Arm Neon colour formulas: zlib License |
| Pillow 12 (`src/libImaging/Resample.c`) | `src/vision/resize.cpp` | MIT-CMU |
| Jinja2 3.1 (lexer, parser, `idtracking`, filters) | `src/tmpl/` (`parser.cpp`, `scope.cpp`, `builtins.cpp`) | BSD-3-Clause |
| ggml (block dequantization formulas) | `src/model/impl/gguf.h` | MIT |
| JEVfire (`jevfire/core.py`, `jevfire/models.py`) | `src/engine/decisions.cpp` | MIT |
| Cloudflare Clef (`joint_schema_model.py`) | `src/engine/clef.h`, `src/engine/clef.cpp` | Apache-2.0 |
| Hugging Face transformers (Qwen2-VL `smart_resize`) | `src/vision/processor.cpp` | Apache-2.0 |
| Unicode Character Database 15.1 | `src/tok/unicode_data.h`, `src/tmpl/unicode_data.inc` | Unicode-3.0 |
| mlx-lm 0.32 (Gemma 4 model semantics: `models/gemma4_text.py`, `models/rope_utils.py`) | `src/model/impl/gemma.h`, `src/model/impl/gemma_load.h`, `src/model/impl/config.h` (`config_from_gemma`), `kernels/80_gemma.metal` | MIT |
| Chat templates of public models (test fixtures) | `tests/ref/tmpl_fixtures/`, `tests/ref/tmpl_ref.py` | Apache-2.0, MIT |

## libjpeg-turbo and the Independent JPEG Group's software

`src/vision/jpeg.cpp` reimplements, in C++, parts of the following
libjpeg-turbo 3.1.0 files so that JPEG images decode exactly as Pillow decodes
them: the inverse DCT of `jidctint.c` (`jpeg_idct_islow`), the Huffman tables
and decoding of `jdhuff.c` and `jdphuff.c`, the upsampling of `jdsample.c`,
the colour conversion of `jdcolor.c`, the APP0/APP14 handling of `jdmarker.c`,
and the colour formulas of the Arm Neon file `simd/arm/jdcolext-neon.c`.
Changes: the routines were rewritten into a single decoder (64-bit bit
buffer, combined code and magnitude lookup, restart intervals decoded in
parallel); only the baseline, extended and progressive Huffman paths that
Pillow uses are kept, and everything else falls back to Apple's ImageIO.

Copyright notices of the original files:

```text
jidctint.c: This file was part of the Independent JPEG Group's software:
  Copyright (C) 1991-1998, Thomas G. Lane.
  Modification developed 2002-2018 by Guido Vollbeding.
  libjpeg-turbo Modifications:
  Copyright (C) 2015, 2020, 2022, D. R. Commander.
jdhuff.c: This file was part of the Independent JPEG Group's software:
  Copyright (C) 1991-1997, Thomas G. Lane.
  Lossless JPEG Modifications:
  Copyright (C) 1999, Ken Murchison.
  libjpeg-turbo Modifications:
  Copyright (C) 2009-2011, 2016, 2018-2019, 2022, D. R. Commander.
  Copyright (C) 2018, Matthias Räncker.
jdphuff.c: This file was part of the Independent JPEG Group's software:
  Copyright (C) 1995-1997, Thomas G. Lane.
  Lossless JPEG Modifications:
  Copyright (C) 1999, Ken Murchison.
  libjpeg-turbo Modifications:
  Copyright (C) 2015-2016, 2018-2022, D. R. Commander.
jdsample.c: This file was part of the Independent JPEG Group's software:
  Copyright (C) 1991-1996, Thomas G. Lane.
  libjpeg-turbo Modifications:
  Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
  Copyright (C) 2010, 2015-2016, 2022, 2024, D. R. Commander.
  Copyright (C) 2014, MIPS Technologies, Inc., California.
  Copyright (C) 2015, Google, Inc.
  Copyright (C) 2019-2020, Arm Limited.
jdcolor.c: This file was part of the Independent JPEG Group's software:
  Copyright (C) 1991-1997, Thomas G. Lane.
  Modified 2011 by Guido Vollbeding.
  libjpeg-turbo Modifications:
  Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
  Copyright (C) 2009, 2011-2012, 2014-2015, 2022, 2024, D. R. Commander.
  Copyright (C) 2013, Linaro Limited.
jdmarker.c: This file was part of the Independent JPEG Group's software:
  Copyright (C) 1991-1998, Thomas G. Lane.
  Lossless JPEG Modifications:
  Copyright (C) 1999, Ken Murchison.
  libjpeg-turbo Modifications:
  Copyright (C) 2012, 2015, 2022, 2024, D. R. Commander.
```

The IJG License is the "LEGAL ISSUES" section of the IJG README, included
unaltered as [third_party/libjpeg-turbo/README.ijg](third_party/libjpeg-turbo/README.ijg);
libjpeg-turbo's summary of its licenses is
[third_party/libjpeg-turbo/LICENSE.md](third_party/libjpeg-turbo/LICENSE.md).

The Arm Neon colour formulas (`simd/arm/jdcolext-neon.c`) are under the zlib License:

```text
Copyright (C) 2020, Arm Limited.  All Rights Reserved.
Copyright (C) 2020, D. R. Commander.  All Rights Reserved.

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
```

## Pillow

`src/vision/resize.cpp` is a port of the bicubic resampling of Pillow 12
(`src/libImaging/Resample.c`: `precompute_coeffs`, `normalize_coeffs_8bpc` and
the horizontal and vertical passes) for 8-bit RGB images, so that resized
images are byte-identical to Pillow's.

```text
The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh and contributors

Pillow is the friendly PIL fork. It is

    Copyright © 2010 by Jeffrey 'Alex' Clark and contributors

Like PIL, Pillow is licensed under the open source MIT-CMU License:

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:

Permission to use, copy, modify and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```

## Jinja2

The chat-template interpreter in `src/tmpl/` follows Jinja2 3.1 rule for rule:
the lexer and parser (`lexer.py`, `parser.py`) in `parser.cpp`, the static
variable scoping of `idtracking.py` and `compiler.py` in `scope.cpp`, and
filters such as `do_filesizeformat` and `make_attrgetter` in `builtins.cpp`.

```text
Copyright 2007 Pallets

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1.  Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

2.  Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

3.  Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from
    this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## ggml

`src/model/impl/gguf.h` converts GGUF quantized blocks (Q4_0, Q4_1, Q5_0,
Q5_1, Q8_0, Q2_K to Q6_K, IQ4_NL, IQ4_XS) with the formulas of ggml's
`dequantize_row_*` functions (including `get_scale_min_k4` and the `IQ4_NL`
value table).

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## JEVfire

`src/engine/decisions.cpp` ports the decision API of JEVfire
(https://github.com/kikoncuo/jevfire, `jevfire/core.py` and
`jevfire/models.py`): the classification system prompt, the request
validation messages, the label construction (`make_labels`) and the field
decoding (`decode_field`).

```text
MIT License

Copyright (c) 2026 kikoncuo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Cloudflare Clef

`src/engine/clef.h` and `src/engine/clef.cpp` port `joint_schema_model.py` of
Cloudflare's Clef releases (https://huggingface.co/Cloudflare/clef-flash,
revision 17f0b0a; Apache License 2.0): the record encoding (`encode_record`),
the system prompt, the option lists and the joint schema head. Changes:
translated from Python to C++; the head runs in fp32 on the CPU (Accelerate)
instead of bf16; the backbone's hidden states come from Espetero's own
forward pass. These two files are licensed under the Apache License 2.0
([third_party/apache-2.0/LICENSE-2.0.txt](third_party/apache-2.0/LICENSE-2.0.txt)).

## Hugging Face transformers

`smart_resize` in `src/vision/processor.cpp` is a C++ translation of
`smart_resize` in `transformers/models/qwen2_vl/image_processing_qwen2_vl.py`
(same rounding, same limits and error message), licensed under the Apache
License 2.0 ([third_party/apache-2.0/LICENSE-2.0.txt](third_party/apache-2.0/LICENSE-2.0.txt)):

```text
Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
```

## Unicode Character Database

`src/tok/unicode_data.h` (generated by `tools/gen_unicode.py`) and
`src/tmpl/unicode_data.inc` (generated by `tests/ref/tmpl_ref.py
--gen-unicode`) contain tables derived from the Unicode Character Database
15.1 (through Python's `unicodedata`).

```text
UNICODE LICENSE V3

COPYRIGHT AND PERMISSION NOTICE

Copyright © 1991-2026 Unicode, Inc.

NOTICE TO USER: Carefully read the following legal agreement. BY
DOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR
SOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE
TERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT
DOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.

Permission is hereby granted, free of charge, to any person obtaining a
copy of data files and any associated documentation (the "Data Files") or
software and any associated documentation (the "Software") to deal in the
Data Files or Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, and/or sell
copies of the Data Files or Software, and to permit persons to whom the
Data Files or Software are furnished to do so, provided that either (a)
this copyright and permission notice appear with all copies of the Data
Files or Software, or (b) this copyright and permission notice appear in
associated Documentation.

THE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF
THIRD PARTY RIGHTS.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE
BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,
OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA
FILES OR SOFTWARE.

Except as contained in this notice, the name of a copyright holder shall
not be used in advertising or otherwise to promote the sale, use or other
dealings in these Data Files or Software without prior written
authorization of the copyright holder.
```

## mlx-lm

The Gemma 4 decoder (`src/model/impl/gemma.h`, `src/model/impl/gemma_load.h`,
`kernels/80_gemma.metal` and `config_from_gemma` in `src/model/impl/config.h`)
follows the semantics of mlx-lm 0.32's `models/gemma4_text.py` and
`ProportionalRoPE` in `models/rope_utils.py`: the configuration defaults, the
order of the norms, the layer scalar, the router (normalized input, top-k
softmax times per-expert scales), K = V attention with a weightless value
norm, attention scale 1, the proportional RoPE frequencies and the logit
soft-capping. The Metal kernels and the host code are new; no mlx-lm source is
copied.

```text
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Chat templates (test fixtures)

The tests render the chat templates of public models, copied unmodified from
their Hugging Face repositories, and four synthetic templates in
`tests/ref/tmpl_ref.py` that follow public templates closely. Each keeps its
model's license: Apache License 2.0
([third_party/apache-2.0/LICENSE-2.0.txt](third_party/apache-2.0/LICENSE-2.0.txt))
or, for DeepSeek-R1, the MIT License (Copyright (c) 2023 DeepSeek).

| Fixture | Origin | License |
|:--|:--|:--|
| `Dream-v0-Instruct-7B-8bit.jinja` | Dream-org/Dream-v0-Instruct-7B | apache-2.0 |
| `Muse-Glimmer-30B-4bit.jinja` | meta-models/Muse-Glimmer-30B | apache-2.0 |
| `Qwen-Drive-1.0-4B.jinja` | Qwen/Qwen-Drive-1.0-4B (same text as Qwen/Qwen3.5-9B) | apache-2.0 |
| `Qwen-Drive-1.0-4B_2.jinja` | Qwen/Qwen-Drive-1.0-4B (chat_template.json) | apache-2.0 |
| `mlx-community__Qwen3.5-2B-4bit.jinja` | mlx-community/Qwen3.5-2B-4bit (Qwen/Qwen3.5-2B) | apache-2.0 |
| `mlx-community__Qwen3.8-27B-4bit.jinja` | mlx-community/Qwen3.8-27B-4bit | apache-2.0 |
| `Qwen__Qwen3-4B-Instruct-2507.jinja` | Qwen/Qwen3-4B-Instruct-2507 | apache-2.0 |
| `RND1-Base-0910-4bit.jinja` | radicalnumerics/RND1-Base-0910 | apache-2.0 |
| `XHToken__Spark-X2.5-4B.jinja` | XHToken/Spark-X2.5-4B | apache-2.0 |
| `mlx-community__gemma-4-31b-it-4bit.jinja` | mlx-community/gemma-4-31b-it-4bit (google/gemma-4-31B-it) | apache-2.0 |
| `mlx-community__Qwen3.6-35B-A3B-8bit.jinja` | mlx-community/Qwen3.6-35B-A3B-8bit (Qwen/Qwen3.6-35B-A3B) | apache-2.0 |
| `mlx-community__diffusiongemma-26B-A4B-it-8bit.jinja` | mlx-community/diffusiongemma-26B-A4B-it-8bit | apache-2.0 |
| `numind__NuExtract3.jinja` | numind/NuExtract3 | apache-2.0 |
| `openbmb__MiniCPM5-1B.jinja` | openbmb/MiniCPM5-1B | apache-2.0 |
| `openbmb__MiniCPM5-2B.jinja` | openbmb/MiniCPM5-2B | apache-2.0 |
| `Qwen__Qwen3-Embedding-0.6B.jinja` | Qwen/Qwen3-Embedding-0.6B | apache-2.0 |
| `Qwen__Qwen3-VL-2B-Instruct.jinja` | Qwen/Qwen3-VL-2B-Instruct | apache-2.0 |
| `HuggingFaceTB__SmolLM2-135M-Instruct.jinja` | HuggingFaceTB/SmolLM2-135M-Instruct | apache-2.0 |
| `HuggingFaceTB__SmolVLM2-500M-Video-Instruct.jinja` | HuggingFaceTB/SmolVLM2-500M-Video-Instruct | apache-2.0 |
| `XHToken__Spark-X2.5-4B-6bit.jinja` | XHToken/Spark-X2.5-4B (6-bit MLX conversion) | apache-2.0 |
| `synthetic__mistral_style.jinja` | synthetic, after mistralai/Mistral-7B-Instruct-v0.3 | apache-2.0 |
| `synthetic__deepseek_style.jinja` | synthetic, after deepseek-ai/DeepSeek-R1 | mit |
| `synthetic__hermes_style.jinja` | synthetic, json_to_python_type from NousResearch/Hermes-2-Pro-Mistral-7B | apache-2.0 |
| `synthetic__harmony_style.jinja` | synthetic, render_typescript_type from openai/gpt-oss-20b | apache-2.0 |

## Acknowledgements (ideas, no code)

These projects informed the design without contributing code:

- TensorFold (https://github.com/ashhart/TensorFold, read at 0.3.x, MIT; Apache-2.0 since 0.6.0): the exact position-keyed sampling scheme (`src/engine/sampling.h`) and the structure of the MoE down projection with the expert combine folded in (`kernels/15_moe.metal`).
- MLX (https://github.com/ml-explore/mlx, MIT): the chunked gated delta rule formulation (`kernels/30_deltanet.metal`).
- mlx-vlm (https://github.com/Blaizzy/mlx-vlm, MIT): the reference the vision path and the Qwen3.5 / Qwen3.8-Flash-Next models are checked against; `tools/vision_fidelity.py` and `tests/ref/moe_ref.py` run it as an installed package.
- llama.cpp (https://github.com/ggml-org/llama.cpp, MIT): the GGUF format and the names of its pre-tokenizers (`src/tok/pretok.cpp`).
- BLAKE2 (RFC 7693): the image digest (`src/vision/digest.cpp`) implements BLAKE2b from the specification.
