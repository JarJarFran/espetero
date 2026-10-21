# Third-party notices

Espetero is licensed under the MIT License (see [LICENSE](LICENSE)), except
`src/engine/clef.h`, `src/engine/clef.cpp`, `src/engine/pplx.h`,
`src/engine/pplx.cpp` and `src/engine/pplx_systemone.cpp`, which are licensed
under the Apache License 2.0 because they are derived from Cloudflare's Clef
release and Perplexity's pplx-decider release.

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
| ggml (block dequantization formulas, IQ2_S / IQ3_S grids) | `src/model/impl/gguf.h`, `src/model/impl/gguf_iq_grids.h` | MIT |
| JEVfire (`jevfire/core.py`, `jevfire/models.py`) | `src/engine/decisions.cpp` | MIT |
| Cloudflare Clef, Clef-Omni (`joint_schema_model.py`) | `src/engine/clef.h`, `src/engine/clef.cpp`, `src/engine/engine.cpp` (Clef-Omni's media limits), `tests/ref/clef_omni_ref.py` | Apache-2.0 |
| Hugging Face transformers (Qwen2-VL `smart_resize`, Gemma 4 `get_aspect_ratio_preserving_size`) | `src/vision/processor.cpp` | Apache-2.0 |
| Hugging Face transformers (Qwen3-Omni thinker semantics: `models/qwen3_omni_moe`) | `src/model/impl/qwen3_omni.h`, `src/model/impl/qwen3_omni_load.h`, `kernels/82_qwen3_omni.metal`, the DeepStack mergers in `src/model/impl/vision.h` and `src/model/impl/load.h`, `tools/clef_convert.py`, `tests/ref/clef_omni_ref.py` | Apache-2.0 |
| Hugging Face transformers (LFM2, LFM2-VL and SigLIP2 model semantics; the LFM2-VL processor's geometry and token runs) | `src/model/impl/lfm2.h`, `src/model/impl/lfm2_load.h`, `src/model/impl/config.h` (`config_from_lfm2`, `vision_from_lfm2`), `src/model/impl/vision.h` (SigLIP2 position table), `kernels/90_lfm2.metal`, `src/vision/lfm2.cpp`, `tests/ref/lfm2_ref.py` | Apache-2.0 |
| LiquidAI d1 (`prompt.py`, `api.py`: the System One prompt, option codes, readout and answer format) | `src/engine/d1.h`, `src/engine/d1.cpp`, `tests/ref/lfm2_ref.py` | LFM Open License v1.0 |
| Unicode Character Database 15.1 | `src/tok/unicode_data.h`, `src/tmpl/unicode_data.inc` | Unicode-3.0 |
| mlx-lm 0.32 (Gemma 4 model semantics: `models/gemma4_text.py`, `models/rope_utils.py`) | `src/model/impl/gemma.h`, `src/model/impl/gemma_load.h`, `src/model/impl/config.h` (`config_from_gemma`), `kernels/80_gemma.metal` | MIT |
| Hugging Face transformers 5.17 (Gemma 4 unified semantics: `models/gemma4_unified/modeling_gemma4_unified.py`, `image_processing_gemma4_unified.py`, `configuration_gemma4_unified.py`) | `src/model/impl/gemma_vision.h` (unified), `src/model/impl/gemma_load.h` (`load_gemma_unified_vision`), `src/model/impl/config.h` (`vision_from_gemma_unified`, `per_layer_config`), `kernels/85_gemma_vision.metal` (`g4u_embed_norm`), `tools/gemma4_unified_convert.py` | Apache-2.0 |
| mlx-vlm 0.7.1 (Gemma 4 vision semantics: `models/gemma4/vision.py`, `gemma4.py`, `language.py`) | `src/model/impl/gemma_vision.h`, `src/model/impl/gemma_load.h` (`load_gemma_vision`), `src/model/impl/config.h` (`vision_from_gemma`), `kernels/85_gemma_vision.metal`, `kernels/80_gemma.metal` (the image mask) | MIT |
| Decision prompts of public models (akhilaaa3/Jev-Omni `jev_omni.py`; EldanRing/winnow-inference `native/protocol.h`, `native/engine.h`) | `tests/ref/gemma4_unified_ref.py` | Apache-2.0, MIT |
| perplexity-ai/pplx-decider-v1.1-27b (`model.py`: the decision prompt, the read-out, `enable_noncausal_full_attention`) | `tests/ref/pplx_decider_ref.py`, `tools/pplx_decider_convert.py`, `src/engine/hidden_head.h`, `src/engine/hidden_head.cpp`, `src/engine/engine.cpp`, `src/engine/chat.cpp` (the head's temperature), `src/engine/pplx.h`, `src/engine/pplx.cpp`, `src/engine/pplx_systemone.cpp` (`/v1/systemone`), `src/model/api.h`, `src/model/impl/forest.h`, `kernels/40_attention.metal` (`TreeSpec::noncausal`) | Apache-2.0 |
| Kev (github.com/jaredpalmer/kev: `kev/api.py`, `kev/model.py`, `kev/serve.py`: the System One request and answer formats, the encoding, the pointer head) | `src/engine/kev.h`, `src/engine/kev.cpp`, `src/engine/kev_engine.cpp`, `src/model/api.h`, `src/model/impl/forest.h` (`TreeSpec::out_rows`), `tools/kev_convert.py`, `tests/ref/kev_ref.py`, `tests/kev_test.cpp` | Apache-2.0 |
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
Q5_1, Q8_0, Q2_K to Q6_K, IQ4_NL, IQ4_XS, IQ3_S, IQ2_S; the IQ2_S and IQ3_S
codebook grids are copied into `src/model/impl/gguf_iq_grids.h`) with the formulas of ggml's
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

Clef-Omni (https://huggingface.co/Cloudflare/clef-omni, revision 0db1cd2;
Apache License 2.0) ships its own `joint_schema_model.py` with the same head:
the differences Espetero follows are its media block (`"\n"` after each image),
its image size limits (`IMAGE_SIZE`, 65,536 to 1,048,576 pixels), its default
`max_length` (64,000) and its image loading without EXIF transposition
(`src/engine/clef.cpp`, `src/engine/engine.cpp`). `tests/ref/clef_omni_ref.py`
re-implements its `encode_record` and head in Python and NumPy for the parity
checks (the release's file is read, never imported).

## Hugging Face transformers

`smart_resize` in `src/vision/processor.cpp` is a C++ translation of
`smart_resize` in `transformers/models/qwen2_vl/image_processing_qwen2_vl.py`
(same rounding, same limits and error message), licensed under the Apache
License 2.0 ([third_party/apache-2.0/LICENSE-2.0.txt](third_party/apache-2.0/LICENSE-2.0.txt)):

```text
Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
```

`gemma_resize` in the same file translates `get_aspect_ratio_preserving_size`
in `transformers/models/gemma4/image_processing_gemma4.py` (the target size,
its rounding and edge cases), and the Gemma 4 attention mask for images
follows `create_masks_for_vision_model` in `models/gemma4/modeling_gemma4.py`
(image tokens see their whole image in the sliding-window layers only; no
code copied), both under the Apache License 2.0:

```text
Copyright 2026 the HuggingFace Team. All rights reserved.
```

The Gemma 4 unified embedder (`prepare_images_g4u` and `encode_vision_g4u`
in `src/model/impl/gemma_vision.h`, `load_gemma_unified_vision` in
`src/model/impl/gemma_load.h`, `vision_from_gemma_unified` and the
`per_layer_config` reading in `src/model/impl/config.h`, `g4u_embed_norm` in
`kernels/85_gemma_vision.metal`) and the converter
`tools/gemma4_unified_convert.py` follow transformers 5.17's
`models/gemma4_unified/` (`Gemma4UnifiedVisionEmbedder`: LayerNorm, the
patch projection, LayerNorm, the factorized position embedding, LayerNorm, the
weightless RMSNorm and the projection; `patches_merge` of the image
processor: 48-pixel patches in raster order; the per-layer geometry of
`Gemma4UnifiedTextConfig`; the image tokens' bidirectional attention in every
layer of `Gemma4UnifiedModel.forward`), under the Apache License 2.0
(Copyright 2026 the HuggingFace Team). The code is new; no transformers
source is copied.

The Qwen3-Omni thinker (`src/model/impl/qwen3_omni.h`,
`src/model/impl/qwen3_omni_load.h`, `kernels/82_qwen3_omni.metal`, the
DeepStack mergers of the vision tower in `src/model/impl/vision.h` and
`src/model/impl/load.h`) follows the semantics of transformers 5.17's
`models/qwen3_omni_moe/modeling_qwen3_omni_moe.py` and
`configuration_qwen3_omni_moe.py`: the configuration defaults, the per-head
q / k RMSNorms, the interleaved M-RoPE sections over the whole head, the
softmax top-k router renormalized over the k picked experts, the DeepStack
features added to the image rows after the first layers, the post-shuffle
LayerNorm of the DeepStack mergers and the image M-RoPE positions of
`get_rope_index`. `tools/clef_convert.py` converts the thinker's tensors to
the names of mlx-vlm 0.7.1's `qwen3_omni_moe` `sanitize` (MIT), and
`tests/ref/clef_omni_ref.py` re-implements the thinker in MLX from the same
transformers semantics. The Metal kernels and the host code are new; no
transformers or mlx-vlm source is copied.

```text
Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
```

The LFM2 decoder (`src/model/impl/lfm2.h`, `src/model/impl/lfm2_load.h`,
`kernels/90_lfm2.metal`, `config_from_lfm2` in `src/model/impl/config.h`)
follows the semantics of transformers' `models/lfm2/modeling_lfm2.py`
(gated short convolution `C * conv(B * x)`, per-head q/k RMSNorms, RoPE over
the whole head, the MLP sizing rule), the vision path (`vision_from_lfm2`,
`load_lfm2_vision`, the SigLIP2 branch of `src/model/impl/vision.h`) that of
`models/siglip2/modeling_siglip2.py` (the position table resized with
antialiased bilinear interpolation, `resize_positional_embeddings`) and
`models/lfm2_vl/modeling_lfm2_vl.py` (pixel unshuffle, projector), and
`src/vision/lfm2.cpp` translates the geometry of
`models/lfm2_vl/image_processing_lfm2_vl.py` (`smart_resize`,
`_is_image_too_large`, `find_closest_aspect_ratio` over `_target_ratios`,
`crop_image_to_patches`) and the token runs of `processing_lfm2_vl.py`
(`_build_image_tokens`). The antialiased weights of the position table follow
Pillow's `precompute_coeffs` as torch evaluates them in float. Licensed under
the Apache License 2.0 ([third_party/apache-2.0/LICENSE-2.0.txt](third_party/apache-2.0/LICENSE-2.0.txt)):

```text
Copyright 2025 the HuggingFace Inc. team. All rights reserved.
```

## LiquidAI d1

`src/engine/d1.h` and `src/engine/d1.cpp` reproduce the System One prompt and
readout of LiquidAI's d1 releases (https://huggingface.co/LiquidAI/d1-3B,
revision 051bcc4: `prompt.py` and `api.py`): the user turn with the images'
placeholders and the state (`json_only` style, no system turn), the question
blocks and their reply instructions, the option codes and their fallback
pool, the max-pooled option tokens and the answer objects; `tests/ref/lfm2_ref.py`
re-implements the same rules in Python as the parity reference. The release
is licensed under the LFM Open License v1.0 (Liquid AI, Inc.), whose
commercial-use grant is limited to entities below an annual revenue
threshold; the model files themselves are not redistributed here.

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

## mlx-vlm

The Gemma 4 vision tower and its projection (`src/model/impl/gemma_vision.h`,
`load_gemma_vision` in `src/model/impl/gemma_load.h`, `vision_from_gemma` in
`src/model/impl/config.h`, `kernels/85_gemma_vision.metal`) follow the
semantics of mlx-vlm 0.7.1's `models/gemma4/vision.py` and `gemma4.py`: the
patch embedding of 2 · (pixels − 0.5), the two learned position tables, the
q / k / v norms with the two-dimensional RoPE and attention scale 1, the
order of the four norms per block, the GeGLU MLP, the average pooling scaled
by the square root of the width, the standardization and the weightless norm
before the projection, and the soft tokens replacing the scaled embeddings.
The Metal kernels and the host code are new; no mlx-vlm source is copied.

```text
MIT License

Copyright © 2025 Prince Canuma

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

## Decision prompts of public models (test reference)

`tests/ref/gemma4_unified_ref.py` reproduces two releases' decision
protocols to check the engine against them: the prompt text of `_prompt` in
`jev_omni.py` of akhilaaa3/Jev-Omni (Apache License 2.0, as its model card
states; [third_party/apache-2.0/LICENSE-2.0.txt](third_party/apache-2.0/LICENSE-2.0.txt))
and the decision prompt, label choice and read-out of `compile`, `answer`
(`native/protocol.h`) and the label selection in `native/engine.h` of
EldanRing/winnow-inference, under the MIT License:

```text
MIT License

Copyright (c) 2026 Winnow contributors

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

## pplx-decider (decision prompt, read-out, noncausal attention)

`tests/ref/pplx_decider_ref.py` reproduces the decision protocol of
`source/src/autojev/model.py` in perplexity-ai/pplx-decider-v1.1-27b (Apache
License 2.0, as the release's LICENSE states;
[third_party/apache-2.0/LICENSE-2.0.txt](third_party/apache-2.0/LICENSE-2.0.txt)):
the system message and prompt text of `decision_messages`, the option lists
of `options` and the read-out of `DecisionModel.forward` and `predict` (the
last token's final-normed state through the readout, the first n slots, the
calibration temperature). `tools/pplx_decider_convert.py` reads the
release's `decision_config.json` and `readout.safetensors` into a
hidden-state head (`src/engine/hidden_head.*`), whose probabilities the
server divides by that temperature (`src/engine/chat.cpp`), and the
noncausal attention (`TreeSpec::noncausal` in `src/model/api.h` and
`src/model/impl/forest.h`, the batch-key range of `flash_attention` in
`kernels/40_attention.metal`, the one-pass uncached head requests in
`src/engine/engine.cpp`) follows `enable_noncausal_full_attention`: the
full-attention layers see every position of the prompt, the linear-attention
layers keep their recurrence. The release's files are read, never imported;
the code is new.

`src/engine/pplx.h`, `src/engine/pplx.cpp` and `src/engine/pplx_systemone.cpp`
port the request side of `model.py` to C++: the decision prompt
(`decision_messages`, `options`, `describe`), the answer codes and `answer`.
Changes: images as image parts before the text (the release's own message
layout), each question run as a hidden-state head request. These three files
are licensed under the Apache License 2.0
([third_party/apache-2.0/LICENSE-2.0.txt](third_party/apache-2.0/LICENSE-2.0.txt)).

## Kev (System One encoding, pointer head)

`src/engine/kev.h`, `src/engine/kev.cpp` and `src/engine/kev_engine.cpp`
follow the decision protocol of Kev (https://github.com/jaredpalmer/kev,
Apache License 2.0, as the repository's LICENSE states;
[third_party/apache-2.0/LICENSE-2.0.txt](third_party/apache-2.0/LICENSE-2.0.txt))
and of its released checkpoint jaredpalmer/kev-27b (Apache-2.0, as its model
card states): the request validation, `render`, `option_text`, `to_record`
and `to_answers` of `kev/api.py` (with the confidence formulas it takes from
TypeSafe's system-one-adapter), the delimiter tokens, `user_tokens` and
`encode` of `kev/model.py` (one row per question continuing the state), the
`PointerHead` read-out and temperature, and the response body and serving
limits of `kev/serve.py`. `tools/kev_convert.py` reads the checkpoint's
`head.pt` (the layout of `kev/checkpoint.py`) without unpickling it, and
`tests/ref/kev_ref.py` re-implements the encoding for the parity checks.
The release's files are read, never imported; the code is new.

## Acknowledgements (ideas, no code)

These projects informed the design without contributing code:

- TensorFold (https://github.com/ashhart/TensorFold, read at 0.3.x, MIT; Apache-2.0 since 0.6.0): the exact position-keyed sampling scheme (`src/engine/sampling.h`) and the structure of the MoE down projection with the expert combine folded in (`kernels/15_moe.metal`).
- MLX (https://github.com/ml-explore/mlx, MIT): the chunked gated delta rule formulation (`kernels/30_deltanet.metal`).
- mlx-vlm (https://github.com/Blaizzy/mlx-vlm, MIT): the reference the vision path and the Qwen3.5 / Qwen3.8-Flash-Next models are checked against; `tools/vision_fidelity.py`, `tests/ref/moe_ref.py`, `tests/ref/gemma4_vision_ref.py`, `tests/ref/pplx_decider_ref.py` and `tests/ref/kev_ref.py` run it as an installed package (the Gemma 4 vision semantics are listed above).
- llama.cpp (https://github.com/ggml-org/llama.cpp, MIT): the GGUF format and the names of its pre-tokenizers (`src/tok/pretok.cpp`).
- BLAKE2 (RFC 7693): the image digest (`src/vision/digest.cpp`) implements BLAKE2b from the specification.
