// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Image inputs for Qwen2/3-VL style vision towers (Qwen3.5 family).
//
// Reproduces the Python reference (jevfire/metal/vision.py: PIL decode,
// smart_resize, PIL bicubic resize, rescale/normalise, temporal duplication,
// patch rows in 2x2 merge-window order, M-RoPE positions) bit for bit:
//
//   * resize_bicubic is a port of Pillow 12's Resample.c (fixed point, same
//     coefficient rounding, same pass order) and matches PIL byte for byte.
//   * preprocess produces the same float32 values as the numpy code.
//   * decode_image follows PIL's Image.open(...).convert("RGB") semantics:
//     no colour management, alpha dropped (not composited),
//     PIL's palette / 16-bit / CMYK rules. PNG, GIF, BMP, TIFF and WebP decode
//     byte-identical to PIL; baseline and progressive JPEGs go through a
//     libjpeg-turbo compatible decoder (jpeg.cpp); anything else (HEIC, AVIF,
//     arithmetic-coded JPEG, ...) is decoded by ImageIO.
//
// All functions are thread-safe; large images use GCD to split rows across
// cores.
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace espetero::json {
class Value;
}

namespace espetero::vision {

// Processor parameters. Defaults are Qwen3.5's; the image budget defaults to
// 131072 pixels (the reference servers' choice, far below Qwen's own).
struct Spec {
  int patch = 16;     // vision_config.patch_size
  int temporal = 2;   // vision_config.temporal_patch_size
  int merge = 2;      // vision_config.spatial_merge_size
  float mean[3] = {0.5f, 0.5f, 0.5f};  // image_mean (rounded to float32 like numpy)
  float std[3] = {0.5f, 0.5f, 0.5f};   // image_std
  int64_t min_pixels = 65536;
  int64_t max_pixels = 131072;
  int32_t image_token_id = 248056;
  int32_t vision_start_id = 248053;
  int32_t vision_end_id = 248054;

  int factor() const { return patch * merge; }
  int row_dim() const { return 3 * temporal * patch * patch; }
  // The reference couples the budget: min_pixels = min(65536, max_pixels).
  void set_budget(int64_t max_px) {
    max_pixels = max_px;
    min_pixels = max_px < 65536 ? max_px : 65536;
  }
};

// Spec from a Hugging Face checkpoint's parsed config.json and (optional)
// preprocessor_config.json / processor_config.json, like VisionSpec.from_model.
// GGUF models: fill the struct from the vocabulary / mmproj metadata instead.
Spec spec_from_config(const json::Value& config, const json::Value* preprocessor,
                      int64_t max_pixels = 131072);
// Reads <dir>/config.json and the first of preprocessor_config.json,
// processor_config.json that exists. Throws std::runtime_error on failure.
Spec spec_from_model_dir(const std::string& dir, int64_t max_pixels = 131072);

// Packed 8-bit RGB, rows top to bottom, no padding.
struct RGBImage {
  int width = 0, height = 0;
  std::vector<uint8_t> rgb;  // width * height * 3
};

// Patch rows for one image, as the Qwen2/3-VL processor produces them.
struct Prepared {
  std::vector<float> pixels;  // [grid_t*grid_h*grid_w, 3*temporal*patch*patch]
  int grid_t = 1, grid_h = 0, grid_w = 0;
  // Exact cache key: equal digests mean equal grid and pixels (see digest()).
  std::array<uint8_t, 16> digest{};

  std::array<int, 3> grid() const { return {grid_t, grid_h, grid_w}; }
  int rows() const { return grid_t * grid_h * grid_w; }
  int tokens(int merge) const { return grid_t * grid_h * grid_w / (merge * merge); }
  std::string digest_hex() const;
};

// ---- decoding -------------------------------------------------------------

// "data:<mime>;base64,<payload>" -> raw bytes. Like the reference (partition
// at the first comma, ";base64" required in the header, Python's lenient
// base64: characters outside the alphabet are skipped, padding checked).
bool decode_data_url(std::string_view url, std::string& bytes, std::string& err);

// Encoded image bytes -> RGB, following PIL's Image.open(b).convert("RGB").
// Formats: PNG, GIF (first frame), JPEG, BMP, TIFF, WebP, HEIC/AVIF (ImageIO).
bool decode_image(std::string_view bytes, RGBImage& out, std::string& err);

// EXIF orientation (tag 0x0112 of a JPEG / PNG / WebP / TIFF file; 1 when
// absent) and PIL's ImageOps.exif_transpose for it. decode_image applies it
// by default, as mlx-vlm's load_image and transformers do, so phone photos
// reach the tower upright; set_exif_orientation(false) restores the plain
// Image.open(...).convert("RGB") pixels of the Python reference server.
int exif_orientation(std::string_view bytes);
RGBImage apply_orientation(const RGBImage& image, int orientation);
void set_exif_orientation(bool on);
bool exif_orientation_enabled();

// Raw encoded bytes or a data URL (the reference's load_rgb for bytes/str).
bool load_rgb(std::string_view bytes_or_data_url, RGBImage& out, std::string& err);
bool load_rgb_file(const std::string& path, RGBImage& out, std::string& err);

// ---- limits -------------------------------------------------------------------

// Largest image decode_image accepts, width x height as the file declares it
// (checked before any pixel is decoded: a 26-byte GIF can declare 179 MP).
// Default 64 MP (a 48 MP phone photo fits; 102 MP medium format does not);
// never above PIL's DecompressionBombError limit, 178,956,970.
constexpr int64_t kDefaultMaxImagePixels = 64000000;
void set_max_image_pixels(int64_t pixels);
int64_t max_image_pixels();

// Width and height of the image decode_image would return, read from the
// file's header without decoding a pixel: JPEG SOFn, PNG IHDR, GIF (logical
// screen enlarged by the first frame, gif.cpp's canvas), any other format from
// ImageIO's properties; EXIF orientations 5-8 swap the sides when decode_image
// rotates. Sides are 64-bit (headers can declare 2^32 - 1). False with `err`
// when the header cannot be read (decoding then gives its own verdict).
bool probe_image(std::string_view bytes, int64_t& width, int64_t& height, std::string& err);

// ---- geometry and resampling ------------------------------------------------

// Qwen2-VL smart_resize: returns (height, width), multiples of `factor`
// within [min_pixels, max_pixels]. Throws std::invalid_argument when the
// aspect ratio exceeds 200 (or a side is not positive).
std::pair<int, int> smart_resize(int height, int width, int factor, int64_t min_pixels,
                                 int64_t max_pixels);

// PIL's Image.resize((w, h), Image.BICUBIC) on an RGB image, byte for byte
// (Pillow 12: fixed-point two-pass resampling, tall images vertical first).
RGBImage resize_bicubic(const RGBImage& in, int width, int height);

// ---- processor --------------------------------------------------------------

// smart_resize + resize + normalise + patchify (+ digest). Throws
// std::invalid_argument for images the reference rejects (aspect > 200).
Prepared preprocess(const RGBImage& image, const Spec& spec);

// Same pipeline from an image that already has the target size (no resize).
// Exposed for tests; `resized` must be smart_resize'd for spec.
Prepared patchify(const RGBImage& resized, const Spec& spec);

// Digest of a grid and its pixel rows: BLAKE2b-128 over a header and the
// BLAKE2b-256 digests of 256 KiB chunks of the little-endian float32 pixels
// (chunks hashed in parallel):
//   leaf_i = BLAKE2b-256(bytes[i*262144 : (i+1)*262144])
//   digest = BLAKE2b-128("espetero.vision.px1" || le32 t,h,w || le64 nbytes || leaf_0 || ...)
std::array<uint8_t, 16> digest(const float* pixels, size_t count, std::array<int, 3> grid);

// ---- tokens and positions -----------------------------------------------------

int image_tokens(std::array<int, 3> grid, int merge = 2);

// The grid preprocess() gives an image of this size (smart_resize, then one
// patch row per patch): with probe_image's size, an image's token count
// before any pixel is decoded. Throws like smart_resize.
std::array<int, 3> grid_for(int height, int width, const Spec& spec);

// Each image_token_id placeholder in `ids` becomes the image's merged-token
// run (the engine's _expand). Throws when placeholders and grids differ.
std::vector<int32_t> expand_image_tokens(const std::vector<int32_t>& ids,
                                         const std::vector<std::array<int, 3>>& grids,
                                         const Spec& spec);

// M-RoPE (t, h, w) positions for `ids` (images already expanded: vision_start
// followed by the image's token run). Text advances one position per token;
// an image occupies its (t, h/merge, w/merge) grid starting at the current
// position; text after it continues at the block's largest position + 1.
// `start` is the position of ids[0]; *next_pos receives the next text position.
// Throws std::invalid_argument like the reference.
std::vector<std::array<int32_t, 3>> mrope_positions(const std::vector<int32_t>& ids,
                                                    const std::vector<std::array<int, 3>>& grids,
                                                    const Spec& spec, int start = 0,
                                                    int* next_pos = nullptr);

// Row (index into ids) where each image's token run starts (the reference's
// native_images). Throws when runs and grids differ in number.
std::vector<int> image_rows(const std::vector<int32_t>& ids,
                            const std::vector<std::array<int, 3>>& grids, const Spec& spec);

}  // namespace espetero::vision
