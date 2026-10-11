// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Unicode helpers with CPython str semantics: code point indexing, the
// whitespace set of str.split()/strip(), str.isprintable() for repr(), and
// full case mappings for upper()/lower()/title()/capitalize(). The property
// and mapping tables are generated from Python's unicodedata
// (tests/ref/tmpl_ref.py --gen-unicode) so results match the reference
// Python byte for byte.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace espetero::tmpl::uni {

// Decodes the code point at s[i] and advances i (invalid bytes: U+FFFD, +1).
uint32_t next(std::string_view s, size_t& i);
void append(std::string& out, uint32_t cp);

bool is_ascii(std::string_view s);
size_t length(std::string_view s);                  // number of code points
size_t byte_offset(std::string_view s, size_t cp);  // byte offset of code point #cp (clamped to size)
// Byte range of the code point slice [cp_begin, cp_end) (clamped).
std::string_view slice(std::string_view s, size_t cp_begin, size_t cp_end);

bool is_space(uint32_t c);      // str.isspace(), \s in re
bool is_printable(uint32_t c);  // str.isprintable()
bool is_alpha(uint32_t c);
bool is_decimal(uint32_t c);
bool is_digit(uint32_t c);
bool is_numeric(uint32_t c);
bool is_alnum(uint32_t c);
bool is_word(uint32_t c);  // \w in re
bool is_lower(uint32_t c);  // Lowercase property
bool is_upper(uint32_t c);  // Uppercase property
bool is_titlecase(uint32_t c);
bool is_cased(uint32_t c);
bool is_case_ignorable(uint32_t c);

std::string upper(std::string_view s);
std::string lower(std::string_view s);
std::string capitalize(std::string_view s);
std::string title(std::string_view s);  // str.title()
std::string swapcase(std::string_view s);
void append_upper(std::string& out, uint32_t c);
void append_lower(std::string& out, uint32_t c);
void append_title(std::string& out, uint32_t c);

}  // namespace espetero::tmpl::uni
