// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// JSON Schema -> grammar: the subset of JSON Schema the structured
// output automaton enforces (grammar.h), compiled into its nodes.
//
// Enforced: type (one or a list; nullable: true adds null), enum and const
// (any JSON values; filtered by the schema's other keywords), properties,
// required, additionalProperties (false, true or a schema), minProperties,
// maxProperties, items (a schema, false, or a draft-7 tuple with
// additionalItems), prefixItems, minItems, maxItems, minLength and maxLength
// (code points), minimum, maximum, exclusiveMinimum and exclusiveMaximum
// (also draft-4 booleans) on integer and number, anyOf and oneOf (as
// alternatives: oneOf's "exactly one" is not checked), allOf (merged), $ref
// to the document itself (#, #/$defs/..., #/definitions/..., any JSON
// pointer; recursive references too), and unevaluatedProperties /
// unevaluatedItems false. Annotations (title, description, default,
// examples, $comment, readOnly, ...) and unknown keywords are ignored, as
// validators ignore them.
//
// Not enforced: pattern, format, multipleOf, uniqueItems true,
// patternProperties, propertyNames, dependentRequired, dependentSchemas,
// dependencies, if / then / else, not, contains (and min/maxContains),
// unevaluated* schemas, $dynamicRef / $recursiveRef. A strict schema
// ("strict": true) with any of them is refused (SchemaError naming the
// keyword and where it is); otherwise the reply follows the rest of the
// schema and the response lists what was left out.
//
// Generation conventions (each output is valid under the schema): an object
// with properties and no additionalProperties gets no other keys; without
// properties it may get any (json_object). An array without items may hold
// any values. A type-less schema takes the types its keywords imply (any
// JSON value when none). Numbers with bounds have no exponent and at most 15
// significant digits.
#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../util/json.h"
#include "grammar.h"

namespace espetero::grammar {

struct SchemaError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct SchemaOptions {
  bool strict = false;  // keywords that are not enforced: a SchemaError (else listed)
};

// The grammar of a schema's instances. Throws SchemaError for invalid,
// unsatisfiable or (strict) unenforceable schemas; `unenforced` gets the
// keywords left out of a non-strict one ("pattern at #/properties/zip").
std::shared_ptr<const Grammar> compile_schema(const json::Value& schema, const SchemaOptions& opt,
                                              std::vector<std::string>* unenforced = nullptr);

// json_object: any JSON object.
std::shared_ptr<const Grammar> any_object_grammar();

}  // namespace espetero::grammar
