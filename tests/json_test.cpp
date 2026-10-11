// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Round-trip check against Python: reads a JSON file of cases
// [{"value":..., "opts":{...}, "expect":"..."}] produced by tests/ref/json_ref.py.
#include "../src/util/json.h"
#include <cstdio>
#include <fstream>
#include <sstream>
using namespace espetero::json;
int main(int argc, char** argv) {
  std::ifstream f(argv[1]);
  std::stringstream ss; ss << f.rdbuf();
  Value cases = parse(ss.str());
  int bad = 0, n = 0;
  for (const Value& c : cases.as_array()) {
    DumpOptions o;
    const Value& op = *c.get("opts");
    std::string is = op.get_string("item_sep", ","), ks = op.get_string("key_sep", ":");
    o.item_sep = is; o.key_sep = ks;
    o.ensure_ascii = op.get_bool("ensure_ascii", false);
    o.indent = int(op.get_int("indent", -1));
    o.sort_keys = op.get_bool("sort_keys", false);
    std::string got = dump(*c.get("value"), o);
    ++n;
    if (got != c.get("expect")->as_string()) {
      if (++bad < 5) fprintf(stderr, "MISMATCH\n got: %s\n exp: %s\n", got.c_str(), c.get("expect")->as_string().c_str());
    }
  }
  printf("json: %d/%d ok\n", n - bad, n);

  // Doubles as integers: never a cast outside int64's range (undefined
  // behaviour). as_int refuses them; get_int clamps (its callers bound it).
  auto throws = [](const Value& v) {
    try {
      (void)v.as_int();
      return false;
    } catch (const Error&) {
      return true;
    }
  };
  int ibad = 0;
  for (const char* t : {"1e300", "-1e300", "9223372036854775808.0", "1.5"}) ibad += !throws(parse(t));
  ibad += throws(parse("4096.0")) || parse("4096.0").as_int() != 4096;
  ibad += parse("-9223372036854775808.0").as_int() != INT64_MIN;
  const Value o = parse("{\"a\":1e300,\"b\":-1e300,\"c\":-2.7,\"d\":9007199254740993}");
  ibad += o.get_int("a", 0) != INT64_MAX;
  ibad += o.get_int("b", 0) != INT64_MIN;
  ibad += o.get_int("c", 0) != -2;
  ibad += o.get_int("d", 0) != 9007199254740993;
  printf("json ints: %s\n", ibad ? "FAILED" : "ok");
  return bad || ibad ? 1 : 0;
}
