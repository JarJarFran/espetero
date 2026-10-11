// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Tree-walking evaluator: runtime objects that need the AST (activations,
// macros, loop contexts) and the Interp state of one render call.
#pragma once

#include <new>
#include <string>
#include <vector>

#include "ast.h"

namespace espetero::tmpl {

// The variables of one Python function scope of jinja2's generated code: the
// template root or one macro call. Macros keep their defining activation
// alive (closures).
struct Activation final : Obj {
  Activation* parent;
  Val* slots;  // `n` values stored right after the object: one allocation per call
  uint32_t n;
  bool tracked = false;  // registered with Interp::track (captured by a closure)

  static Activation* make(size_t n, Activation* parent) {
    void* mem = ::operator new(sizeof(Activation) + n * sizeof(Val));
    return new (mem) Activation(n, parent);
  }
  ~Activation() override {
    for (uint32_t i = 0; i < n; ++i) slots[i].~Val();
    if (parent && --parent->rc == 0) delete parent;
  }
  static void operator delete(void* p) { ::operator delete(p); }

 private:
  Activation(size_t count, Activation* p) : parent(p), slots(reinterpret_cast<Val*>(this + 1)), n(uint32_t(count)) {
    static_assert(sizeof(Activation) % alignof(Val) == 0);
    for (uint32_t i = 0; i < n; ++i) new (&slots[i]) Val(Val::missing());
    if (p) ++p->rc;
  }
};

struct MacroObj final : Obj {
  const MacroDef* def;
  Activation* closure;
  MacroObj(const MacroDef* d, Activation* c) : def(d), closure(c) { ++c->rc; }
  ~MacroObj() override {
    if (--closure->rc == 0) delete closure;
  }
};

class Interp;

// jinja2.runtime.LoopContext, including the lazy loop filter (`for x in y if
// cond`) and the one-item look-ahead behind loop.last / loop.nextitem.
struct LoopObj final : Obj {
  Interp* in;
  const ForStmt* node;
  Activation* act;
  Val iterable;
  PyIter it;
  std::vector<Val> buffered;  // remaining items once loop.length consumed them
  size_t bpos = 0;
  bool materialized = false;
  int64_t index0 = -1;
  int64_t length = -1;
  Val before = Val::missing(), current = Val::missing(), after;
  bool after_valid = false;
  Val last_changed = Val::missing();

  LoopObj(Interp* i, const ForStmt* n, Activation* a, Val iter);
  ~LoopObj() override;
  bool pull(Val& out);  // next item passing the loop filter
  bool next(Val& out);
  const Val* peek();
  int64_t len();
};

// Arguments of a call: positional and keyword (names are Str values).
struct Args {
  std::vector<Val> pos;
  std::vector<std::pair<Val, Val>> kw;
  bool take(std::string_view name, Val& out) {
    for (size_t i = 0; i < kw.size(); ++i)
      if (kw[i].first.sv() == name) {
        out = std::move(kw[i].second);
        kw.erase(kw.begin() + long(i));
        return true;
      }
    return false;
  }
  bool has(std::string_view name) const {
    for (const auto& k : kw)
      if (k.first.sv() == name) return true;
    return false;
  }
};

enum class Flow : uint8_t { Normal, Break, Continue };

class Interp {
 public:
  // Variables are looked up in `fixed` first, then in each JSON object of
  // `layers` in order, then among the globals.
  Interp(const TemplateImpl& t, const std::vector<std::pair<std::string_view, Val>>& fixed,
         const std::vector<const json::Value*>& layers, const RenderOptions& opts);
  ~Interp();
  Interp(const Interp&) = delete;
  Interp& operator=(const Interp&) = delete;

  std::string run();

  Val eval(const Expr* e, Activation* act);
  Val call(const Val& callee, Args& args);
  Val call_macro(MacroObj* m, Args& args);
  bool eval_test(const Expr* test, const Val& item, const Expr* target, Activation* act);

  // Keeps a cycle-capable object alive until the render ends; its contents
  // are cleared then so reference cycles cannot leak.
  void track(Obj* o) {
    ++o->rc;
    gc_.push_back(o);
  }

  void track_act(Activation* a) {
    if (!a->tracked) {
      a->tracked = true;
      track(a);
    }
  }

  const RenderOptions& opts;

 private:
  friend struct LoopObj;
  Flow exec(const Body& body, Activation* act);
  Flow exec_stmt(const Stmt* s, Activation* act);
  Flow exec_for(const ForStmt* s, Activation* act);
  void run_init(const FrameInit& init, Activation* act);
  void assign(const Expr* target, const Val& v, Activation* act);
  void check_nsrefs(const Expr* target, Activation* act);
  void eval_args(const CallArgs& a, Activation* act, Args& out);
  Val eval_filter(const FilterExpr* f, Activation* act, const Val* block_value);
  Val render_body_to_str(const Body& body, Activation* act, Flow* flow);

  Val eval_add(const BinExpr* e, Activation* act);
  void emit_add(const BinExpr* e, Activation* act);
  Val add_rest(Val acc, const Expr* const* chain, int from, int n, Activation* act);

  const TemplateImpl& t_;
  std::vector<Val> ctx_vals_;
  std::string* out_ = nullptr;
  std::vector<Obj*> gc_;
  int depth_ = 0;
};

// ---------------------------------------------------------------- builtins.cpp
Val get_attr(Interp& in, const Val& obj, const Val& name, const AttrHint* hint = nullptr);
// Id of the (safe) builtin method `obj.<attr>` resolves to, or -1; lets
// `obj.method(...)` skip creating a bound-method object.
int builtin_method_id(const Val& obj, const AttrHint& hint);
Val call_method(Interp& in, const Val& self, int id, Args& args);
Val get_item(Interp& in, const Val& obj, const Val& key);
Val get_slice(const Val& obj, const Val& start, const Val& stop, const Val& step);
Val call_filter(Interp& in, int id, const std::string& name, const Val& value, Args& args);
bool call_test(Interp& in, int id, const std::string& name, const Val& value, Args& args);
Val call_builtin(Interp& in, BuiltinId id, Args& args);
Val call_method(Interp& in, const MethodObj& m, Args& args);
Val lookup_global(std::string_view name);  // Missing when not a global

// Python operators (ops in builtins.cpp).
Val py_add(const Val& a, const Val& b);
Val py_arith(EK op, const Val& a, const Val& b);  // - * / // % **
Val py_neg(const Val& a);
Val py_pos(const Val& a);
bool py_contains(const Val& container, const Val& item);
std::string html_escape(std::string_view s);

}  // namespace espetero::tmpl
