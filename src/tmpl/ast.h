// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

// Syntax tree of a parsed template, mirroring jinja2.nodes closely enough
// that the scope analysis (scope.cpp) can follow jinja2's compiler step by
// step. After analysis every variable reference carries its storage slot, so
// rendering never looks names up by string except in the render context.
#pragma once

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "value.h"

namespace espetero::tmpl {

struct Expr;
struct Stmt;
using ExprPtr = std::unique_ptr<Expr>;
using StmtPtr = std::unique_ptr<Stmt>;
using Body = std::vector<StmtPtr>;

enum class EK : uint8_t {
  Const,
  Name,
  NSRef,  // `ns.attr` as an assignment target
  Getattr,
  Getitem,
  Slice,
  Call,
  Filter,
  Test,
  Neg,
  Pos,
  Not,
  Add,
  Sub,
  Mul,
  Div,
  FloorDiv,
  Mod,
  Pow,
  And,
  Or,
  Compare,
  Concat,
  CondExpr,
  List,
  Tuple,
  Dict,
};

enum class CmpOp : uint8_t { Eq, Ne, Lt, Le, Gt, Ge, In, NotIn };

struct Expr {
  EK k;
  int line;
  Expr(EK kind, int ln) : k(kind), line(ln) {}
  virtual ~Expr() = default;
};

struct ConstExpr final : Expr {
  Val v;  // strings borrow TemplateImpl::literals
  ConstExpr(int ln, Val value) : Expr(EK::Const, ln), v(std::move(value)) {}
};

enum class Ctx : uint8_t { Load, Store, Param };

struct NameExpr final : Expr {
  std::string name;
  Ctx ctx;
  int slot = -1;  // variable slot in its activation
  int hops = 0;   // activations to walk up (closures of macros)
  NameExpr(int ln, std::string n, Ctx c) : Expr(EK::Name, ln), name(std::move(n)), ctx(c) {}
};

struct NSRefExpr final : Expr {
  std::string name, attr;
  int slot = -1, hops = 0;
  Val attr_val;
  NSRefExpr(int ln, std::string n, std::string a) : Expr(EK::NSRef, ln), name(std::move(n)), attr(std::move(a)) {
    attr_val = Val::str_ref(attr);
  }
};

// Method ids an attribute name has on the builtin container types, computed
// once at parse time so `obj.name` needs no string search at render time
// (-1: the type has no such method).
struct AttrHint {
  int16_t str = -1, markup = -1, dict = -1, list = -1, tuple = -1;
  bool dunder = false;
};
AttrHint attr_hint(std::string_view name);  // builtins.cpp

struct GetattrExpr final : Expr {
  ExprPtr node;
  std::string attr;
  Val attr_val;
  AttrHint hint;
  GetattrExpr(int ln, ExprPtr n, std::string a) : Expr(EK::Getattr, ln), node(std::move(n)), attr(std::move(a)) {
    attr_val = Val::str_ref(attr);
    hint = attr_hint(attr);
  }
};

struct GetitemExpr final : Expr {
  ExprPtr node, arg;
  GetitemExpr(int ln, ExprPtr n, ExprPtr a) : Expr(EK::Getitem, ln), node(std::move(n)), arg(std::move(a)) {}
};

struct SliceExpr final : Expr {
  ExprPtr start, stop, step;
  explicit SliceExpr(int ln) : Expr(EK::Slice, ln) {}
};

struct CallArgs {
  std::vector<ExprPtr> args;
  std::vector<std::pair<std::string, ExprPtr>> kwargs;
  ExprPtr dyn_args, dyn_kwargs;
};

struct CallExpr final : Expr {
  ExprPtr node;
  CallArgs a;
  CallExpr(int ln, ExprPtr n) : Expr(EK::Call, ln), node(std::move(n)) {}
};

// Filter and test application. `node` is null for the filter of a
// {% filter %} / {% set x | f %} block (the block's rendered body).
struct FilterExpr final : Expr {
  ExprPtr node;
  std::string name;
  int id = -1;  // builtin id; -1 when unknown (error raised when evaluated)
  CallArgs a;
  FilterExpr(EK kind, int ln, ExprPtr n, std::string nm) : Expr(kind, ln), node(std::move(n)), name(std::move(nm)) {}
};

struct UnaryExpr final : Expr {
  ExprPtr node;
  UnaryExpr(EK kind, int ln, ExprPtr n) : Expr(kind, ln), node(std::move(n)) {}
};

struct BinExpr final : Expr {
  ExprPtr left, right;
  BinExpr(EK kind, int ln, ExprPtr l, ExprPtr r) : Expr(kind, ln), left(std::move(l)), right(std::move(r)) {}
};

struct CompareExpr final : Expr {
  ExprPtr expr;
  std::vector<std::pair<CmpOp, ExprPtr>> ops;
  CompareExpr(int ln, ExprPtr e) : Expr(EK::Compare, ln), expr(std::move(e)) {}
};

struct ConcatExpr final : Expr {
  std::vector<ExprPtr> nodes;
  explicit ConcatExpr(int ln) : Expr(EK::Concat, ln) {}
};

struct CondExpr final : Expr {
  ExprPtr test, expr1, expr2;  // expr1 if test else expr2 (expr2 may be null)
  std::string hint;            // message of the Undefined produced without else
  explicit CondExpr(int ln)
      : Expr(EK::CondExpr, ln),
        hint("the inline if-expression on line " + std::to_string(ln) +
             " evaluated to false and no else section was defined.") {}
};

struct SeqExpr final : Expr {  // list or tuple display
  std::vector<ExprPtr> items;
  SeqExpr(EK kind, int ln) : Expr(kind, ln) {}
};

struct DictExpr final : Expr {
  std::vector<std::pair<ExprPtr, ExprPtr>> items;
  explicit DictExpr(int ln) : Expr(EK::Dict, ln) {}
};

// ---------------------------------------------------------------- frames
// Variable initialisation when a frame is entered (jinja2 enter_frame):
// resolve from the render context, copy ("alias") an outer variable, or mark
// the slot as not yet assigned.
struct InitOp {
  enum Op : uint8_t { Resolve, Alias, Missing } op;
  int slot;
  int ctx_id;    // Resolve: index into TemplateImpl::ctx_names
  int src_hops;  // Alias: source variable location
  int src_slot;
};
using FrameInit = std::vector<InitOp>;

// ---------------------------------------------------------------- statements
enum class SK : uint8_t { Output, If, For, Assign, AssignBlock, Macro, CallBlock, FilterBlock, With, Break, Continue };

struct Stmt {
  SK k;
  int line;
  Stmt(SK kind, int ln) : k(kind), line(ln) {}
  virtual ~Stmt() = default;
};

struct OutputStmt final : Stmt {
  struct Item {
    std::string text;  // template data when expr is null
    ExprPtr expr;
  };
  std::vector<Item> items;
  explicit OutputStmt(int ln) : Stmt(SK::Output, ln) {}
};

struct IfStmt final : Stmt {
  ExprPtr test;
  Body body;
  std::vector<std::unique_ptr<IfStmt>> elifs;
  Body else_;
  explicit IfStmt(int ln) : Stmt(SK::If, ln) {}
};

struct ForStmt final : Stmt {
  ExprPtr target, iter, test;
  ExprPtr test_target;  // the target bound in the loop-filter frame (own slots)
  Body body, else_;
  bool recursive = false;
  FrameInit loop_init, else_init, test_init;
  int loop_slot = -1;  // slot of `loop`, -1 when the body never uses it
  explicit ForStmt(int ln) : Stmt(SK::For, ln) {}
};

struct AssignStmt final : Stmt {
  ExprPtr target, value;
  explicit AssignStmt(int ln) : Stmt(SK::Assign, ln) {}
};

struct AssignBlockStmt final : Stmt {  // {% set x %}...{% endset %}
  ExprPtr target, filter;
  Body body;
  FrameInit init;
  explicit AssignBlockStmt(int ln) : Stmt(SK::AssignBlock, ln) {}
};

// A macro or the body of a call block (which is an anonymous `caller` macro).
struct MacroDef {
  std::string name;
  std::vector<std::unique_ptr<NameExpr>> args;
  std::vector<ExprPtr> defaults;  // aligned to the last args
  Body body;
  FrameInit init;
  int nslots = 0;  // size of an activation of this macro
  bool catch_kwargs = false, catch_varargs = false, caller = false, explicit_caller = false;
  int caller_slot = -1, kwargs_slot = -1, varargs_slot = -1;
  int line = 0;
};

struct MacroStmt final : Stmt {
  std::unique_ptr<MacroDef> def;
  int name_slot = -1;
  explicit MacroStmt(int ln) : Stmt(SK::Macro, ln) {}
};

struct CallBlockStmt final : Stmt {
  ExprPtr call;  // a CallExpr; invoked with caller=<the body macro>
  std::unique_ptr<MacroDef> def;
  explicit CallBlockStmt(int ln) : Stmt(SK::CallBlock, ln) {}
};

struct FilterBlockStmt final : Stmt {
  Body body;
  ExprPtr filter;
  FrameInit init;
  explicit FilterBlockStmt(int ln) : Stmt(SK::FilterBlock, ln) {}
};

struct WithStmt final : Stmt {
  std::vector<ExprPtr> targets, values;
  Body body;
  FrameInit init;
  explicit WithStmt(int ln) : Stmt(SK::With, ln) {}
};

struct LoopCtlStmt final : Stmt {  // break / continue
  LoopCtlStmt(SK kind, int ln) : Stmt(kind, ln) {}
};

// ---------------------------------------------------------------- template
struct TemplateImpl {
  Body body;
  FrameInit root_init;
  int root_slots = 0;
  int self_slot = -1;                  // `self` (TemplateReference) when the template uses it
  std::vector<std::string> ctx_names;  // variables read from the render context
  // String literals, interned like the constants of a Python code object (so
  // identical literals are the same object for `is sameas`).
  std::unordered_set<std::string> literals;
};

// parser.cpp
void parse_template(std::string_view source, TemplateImpl& out);
// scope.cpp: jinja2's symbol analysis; fills slots, hops and frame inits.
void resolve_scopes(TemplateImpl& t);

// builtins.cpp: filter/test registries. Returns -1 when jinja2 has no such name.
int filter_id(std::string_view name);
int test_id(std::string_view name);

}  // namespace espetero::tmpl
