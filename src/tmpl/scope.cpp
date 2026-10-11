// SPDX-License-Identifier: MIT AND BSD-3-Clause
// Copyright (c) 2026 Francisco José García Barbero
//
// Portions are derived from Jinja2, Copyright 2007 Pallets; BSD-3-Clause License, see THIRD_PARTY_NOTICES.md.

// Variable scoping, ported from jinja2 (idtracking.py and the frame handling
// of compiler.py). jinja2 resolves every name statically: each frame (the
// template, a for-loop body, a macro, a {% set %} block, ...) decides at
// compile time whether a variable is a parameter, a copy of an outer
// variable taken when the frame is entered ("alias"), a lookup in the render
// context ("resolve"), or unset. The observable consequences are subtle -- a
// loop body starts every iteration from the outer values, a variable set
// after a macro definition is visible to the macro but the render-context
// value of the same name is not -- so the analysis is mirrored step by step
// instead of approximated with dynamic scopes.
//
// Storage model: an Activation is a Python function scope of the generated
// jinja2 code (the template root, one macro call). Every frame gets its own
// slots inside its activation; a name reference compiles to (hops, slot)
// where hops counts macro-closure boundaries.
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <unordered_map>

#include "ast.h"

namespace espetero::tmpl {

namespace {

struct Act {
  int depth = 0;
  int nslots = 0;
};

struct Storage {
  Act* act;
  std::unordered_map<std::string, int> slots;
  int slot_for(const std::string& name) {
    auto it = slots.find(name);
    if (it != slots.end()) return it->second;
    const int s = act->nslots++;
    slots.emplace(name, s);
    return s;
  }
};

struct Ref {
  Storage* st = nullptr;
  int slot = -1;
};

struct Load {
  enum K { Param, Resolve, Alias, Undefined } k;
  Ref alias;
  std::string name;
};

// jinja2.idtracking.Symbols
struct Symbols {
  Symbols* parent = nullptr;
  int level = 0;
  Storage* st = nullptr;
  std::map<std::string, Ref> refs;
  std::vector<std::pair<int, Load>> loads;  // insertion-ordered, keyed by slot
  std::set<std::string> stores;

  void set_load(int slot, Load l) {
    for (auto& [s, x] : loads)
      if (s == slot) {
        x = std::move(l);
        return;
      }
    loads.emplace_back(slot, std::move(l));
  }
  Ref define_ref(const std::string& name, std::optional<Load> load) {
    Ref r{st, st->slot_for(name)};
    refs[name] = r;
    if (load) set_load(r.slot, std::move(*load));
    return r;
  }
  const Ref* find_ref(const std::string& name) const {
    auto it = refs.find(name);
    if (it != refs.end()) return &it->second;
    return parent ? parent->find_ref(name) : nullptr;
  }
  void store(const std::string& name) {
    stores.insert(name);
    if (refs.count(name)) return;
    if (parent)
      if (const Ref* outer = parent->find_ref(name)) {
        define_ref(name, Load{Load::Alias, *outer, name});
        return;
      }
    define_ref(name, Load{Load::Undefined, {}, name});
  }
  Ref declare_parameter(const std::string& name) {
    stores.insert(name);
    return define_ref(name, Load{Load::Param, {}, name});
  }
  void load(const std::string& name) {
    if (!find_ref(name)) define_ref(name, Load{Load::Resolve, {}, name});
  }
  void branch_update(const std::vector<Symbols*>& branches) {
    std::set<std::string> new_stores;
    for (const Symbols* b : branches) new_stores.insert(b->stores.begin(), b->stores.end());
    for (const std::string& s : stores) new_stores.erase(s);
    for (const Symbols* b : branches) {
      for (const auto& [n, r] : b->refs) refs[n] = r;
      for (const auto& [slot, l] : b->loads) set_load(slot, l);
      stores.insert(b->stores.begin(), b->stores.end());
    }
    for (const std::string& name : new_stores) {
      const Ref target = *find_ref(name);
      if (parent)
        if (const Ref* outer = parent->find_ref(name)) {
          set_load(target.slot, Load{Load::Alias, *outer, name});
          continue;
        }
      set_load(target.slot, Load{Load::Resolve, {}, name});
    }
  }
};

enum class ForBranch { Body, Else, Test };

// jinja2.compiler.UndeclaredNameVisitor: which of `names` are loaded (anywhere
// below `nodes`, closures included) before being declared.
class Undeclared {
 public:
  explicit Undeclared(std::set<std::string> names) : names_(std::move(names)) {}
  std::set<std::string> found;

  void body(const Body& b) {
    for (const auto& s : b)
      if (!done_) stmt(s.get());
  }
  void stmt(const Stmt* s) {
    if (done_) return;
    switch (s->k) {
      case SK::Output:
        for (const auto& it : static_cast<const OutputStmt*>(s)->items)
          if (it.expr) expr(it.expr.get());
        break;
      case SK::If: if_stmt(static_cast<const IfStmt*>(s)); break;
      case SK::For: {
        auto* f = static_cast<const ForStmt*>(s);
        expr(f->target.get());
        expr(f->iter.get());
        body(f->body);
        body(f->else_);
        if (f->test) expr(f->test.get());
        break;
      }
      case SK::Assign: {
        auto* a = static_cast<const AssignStmt*>(s);
        expr(a->target.get());
        expr(a->value.get());
        break;
      }
      case SK::AssignBlock: {
        auto* a = static_cast<const AssignBlockStmt*>(s);
        expr(a->target.get());
        if (a->filter) expr(a->filter.get());
        body(a->body);
        break;
      }
      case SK::Macro: macro(*static_cast<const MacroStmt*>(s)->def); break;
      case SK::CallBlock: {
        auto* c = static_cast<const CallBlockStmt*>(s);
        expr(c->call.get());
        macro(*c->def);
        break;
      }
      case SK::FilterBlock: {
        auto* fb = static_cast<const FilterBlockStmt*>(s);
        body(fb->body);
        expr(fb->filter.get());
        break;
      }
      case SK::With: {
        auto* w = static_cast<const WithStmt*>(s);
        for (const auto& t : w->targets) expr(t.get());
        for (const auto& v : w->values) expr(v.get());
        body(w->body);
        break;
      }
      case SK::Break:
      case SK::Continue: break;
    }
  }
  void expr(const Expr* e);

 private:
  void if_stmt(const IfStmt* s) {
    expr(s->test.get());
    body(s->body);
    for (const auto& e : s->elifs) if_stmt(e.get());
    body(s->else_);
  }
  void macro(const MacroDef& d) {
    for (const auto& a : d.args) expr(a.get());
    for (const auto& x : d.defaults) expr(x.get());
    body(d.body);
  }
  void args(const CallArgs& a) {
    for (const auto& x : a.args) expr(x.get());
    for (const auto& kv : a.kwargs) expr(kv.second.get());
    if (a.dyn_args) expr(a.dyn_args.get());
    if (a.dyn_kwargs) expr(a.dyn_kwargs.get());
  }
  std::set<std::string> names_;
  bool done_ = false;
};

void Undeclared::expr(const Expr* e) {
  if (!e || done_) return;
  switch (e->k) {
    case EK::Name: {
      auto* n = static_cast<const NameExpr*>(e);
      if (n->ctx == Ctx::Load && names_.count(n->name)) {
        found.insert(n->name);
        if (found == names_) done_ = true;
      } else {
        names_.erase(n->name);
      }
      break;
    }
    case EK::Getattr: expr(static_cast<const GetattrExpr*>(e)->node.get()); break;
    case EK::Getitem: {
      auto* g = static_cast<const GetitemExpr*>(e);
      expr(g->node.get());
      expr(g->arg.get());
      break;
    }
    case EK::Slice: {
      auto* s = static_cast<const SliceExpr*>(e);
      expr(s->start.get());
      expr(s->stop.get());
      expr(s->step.get());
      break;
    }
    case EK::Call: {
      auto* c = static_cast<const CallExpr*>(e);
      expr(c->node.get());
      args(c->a);
      break;
    }
    case EK::Filter:
    case EK::Test: {
      auto* f = static_cast<const FilterExpr*>(e);
      expr(f->node.get());
      args(f->a);
      break;
    }
    case EK::Neg:
    case EK::Pos:
    case EK::Not: expr(static_cast<const UnaryExpr*>(e)->node.get()); break;
    case EK::Compare: {
      auto* c = static_cast<const CompareExpr*>(e);
      expr(c->expr.get());
      for (const auto& op : c->ops) expr(op.second.get());
      break;
    }
    case EK::Concat:
      for (const auto& n : static_cast<const ConcatExpr*>(e)->nodes) expr(n.get());
      break;
    case EK::CondExpr: {
      auto* c = static_cast<const CondExpr*>(e);
      expr(c->test.get());
      expr(c->expr1.get());
      expr(c->expr2.get());
      break;
    }
    case EK::List:
    case EK::Tuple:
      for (const auto& n : static_cast<const SeqExpr*>(e)->items) expr(n.get());
      break;
    case EK::Dict:
      for (const auto& kv : static_cast<const DictExpr*>(e)->items) {
        expr(kv.first.get());
        expr(kv.second.get());
      }
      break;
    case EK::Const:
    case EK::NSRef: break;
    default: {
      auto* b = static_cast<const BinExpr*>(e);
      expr(b->left.get());
      expr(b->right.get());
      break;
    }
  }
}

// Does any Name stored below `s` equal `name`? (jinja2 node.find_all check)
bool stores_name(const Expr* e, const std::string& name);
bool stores_name_body(const Body& b, const std::string& name);

bool stores_name_macro(const MacroDef& d, const std::string& name) { return stores_name_body(d.body, name); }

bool stores_name_stmt(const Stmt* s, const std::string& name) {
  switch (s->k) {
    case SK::If: {
      auto* i = static_cast<const IfStmt*>(s);
      if (stores_name_body(i->body, name) || stores_name_body(i->else_, name)) return true;
      for (const auto& e : i->elifs)
        if (stores_name_stmt(e.get(), name)) return true;
      return false;
    }
    case SK::For: {
      auto* f = static_cast<const ForStmt*>(s);
      return stores_name(f->target.get(), name) || stores_name_body(f->body, name) ||
             stores_name_body(f->else_, name);
    }
    case SK::Assign: return stores_name(static_cast<const AssignStmt*>(s)->target.get(), name);
    case SK::AssignBlock: {
      auto* a = static_cast<const AssignBlockStmt*>(s);
      return stores_name(a->target.get(), name) || stores_name_body(a->body, name);
    }
    case SK::Macro: return stores_name_macro(*static_cast<const MacroStmt*>(s)->def, name);
    case SK::CallBlock: return stores_name_macro(*static_cast<const CallBlockStmt*>(s)->def, name);
    case SK::FilterBlock: return stores_name_body(static_cast<const FilterBlockStmt*>(s)->body, name);
    case SK::With: return stores_name_body(static_cast<const WithStmt*>(s)->body, name);
    default: return false;
  }
}

bool stores_name_body(const Body& b, const std::string& name) {
  for (const auto& s : b)
    if (stores_name_stmt(s.get(), name)) return true;
  return false;
}

bool stores_name(const Expr* e, const std::string& name) {
  if (!e) return false;
  if (e->k == EK::Name) {
    auto* n = static_cast<const NameExpr*>(e);
    return n->ctx == Ctx::Store && n->name == name;
  }
  if (e->k == EK::Tuple)
    for (const auto& i : static_cast<const SeqExpr*>(e)->items)
      if (stores_name(i.get(), name)) return true;
  return false;
}

ExprPtr clone_target(const Expr* e) {
  if (e->k == EK::Name) {
    auto* n = static_cast<const NameExpr*>(e);
    return std::make_unique<NameExpr>(n->line, n->name, n->ctx);
  }
  auto* t = static_cast<const SeqExpr*>(e);
  auto out = std::make_unique<SeqExpr>(EK::Tuple, t->line);
  for (const auto& i : t->items) out->items.push_back(clone_target(i.get()));
  return out;
}

struct Frame {
  Symbols* sym;
  bool soft = false;   // If / CondExpr: unknown filters and tests only fail when evaluated
  int loop_depth = 0;  // enclosing loops of the same activation (break/continue)
};

class Resolver {
 public:
  explicit Resolver(TemplateImpl& t) : t_(t) {}

  void run() {
    Act* act = new_act(0);
    Symbols* root = new_symbols(nullptr, act);
    root->level = 0;
    Undeclared self_use({"self"});
    self_use.body(t_.body);
    if (self_use.found.count("self")) t_.self_slot = root->declare_parameter("self").slot;
    for (auto& s : t_.body) sv_stmt(*root, s.get());
    t_.root_init = make_init(*root);
    Frame f{root};
    rs_body(f, t_.body);
    t_.root_slots = act->nslots;
  }

 private:
  Act* new_act(int depth) {
    acts_.emplace_back();
    acts_.back().depth = depth;
    return &acts_.back();
  }
  Symbols* new_symbols(Symbols* parent, Act* act) {
    storages_.push_back(Storage{act, {}});
    symbols_.emplace_back();
    Symbols* s = &symbols_.back();
    s->parent = parent;
    s->level = parent ? parent->level + 1 : 0;
    s->st = &storages_.back();
    return s;
  }
  Symbols* inner(const Frame& f) { return new_symbols(f.sym, f.sym->st->act); }

  // ---------------------------------------------------------- FrameSymbolVisitor
  void sv_args(Symbols& s, CallArgs& a) {
    for (auto& x : a.args) sv_expr(s, x.get());
    for (auto& kv : a.kwargs) sv_expr(s, kv.second.get());
    if (a.dyn_args) sv_expr(s, a.dyn_args.get());
    if (a.dyn_kwargs) sv_expr(s, a.dyn_kwargs.get());
  }

  void sv_expr(Symbols& s, Expr* e, bool store_as_param = false) {
    if (!e) return;
    switch (e->k) {
      case EK::Name: {
        auto* n = static_cast<NameExpr*>(e);
        if (store_as_param || n->ctx == Ctx::Param)
          s.declare_parameter(n->name);
        else if (n->ctx == Ctx::Store)
          s.store(n->name);
        else
          s.load(n->name);
        break;
      }
      case EK::NSRef: s.load(static_cast<NSRefExpr*>(e)->name); break;
      case EK::Const: break;
      case EK::Getattr: sv_expr(s, static_cast<GetattrExpr*>(e)->node.get()); break;
      case EK::Getitem: {
        auto* g = static_cast<GetitemExpr*>(e);
        sv_expr(s, g->node.get());
        sv_expr(s, g->arg.get());
        break;
      }
      case EK::Slice: {
        auto* x = static_cast<SliceExpr*>(e);
        sv_expr(s, x->start.get());
        sv_expr(s, x->stop.get());
        sv_expr(s, x->step.get());
        break;
      }
      case EK::Call: {
        auto* c = static_cast<CallExpr*>(e);
        sv_expr(s, c->node.get());
        sv_args(s, c->a);
        break;
      }
      case EK::Filter:
      case EK::Test: {
        auto* f = static_cast<FilterExpr*>(e);
        sv_expr(s, f->node.get());
        sv_args(s, f->a);
        break;
      }
      case EK::Neg:
      case EK::Pos:
      case EK::Not: sv_expr(s, static_cast<UnaryExpr*>(e)->node.get()); break;
      case EK::Compare: {
        auto* c = static_cast<CompareExpr*>(e);
        sv_expr(s, c->expr.get());
        for (auto& op : c->ops) sv_expr(s, op.second.get());
        break;
      }
      case EK::Concat:
        for (auto& n : static_cast<ConcatExpr*>(e)->nodes) sv_expr(s, n.get());
        break;
      case EK::CondExpr: {
        auto* c = static_cast<CondExpr*>(e);
        sv_expr(s, c->test.get());
        sv_expr(s, c->expr1.get());
        sv_expr(s, c->expr2.get());
        break;
      }
      case EK::List:
      case EK::Tuple:
        for (auto& n : static_cast<SeqExpr*>(e)->items) sv_expr(s, n.get(), store_as_param);
        break;
      case EK::Dict:
        for (auto& kv : static_cast<DictExpr*>(e)->items) {
          sv_expr(s, kv.first.get());
          sv_expr(s, kv.second.get());
        }
        break;
      default: {
        auto* b = static_cast<BinExpr*>(e);
        sv_expr(s, b->left.get());
        sv_expr(s, b->right.get());
        break;
      }
    }
  }

  // visit_If: each branch works on a copy of the symbols, then the stores are merged.
  void sv_if(Symbols& s, IfStmt* node) {
    sv_expr(s, node->test.get());
    Symbols body_s = s, elif_s = s, else_s = s;
    for (auto& st : node->body) sv_stmt(body_s, st.get());
    for (auto& e : node->elifs) sv_if(elif_s, e.get());
    for (auto& st : node->else_) sv_stmt(else_s, st.get());
    s.branch_update({&body_s, &elif_s, &else_s});
  }

  void sv_stmt(Symbols& s, Stmt* st) {
    switch (st->k) {
      case SK::Output:
        for (auto& it : static_cast<OutputStmt*>(st)->items)
          if (it.expr) sv_expr(s, it.expr.get());
        break;
      case SK::If: sv_if(s, static_cast<IfStmt*>(st)); break;
      case SK::For: sv_expr(s, static_cast<ForStmt*>(st)->iter.get()); break;
      case SK::Assign: {
        auto* a = static_cast<AssignStmt*>(st);
        sv_expr(s, a->value.get());
        sv_expr(s, a->target.get());
        break;
      }
      case SK::AssignBlock: sv_expr(s, static_cast<AssignBlockStmt*>(st)->target.get()); break;
      case SK::Macro: s.store(static_cast<MacroStmt*>(st)->def->name); break;
      case SK::CallBlock: sv_expr(s, static_cast<CallBlockStmt*>(st)->call.get()); break;
      case SK::FilterBlock: sv_expr(s, static_cast<FilterBlockStmt*>(st)->filter.get()); break;
      case SK::With:
        for (auto& v : static_cast<WithStmt*>(st)->values) sv_expr(s, v.get());
        break;
      case SK::Break:
      case SK::Continue: break;
    }
  }

  // RootVisitor entry points
  void analyze_for(Symbols& s, ForStmt* node, ForBranch branch) {
    if (branch == ForBranch::Body) {
      sv_expr(s, node->target.get(), true);
      for (auto& st : node->body) sv_stmt(s, st.get());
    } else if (branch == ForBranch::Else) {
      for (auto& st : node->else_) sv_stmt(s, st.get());
    } else {
      sv_expr(s, node->target.get(), true);
      sv_expr(s, node->test.get());
    }
  }
  void analyze_macro(Symbols& s, MacroDef& d) {
    for (auto& a : d.args) sv_expr(s, a.get());
    for (auto& x : d.defaults) sv_expr(s, x.get());
    for (auto& st : d.body) sv_stmt(s, st.get());
  }

  // ---------------------------------------------------------- code generation
  int ctx_id(const std::string& name) {
    auto it = ctx_ids_.find(name);
    if (it != ctx_ids_.end()) return it->second;
    const int id = int(t_.ctx_names.size());
    t_.ctx_names.push_back(name);
    ctx_ids_.emplace(name, id);
    return id;
  }

  // enter_frame(): initialisation of the frame's variables.
  FrameInit make_init(const Symbols& s) {
    FrameInit out;
    for (const auto& [slot, l] : s.loads) {
      switch (l.k) {
        case Load::Param: break;
        case Load::Resolve: out.push_back(InitOp{InitOp::Resolve, slot, ctx_id(l.name), 0, 0}); break;
        case Load::Alias:
          out.push_back(InitOp{InitOp::Alias, slot, 0, s.st->act->depth - l.alias.st->act->depth, l.alias.slot});
          break;
        case Load::Undefined: out.push_back(InitOp{InitOp::Missing, slot, 0, 0, 0}); break;
      }
    }
    return out;
  }

  const Ref& ref(const Frame& f, const std::string& name, int line) {
    const Ref* r = f.sym->find_ref(name);
    if (!r)
      throw TemplateError(ErrorKind::Assertion,
                          "Tried to resolve a name to a reference that was unknown to the frame ('" + name + "')",
                          line);
    return *r;
  }

  void rs_target(const Frame& f, Expr* e) {
    if (e->k == EK::Name) {
      auto* n = static_cast<NameExpr*>(e);
      const Ref& r = ref(f, n->name, n->line);
      n->slot = r.slot;
      n->hops = f.sym->st->act->depth - r.st->act->depth;
    } else if (e->k == EK::Tuple) {
      for (auto& i : static_cast<SeqExpr*>(e)->items) rs_target(f, i.get());
    } else if (e->k == EK::NSRef) {
      rs_expr(f, e);
    }
  }

  void rs_args(const Frame& f, CallArgs& a) {
    for (auto& x : a.args) rs_expr(f, x.get());
    for (auto& kv : a.kwargs) rs_expr(f, kv.second.get());
    if (a.dyn_args) rs_expr(f, a.dyn_args.get());
    if (a.dyn_kwargs) rs_expr(f, a.dyn_kwargs.get());
  }

  void rs_expr(const Frame& f, Expr* e) {
    if (!e) return;
    switch (e->k) {
      case EK::Name: {
        auto* n = static_cast<NameExpr*>(e);
        const Ref& r = ref(f, n->name, n->line);
        n->slot = r.slot;
        n->hops = f.sym->st->act->depth - r.st->act->depth;
        break;
      }
      case EK::NSRef: {
        auto* n = static_cast<NSRefExpr*>(e);
        const Ref& r = ref(f, n->name, n->line);
        n->slot = r.slot;
        n->hops = f.sym->st->act->depth - r.st->act->depth;
        break;
      }
      case EK::Const: break;
      case EK::Getattr: rs_expr(f, static_cast<GetattrExpr*>(e)->node.get()); break;
      case EK::Getitem: {
        auto* g = static_cast<GetitemExpr*>(e);
        rs_expr(f, g->node.get());
        rs_expr(f, g->arg.get());
        break;
      }
      case EK::Slice: {
        auto* x = static_cast<SliceExpr*>(e);
        rs_expr(f, x->start.get());
        rs_expr(f, x->stop.get());
        rs_expr(f, x->step.get());
        break;
      }
      case EK::Call: {
        auto* c = static_cast<CallExpr*>(e);
        rs_expr(f, c->node.get());
        rs_args(f, c->a);
        break;
      }
      case EK::Filter:
      case EK::Test: {
        auto* x = static_cast<FilterExpr*>(e);
        const bool is_filter = e->k == EK::Filter;
        x->id = is_filter ? filter_id(x->name) : test_id(x->name);
        if (x->id < 0 && !f.soft)
          throw TemplateError(ErrorKind::Assertion,
                              std::string("No ") + (is_filter ? "filter" : "test") + " named '" + x->name + "'.",
                              e->line);
        rs_expr(f, x->node.get());
        rs_args(f, x->a);
        break;
      }
      case EK::Neg:
      case EK::Pos:
      case EK::Not: rs_expr(f, static_cast<UnaryExpr*>(e)->node.get()); break;
      case EK::Compare: {
        auto* c = static_cast<CompareExpr*>(e);
        rs_expr(f, c->expr.get());
        for (auto& op : c->ops) rs_expr(f, op.second.get());
        break;
      }
      case EK::Concat:
        for (auto& n : static_cast<ConcatExpr*>(e)->nodes) rs_expr(f, n.get());
        break;
      case EK::CondExpr: {
        auto* c = static_cast<CondExpr*>(e);
        Frame soft = f;
        soft.soft = true;
        rs_expr(soft, c->test.get());
        rs_expr(soft, c->expr1.get());
        rs_expr(soft, c->expr2.get());
        break;
      }
      case EK::List:
      case EK::Tuple:
        for (auto& n : static_cast<SeqExpr*>(e)->items) rs_expr(f, n.get());
        break;
      case EK::Dict:
        for (auto& kv : static_cast<DictExpr*>(e)->items) {
          rs_expr(f, kv.first.get());
          rs_expr(f, kv.second.get());
        }
        break;
      default: {
        auto* b = static_cast<BinExpr*>(e);
        rs_expr(f, b->left.get());
        rs_expr(f, b->right.get());
        break;
      }
    }
  }

  void rs_body(const Frame& f, Body& b) {
    for (auto& s : b) rs_stmt(f, s.get());
  }

  void rs_if(const Frame& f, IfStmt* node) {
    rs_expr(f, node->test.get());
    rs_body(f, node->body);
    for (auto& e : node->elifs) rs_if(f, e.get());
    rs_body(f, node->else_);
  }

  // macro_body(): a new activation whose parent is the defining frame.
  void rs_macro(const Frame& f, MacroDef& d) {
    Act* act = new_act(f.sym->st->act->depth + 1);
    Symbols* s = new_symbols(f.sym, act);
    analyze_macro(*s, d);
    int explicit_caller = -1;
    bool has_kwargs_arg = false, has_varargs_arg = false;
    for (size_t i = 0; i < d.args.size(); ++i) {
      if (d.args[i]->name == "caller") explicit_caller = int(i);
      if (d.args[i]->name == "kwargs") has_kwargs_arg = true;
      if (d.args[i]->name == "varargs") has_varargs_arg = true;
    }
    Undeclared u({"caller", "kwargs", "varargs"});
    u.body(d.body);
    if (u.found.count("caller")) {
      if (explicit_caller >= 0) {
        if (explicit_caller < int(d.args.size() - d.defaults.size()))
          throw TemplateError(ErrorKind::Assertion,
                              "When defining macros or call blocks the special \"caller\" argument must be omitted "
                              "or be given a default.",
                              d.line);
        d.explicit_caller = true;
      } else {
        d.caller_slot = s->declare_parameter("caller").slot;
      }
      d.caller = true;
    }
    if (u.found.count("kwargs") && !has_kwargs_arg) {
      d.kwargs_slot = s->declare_parameter("kwargs").slot;
      d.catch_kwargs = true;
    }
    if (u.found.count("varargs") && !has_varargs_arg) {
      d.varargs_slot = s->declare_parameter("varargs").slot;
      d.catch_varargs = true;
    }
    analyze_macro(*s, d);  // jinja2 analyses twice; idempotent but kept faithful
    d.init = make_init(*s);
    Frame mf{s, false, 0};
    for (auto& a : d.args) rs_target(mf, a.get());
    if (d.explicit_caller) d.caller_slot = d.args[size_t(explicit_caller)]->slot;
    for (auto& x : d.defaults) rs_expr(mf, x.get());
    rs_body(mf, d.body);
    d.nslots = act->nslots;
  }

  void rs_stmt(const Frame& f, Stmt* st) {
    switch (st->k) {
      case SK::Output:
        for (auto& it : static_cast<OutputStmt*>(st)->items)
          if (it.expr) rs_expr(f, it.expr.get());
        break;
      case SK::If: {
        Frame soft = f;
        soft.soft = true;
        rs_if(soft, static_cast<IfStmt*>(st));
        break;
      }
      case SK::For: {
        auto* node = static_cast<ForStmt*>(st);
        if (node->recursive)
          throw TemplateError(ErrorKind::Unsupported, "recursive for loops are not supported", node->line);
        if (stores_name_stmt(node, "loop"))
          throw TemplateError(ErrorKind::Assertion, "Can't assign to special loop variable in for-loop target",
                              node->line);
        Symbols* loop_s = inner(f);
        Symbols* test_s = inner(f);
        Symbols* else_s = inner(f);
        Undeclared u({"loop"});
        u.body(node->body);
        if (u.found.count("loop")) node->loop_slot = loop_s->declare_parameter("loop").slot;
        analyze_for(*loop_s, node, ForBranch::Body);
        if (!node->else_.empty()) analyze_for(*else_s, node, ForBranch::Else);
        if (node->test) {
          analyze_for(*test_s, node, ForBranch::Test);
          Frame tf{test_s, false, f.loop_depth};
          node->test_target = clone_target(node->target.get());
          set_ctx_store(node->test_target.get());
          rs_target(tf, node->test_target.get());
          rs_expr(tf, node->test.get());
          node->test_init = make_init(*test_s);
        }
        Frame lf{loop_s, false, f.loop_depth + 1};
        rs_target(lf, node->target.get());
        rs_expr(f, node->iter.get());
        node->loop_init = make_init(*loop_s);
        rs_body(lf, node->body);
        if (!node->else_.empty()) {
          node->else_init = make_init(*else_s);
          Frame ef{else_s, false, f.loop_depth};
          rs_body(ef, node->else_);
        }
        break;
      }
      case SK::Assign: {
        auto* a = static_cast<AssignStmt*>(st);
        rs_expr(f, a->value.get());
        rs_target(f, a->target.get());
        break;
      }
      case SK::AssignBlock: {
        auto* a = static_cast<AssignBlockStmt*>(st);
        Symbols* s = inner(f);
        for (auto& b : a->body) sv_stmt(*s, b.get());
        if (a->filter) sv_expr(*s, a->filter.get());
        a->init = make_init(*s);
        Frame bf{s, false, f.loop_depth};
        rs_body(bf, a->body);
        rs_target(f, a->target.get());
        if (a->filter) rs_expr(bf, a->filter.get());
        break;
      }
      case SK::Macro: {
        auto* m = static_cast<MacroStmt*>(st);
        rs_macro(f, *m->def);
        const Ref& r = ref(f, m->def->name, m->line);
        m->name_slot = r.slot;
        break;
      }
      case SK::CallBlock: {
        auto* c = static_cast<CallBlockStmt*>(st);
        rs_macro(f, *c->def);
        rs_expr(f, c->call.get());
        break;
      }
      case SK::FilterBlock: {
        auto* fb = static_cast<FilterBlockStmt*>(st);
        Symbols* s = inner(f);
        for (auto& b : fb->body) sv_stmt(*s, b.get());
        sv_expr(*s, fb->filter.get());
        fb->init = make_init(*s);
        Frame ff{s, false, f.loop_depth};
        rs_body(ff, fb->body);
        rs_expr(ff, fb->filter.get());
        break;
      }
      case SK::With: {
        auto* w = static_cast<WithStmt*>(st);
        Symbols* s = inner(f);
        for (auto& t : w->targets) sv_expr(*s, t.get());
        for (auto& b : w->body) sv_stmt(*s, b.get());
        w->init = make_init(*s);
        Frame wf{s, false, f.loop_depth};
        for (size_t i = 0; i < w->targets.size(); ++i) {
          rs_target(wf, w->targets[i].get());
          rs_expr(f, w->values[i].get());
        }
        rs_body(wf, w->body);
        break;
      }
      case SK::Break:
      case SK::Continue:
        if (f.loop_depth == 0)
          throw TemplateError(ErrorKind::Syntax,
                              std::string("'") + (st->k == SK::Break ? "break" : "continue") + "' outside loop",
                              st->line);
        break;
    }
  }

  static void set_ctx_store(Expr* e) {
    if (e->k == EK::Name)
      static_cast<NameExpr*>(e)->ctx = Ctx::Store;
    else if (e->k == EK::Tuple)
      for (auto& i : static_cast<SeqExpr*>(e)->items) set_ctx_store(i.get());
  }

  TemplateImpl& t_;
  std::deque<Act> acts_;
  std::deque<Storage> storages_;
  std::deque<Symbols> symbols_;
  std::unordered_map<std::string, int> ctx_ids_;
};

}  // namespace

void resolve_scopes(TemplateImpl& t) { Resolver(t).run(); }

}  // namespace espetero::tmpl
