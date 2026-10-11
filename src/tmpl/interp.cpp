// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Francisco José García Barbero

#include "interp.h"

#include <algorithm>

namespace espetero::tmpl {

namespace {

constexpr int kMaxMacroDepth = 200;  // Python's recursion limit hits first in jinja2

// Drops one reference when leaving scope (exception safe).
struct Hold {
  Obj* o;
  ~Hold() {
    if (--o->rc == 0) delete o;
  }
};

}  // namespace

// Hooks used by value.cpp.
int64_t loop_length(Obj* loop) { return static_cast<LoopObj*>(loop)->len(); }
int64_t loop_index(Obj* loop) { return static_cast<LoopObj*>(loop)->index0 + 1; }
std::string_view macro_name(Obj* m) { return static_cast<MacroObj*>(m)->def->name; }

// ---------------------------------------------------------------- LoopContext
LoopObj::LoopObj(Interp* i, const ForStmt* n, Activation* a, Val iter)
    : in(i), node(n), act(a), iterable(std::move(iter)), it(iterable) {
  ++act->rc;
}

LoopObj::~LoopObj() {
  if (--act->rc == 0) delete act;
}

bool LoopObj::pull(Val& out) {
  for (;;) {
    if (materialized) {
      if (bpos >= buffered.size()) return false;
      out = buffered[bpos++];
      return true;
    }
    Val item;
    if (!it.next(item)) return false;
    if (node->test && !in->eval_test(node->test.get(), item, node->test_target.get(), act)) continue;
    out = std::move(item);
    return true;
  }
}

bool LoopObj::next(Val& out) {
  if (after_valid) {
    out = std::move(after);
    after = Val();
    after_valid = false;
  } else if (!pull(out)) {
    return false;
  }
  ++index0;
  before = std::move(current);
  current = out;
  return true;
}

const Val* LoopObj::peek() {
  if (!after_valid) after_valid = pull(after);
  return after_valid ? &after : nullptr;
}

int64_t LoopObj::len() {
  if (length >= 0) return length;
  // len(self._iterable) when it has one (never for a filtered loop, whose
  // iterable is a generator); otherwise the rest is consumed and buffered.
  if (!node->test && has_len(iterable)) {
    length = py_len(iterable);
    return length;
  }
  std::vector<Val> rest;
  Val x;
  while (pull(x)) rest.push_back(std::move(x));
  buffered = std::move(rest);
  bpos = 0;
  materialized = true;
  length = int64_t(buffered.size()) + index0 + 1 + (after_valid ? 1 : 0);
  return length;
}

// ---------------------------------------------------------------- Interp
Interp::Interp(const TemplateImpl& t, const std::vector<std::pair<std::string_view, Val>>& fixed,
               const std::vector<const json::Value*>& layers, const RenderOptions& o)
    : opts(o), t_(t) {
  ctx_vals_.reserve(t.ctx_names.size());
  for (const std::string& name : t.ctx_names) {
    Val v = Val::missing();
    bool found = false;
    for (const auto& [k, val] : fixed)
      if (k == name) {
        v = val;
        found = true;
        break;
      }
    for (size_t i = 0; !found && i < layers.size(); ++i)
      if (layers[i] && layers[i]->is_object())
        if (const json::Value* j = layers[i]->get(name)) {
          v = from_json(*j);
          found = true;
        }
    if (!found) v = lookup_global(name);
    ctx_vals_.push_back(std::move(v));
  }
}

Interp::~Interp() {
  // Break reference cycles (closures capture activations that hold the
  // macros; namespaces and loop objects can contain themselves) first while
  // every tracked object is still pinned, then drop the pins.
  for (Obj* o : gc_) {
    if (auto* a = dynamic_cast<Activation*>(o)) {
      for (uint32_t i = 0; i < a->n; ++i) a->slots[i] = Val();
    } else if (auto* n = dynamic_cast<NsObj*>(o)) {
      MapData dead;
      std::swap(dead, n->m);
    } else if (auto* l = dynamic_cast<LoopObj*>(o)) {
      Val a = std::move(l->iterable), b = std::move(l->before), c = std::move(l->current), d = std::move(l->after),
          e = std::move(l->last_changed);
      std::vector<Val> f;
      f.swap(l->buffered);
      l->it = PyIter(Val::missing());
    }
  }
  for (Obj* o : gc_)
    if (--o->rc == 0) delete o;
}

std::string Interp::run() {
  auto* root = Activation::make(size_t(t_.root_slots), nullptr);
  root->tracked = true;
  gc_.push_back(root);  // owned by gc_
  std::string out;
  out_ = &out;
  if (t_.self_slot >= 0) root->slots[size_t(t_.self_slot)] = Val::builtin(BuiltinId::TemplateReference);
  run_init(t_.root_init, root);
  exec(t_.body, root);
  out_ = nullptr;
  return out;
}

void Interp::run_init(const FrameInit& init, Activation* act) {
  for (const InitOp& op : init) {
    switch (op.op) {
      case InitOp::Resolve: act->slots[size_t(op.slot)] = ctx_vals_[size_t(op.ctx_id)]; break;
      case InitOp::Alias: {
        Activation* a = act;
        for (int h = op.src_hops; h > 0; --h) a = a->parent;
        act->slots[size_t(op.slot)] = a->slots[size_t(op.src_slot)];
        break;
      }
      case InitOp::Missing: act->slots[size_t(op.slot)] = Val::missing(); break;
    }
  }
}

// ---------------------------------------------------------------- statements
Flow Interp::exec(const Body& body, Activation* act) {
  for (const auto& s : body) {
    const Flow f = exec_stmt(s.get(), act);
    if (f != Flow::Normal) return f;
  }
  return Flow::Normal;
}

Val Interp::render_body_to_str(const Body& body, Activation* act, Flow* flow) {
  std::string buf;
  std::string* saved = out_;
  out_ = &buf;
  try {
    *flow = exec(body, act);
  } catch (...) {
    out_ = saved;
    throw;
  }
  out_ = saved;
  return Val::str(std::move(buf));
}

Flow Interp::exec_stmt(const Stmt* s, Activation* act) {
  switch (s->k) {
    case SK::Output: {
      for (const auto& it : static_cast<const OutputStmt*>(s)->items) {
        if (!it.expr) {
          out_->append(it.text);
          continue;
        }
        if (it.expr->k == EK::Add) {
          emit_add(static_cast<const BinExpr*>(it.expr.get()), act);
          continue;
        }
        const Val v = eval(it.expr.get(), act);
        append_str(*out_, v);
      }
      return Flow::Normal;
    }
    case SK::If: {
      auto* i = static_cast<const IfStmt*>(s);
      if (truthy(eval(i->test.get(), act))) return exec(i->body, act);
      for (const auto& e : i->elifs)
        if (truthy(eval(e->test.get(), act))) return exec(e->body, act);
      return exec(i->else_, act);
    }
    case SK::For: return exec_for(static_cast<const ForStmt*>(s), act);
    case SK::Assign: {
      auto* a = static_cast<const AssignStmt*>(s);
      check_nsrefs(a->target.get(), act);  // jinja2 checks namespaces before evaluating
      const Val v = eval(a->value.get(), act);
      assign(a->target.get(), v, act);
      return Flow::Normal;
    }
    case SK::AssignBlock: {
      auto* a = static_cast<const AssignBlockStmt*>(s);
      run_init(a->init, act);
      Flow flow;
      Val v = render_body_to_str(a->body, act, &flow);
      if (flow != Flow::Normal) return flow;
      if (a->filter) v = eval_filter(static_cast<const FilterExpr*>(a->filter.get()), act, &v);
      assign(a->target.get(), v, act);
      return Flow::Normal;
    }
    case SK::Macro: {
      auto* m = static_cast<const MacroStmt*>(s);
      track_act(act);
      act->slots[size_t(m->name_slot)] = Val::object(Kind::Macro, new MacroObj(m->def.get(), act));
      return Flow::Normal;
    }
    case SK::CallBlock: {
      auto* c = static_cast<const CallBlockStmt*>(s);
      track_act(act);
      const Val caller = Val::object(Kind::Macro, new MacroObj(c->def.get(), act));
      auto* call_expr = static_cast<const CallExpr*>(c->call.get());
      const Val fn = eval(call_expr->node.get(), act);
      Args args;
      eval_args(call_expr->a, act, args);
      args.kw.emplace_back(Val::str_ref("caller"), caller);
      const Val r = call(fn, args);
      append_str(*out_, r);
      return Flow::Normal;
    }
    case SK::FilterBlock: {
      auto* fb = static_cast<const FilterBlockStmt*>(s);
      run_init(fb->init, act);
      Flow flow;
      const Val body = render_body_to_str(fb->body, act, &flow);
      if (flow != Flow::Normal) return flow;
      const Val r = eval_filter(static_cast<const FilterExpr*>(fb->filter.get()), act, &body);
      append_str(*out_, r);
      return Flow::Normal;
    }
    case SK::With: {
      auto* w = static_cast<const WithStmt*>(s);
      run_init(w->init, act);
      for (size_t i = 0; i < w->targets.size(); ++i) {
        const Val v = eval(w->values[i].get(), act);
        assign(w->targets[i].get(), v, act);
      }
      return exec(w->body, act);
    }
    case SK::Break: return Flow::Break;
    case SK::Continue: return Flow::Continue;
  }
  return Flow::Normal;
}

Flow Interp::exec_for(const ForStmt* s, Activation* act) {
  Val iterable = eval(s->iter.get(), act);
  if (s->test) run_init(s->test_init, act);
  auto* loop = new LoopObj(this, s, act, std::move(iterable));
  const Val loop_val = Val::object(Kind::Loop, loop);
  if (s->loop_slot >= 0) {
    // The loop object is visible to the body (and may be stored anywhere).
    track(loop);
    track_act(act);
  }
  // jinja2 runs the else block unless some iteration finished its body
  // normally (a `continue` or `break` does not count).
  bool completed = false;
  Val item;
  while (loop->next(item)) {
    assign(s->target.get(), item, act);
    if (s->loop_slot >= 0) act->slots[size_t(s->loop_slot)] = loop_val;
    run_init(s->loop_init, act);
    const Flow f = exec(s->body, act);
    if (f == Flow::Break) break;
    if (f == Flow::Normal) completed = true;
  }
  if (!s->else_.empty() && !completed) {
    run_init(s->else_init, act);
    return exec(s->else_, act);
  }
  return Flow::Normal;
}

void Interp::check_nsrefs(const Expr* t, Activation* act) {
  if (t->k == EK::NSRef) {
    auto* n = static_cast<const NSRefExpr*>(t);
    Activation* a = act;
    for (int h = n->hops; h > 0; --h) a = a->parent;
    if (!a->slots[size_t(n->slot)].is(Kind::Namespace))
      raise(ErrorKind::Runtime, "cannot assign attribute on non-namespace object");
  } else if (t->k == EK::Tuple) {
    for (const auto& i : static_cast<const SeqExpr*>(t)->items) check_nsrefs(i.get(), act);
  }
}

void Interp::assign(const Expr* target, const Val& v, Activation* act) {
  switch (target->k) {
    case EK::Name: act->slots[size_t(static_cast<const NameExpr*>(target)->slot)] = v; return;
    case EK::NSRef: {
      auto* n = static_cast<const NSRefExpr*>(target);
      Activation* a = act;
      for (int h = n->hops; h > 0; --h) a = a->parent;
      const Val& ns = a->slots[size_t(n->slot)];
      if (!ns.is(Kind::Namespace)) raise(ErrorKind::Runtime, "cannot assign attribute on non-namespace object");
      ns.as<NsObj>()->m.set(n->attr_val, v);
      return;
    }
    case EK::Tuple: {
      auto* t = static_cast<const SeqExpr*>(target);
      const size_t want = t->items.size();
      std::vector<Val> items;
      if (v.is_seq()) {
        const size_t n = seq_size(v);
        for (size_t i = 0; i < n && i <= want; ++i) items.push_back(seq_at(v, i));
      } else {
        if (v.is_undef() || v.is_missing() || v.is_str() || v.is_dict() || v.is(Kind::Range) || v.is(Kind::Iter) ||
            v.is(Kind::View)) {
          PyIter it(v);
          Val x;
          while (items.size() <= want && it.next(x)) items.push_back(std::move(x));
        } else {
          raise(ErrorKind::Type, std::string("cannot unpack non-iterable ") + tname(v) + " object");
        }
      }
      if (items.size() > want) raise(ErrorKind::Value, "too many values to unpack (expected " + std::to_string(want) + ")");
      if (items.size() < want)
        raise(ErrorKind::Value, "not enough values to unpack (expected " + std::to_string(want) + ", got " +
                                    std::to_string(items.size()) + ")");
      for (size_t i = 0; i < want; ++i) assign(t->items[i].get(), items[i], act);
      return;
    }
    default: raise(ErrorKind::Runtime, "invalid assignment target");
  }
}

// ---------------------------------------------------------------- expressions
void Interp::eval_args(const CallArgs& a, Activation* act, Args& out) {
  // Python evaluates positional (and *starred) arguments, then keywords.
  out.pos.reserve(a.args.size());
  for (const auto& x : a.args) out.pos.push_back(eval(x.get(), act));
  if (a.dyn_args) {
    const Val d = eval(a.dyn_args.get(), act);
    if (d.is_none() || d.is_number() || d.is(Kind::Namespace) || d.is(Kind::Macro) || d.is(Kind::Builtin))
      raise(ErrorKind::Type, std::string("Value after * must be an iterable, not ") + tname(d));
    PyIter it(d);
    Val x;
    while (it.next(x)) out.pos.push_back(std::move(x));
  }
  for (const auto& [k, v] : a.kwargs) out.kw.emplace_back(Val::str_ref(k), eval(v.get(), act));
  if (a.dyn_kwargs) {
    const Val d = eval(a.dyn_kwargs.get(), act);
    if (!d.is_dict()) raise(ErrorKind::Type, std::string("argument after ** must be a mapping, not ") + tname(d));
    const size_t n = dict_size(d);
    for (size_t i = 0; i < n; ++i) {
      Val k = dict_key(d, i);
      if (!k.is_str()) raise(ErrorKind::Type, "keywords must be strings");
      if (out.has(k.sv()))
        raise(ErrorKind::Type, "got multiple values for keyword argument " + py_repr_str(k.sv()));
      out.kw.emplace_back(std::move(k), dict_value(d, i));
    }
  }
}

Val Interp::eval_filter(const FilterExpr* f, Activation* act, const Val* block_value) {
  Val v;
  if (!f->node)
    v = *block_value;
  else if (f->node->k == EK::Filter && block_value)
    v = eval_filter(static_cast<const FilterExpr*>(f->node.get()), act, block_value);
  else
    v = eval(f->node.get(), act);
  Args args;
  eval_args(f->a, act, args);
  return call_filter(*this, f->id, f->name, v, args);
}

bool Interp::eval_test(const Expr* test, const Val& item, const Expr* target, Activation* act) {
  assign(target, item, act);
  return truthy(eval(test, act));
}

// `a + b + c + ...`: strings accumulate in one buffer instead of one
// temporary per step. Evaluation order and error behaviour stay Python's:
// the first non-string operand falls back to the generic operator.
namespace {

// Operands of a left-associated `a + b + c + ...` chain, in order.
int add_chain(const BinExpr* e, const Expr** chain, int cap) {
  int n = 0;
  const Expr* x = e;
  while (x->k == EK::Add && n < cap - 1) {
    chain[n++] = static_cast<const BinExpr*>(x)->right.get();
    x = static_cast<const BinExpr*>(x)->left.get();
  }
  chain[n++] = x;
  std::reverse(chain, chain + n);
  return n;
}

}  // namespace

Val Interp::eval_add(const BinExpr* e, Activation* act) {
  const Expr* chain[32];
  const int n = add_chain(e, chain, 32);
  return add_rest(eval(chain[0], act), chain, 1, n, act);
}

// `{{ a + b + ... }}`: plain-string operands go straight into the output. At
// the first operand that is not a plain string, the text written so far is
// taken back and Python's `+` continues from it (raising, or building Markup).
void Interp::emit_add(const BinExpr* e, Activation* act) {
  const Expr* chain[32];
  const int n = add_chain(e, chain, 32);
  Val first = eval(chain[0], act);
  if (!first.is_str() || first.markup()) {
    append_str(*out_, add_rest(std::move(first), chain, 1, n, act));
    return;
  }
  std::string& out = *out_;
  const size_t start = out.size();
  out += first.sv();
  for (int i = 1; i < n; ++i) {
    Val r = eval(chain[i], act);
    if (r.is_str() && !r.markup()) {
      out += r.sv();
      continue;
    }
    Val acc = Val::str(out.substr(start));
    out.resize(start);
    acc = py_add(acc, r);
    append_str(out, add_rest(std::move(acc), chain, i + 1, n, act));
    return;
  }
}

Val Interp::add_rest(Val acc, const Expr* const* chain, int from, int n, Activation* act) {
  std::string buf;
  bool buffering = false;
  for (int i = from; i < n; ++i) {
    Val r = eval(chain[i], act);
    const bool acc_plain = buffering || (acc.is_str() && !acc.markup());
    if (acc_plain && r.is_str() && !r.markup()) {
      if (!buffering) {
        buf.reserve(acc.sv().size() + r.sv().size() + 64);
        buf.assign(acc.sv());
        buffering = true;
      }
      buf += r.sv();
      continue;
    }
    if (buffering) {
      acc = Val::str(std::move(buf));
      buf.clear();
      buffering = false;
    }
    acc = py_add(acc, r);
  }
  if (buffering) acc = Val::str(std::move(buf));
  return acc;
}

Val Interp::eval(const Expr* e, Activation* act) {
  switch (e->k) {
    case EK::Const: return static_cast<const ConstExpr*>(e)->v;
    case EK::Name: {
      auto* n = static_cast<const NameExpr*>(e);
      Activation* a = act;
      for (int h = n->hops; h > 0; --h) a = a->parent;
      const Val& v = a->slots[size_t(n->slot)];
      if (v.is_missing()) return Val::undef_name(n->name);
      return v;
    }
    case EK::Getattr: {
      auto* g = static_cast<const GetattrExpr*>(e);
      const Val obj = eval(g->node.get(), act);
      return get_attr(*this, obj, g->attr_val, &g->hint);
    }
    case EK::Getitem: {
      auto* g = static_cast<const GetitemExpr*>(e);
      const Val obj = eval(g->node.get(), act);
      if (g->arg->k == EK::Slice) {  // slices bypass the sandbox's getitem
        auto* s = static_cast<const SliceExpr*>(g->arg.get());
        const Val a = s->start ? eval(s->start.get(), act) : Val::none();
        const Val b = s->stop ? eval(s->stop.get(), act) : Val::none();
        const Val c = s->step ? eval(s->step.get(), act) : Val::none();
        return get_slice(obj, a, b, c);
      }
      const Val key = eval(g->arg.get(), act);
      return get_item(*this, obj, key);
    }
    case EK::Call: {
      auto* c = static_cast<const CallExpr*>(e);
      if (c->node->k == EK::Getattr) {
        // obj.method(...): resolve the attribute first (Python order), and call
        // builtin methods directly instead of materialising a bound method.
        auto* g = static_cast<const GetattrExpr*>(c->node.get());
        const Val obj = eval(g->node.get(), act);
        const int mid = builtin_method_id(obj, g->hint);
        if (mid >= 0) {
          Args args;
          eval_args(c->a, act, args);
          return call_method(*this, obj, mid, args);
        }
        const Val fn = get_attr(*this, obj, g->attr_val, &g->hint);
        Args args;
        eval_args(c->a, act, args);
        return call(fn, args);
      }
      const Val fn = eval(c->node.get(), act);
      Args args;
      eval_args(c->a, act, args);
      return call(fn, args);
    }
    case EK::Filter: return eval_filter(static_cast<const FilterExpr*>(e), act, nullptr);
    case EK::Test: {
      auto* t = static_cast<const FilterExpr*>(e);
      const Val v = eval(t->node.get(), act);
      Args args;
      eval_args(t->a, act, args);
      return Val::boolean(call_test(*this, t->id, t->name, v, args));
    }
    case EK::Not: return Val::boolean(!truthy(eval(static_cast<const UnaryExpr*>(e)->node.get(), act)));
    case EK::Neg: return py_neg(eval(static_cast<const UnaryExpr*>(e)->node.get(), act));
    case EK::Pos: return py_pos(eval(static_cast<const UnaryExpr*>(e)->node.get(), act));
    case EK::And: {
      auto* b = static_cast<const BinExpr*>(e);
      Val l = eval(b->left.get(), act);
      if (!truthy(l)) return l;
      return eval(b->right.get(), act);
    }
    case EK::Or: {
      auto* b = static_cast<const BinExpr*>(e);
      Val l = eval(b->left.get(), act);
      if (truthy(l)) return l;
      return eval(b->right.get(), act);
    }
    case EK::Add: return eval_add(static_cast<const BinExpr*>(e), act);
    case EK::Sub:
    case EK::Mul:
    case EK::Div:
    case EK::FloorDiv:
    case EK::Mod:
    case EK::Pow: {
      auto* b = static_cast<const BinExpr*>(e);
      const Val l = eval(b->left.get(), act);
      const Val r = eval(b->right.get(), act);
      return py_arith(e->k, l, r);
    }
    case EK::Compare: {
      auto* c = static_cast<const CompareExpr*>(e);
      Val left = eval(c->expr.get(), act);
      for (const auto& [op, rhs] : c->ops) {
        Val right = eval(rhs.get(), act);
        bool r = false;
        switch (op) {
          case CmpOp::Eq: r = py_eq(left, right); break;
          case CmpOp::Ne: r = !py_eq(left, right); break;
          case CmpOp::Lt: r = py_compare(0, left, right); break;
          case CmpOp::Le: r = py_compare(1, left, right); break;
          case CmpOp::Gt: r = py_compare(2, left, right); break;
          case CmpOp::Ge: r = py_compare(3, left, right); break;
          case CmpOp::In: r = py_contains(right, left); break;
          case CmpOp::NotIn: r = !py_contains(right, left); break;
        }
        if (!r) return Val::boolean(false);
        left = std::move(right);
      }
      return Val::boolean(true);
    }
    case EK::Concat: {
      std::string s;
      for (const auto& n : static_cast<const ConcatExpr*>(e)->nodes) {
        const Val v = eval(n.get(), act);
        append_str(s, v);
      }
      return Val::str(std::move(s));
    }
    case EK::CondExpr: {
      auto* c = static_cast<const CondExpr*>(e);
      if (truthy(eval(c->test.get(), act))) return eval(c->expr1.get(), act);
      if (c->expr2) return eval(c->expr2.get(), act);
      return Val::undef_hint(c->hint);
    }
    case EK::List:
    case EK::Tuple: {
      auto* s = static_cast<const SeqExpr*>(e);
      std::vector<Val> v;
      v.reserve(s->items.size());
      for (const auto& i : s->items) v.push_back(eval(i.get(), act));
      return e->k == EK::List ? make_list(std::move(v)) : make_tuple(std::move(v));
    }
    case EK::Dict: {
      MapData m;
      for (const auto& [k, v] : static_cast<const DictExpr*>(e)->items) {
        Val key = eval(k.get(), act);
        Val val = eval(v.get(), act);
        require_hashable(key);
        m.set(std::move(key), std::move(val));
      }
      return make_dict(std::move(m));
    }
    case EK::Slice:
    case EK::NSRef: break;
  }
  raise(ErrorKind::Runtime, "invalid expression");
}

// ---------------------------------------------------------------- calls
Val Interp::call(const Val& fn, Args& args) {
  switch (fn.kind()) {
    case Kind::Macro: return call_macro(fn.as<MacroObj>(), args);
    case Kind::Builtin: return call_builtin(*this, fn.builtin_id(), args);
    case Kind::Method: return call_method(*this, *fn.as<MethodObj>(), args);
    case Kind::Joiner: {
      auto* j = fn.as<JoinerObj>();
      if (!args.pos.empty() || !args.kw.empty())
        raise(ErrorKind::Type, "Joiner.__call__() takes 1 positional argument but " +
                                   std::to_string(args.pos.size() + 1) + " were given");
      if (!j->used) {
        j->used = true;
        return Val::str_ref("");
      }
      return j->sep;
    }
    case Kind::Missing:
    case Kind::Undefined: raise_undefined(fn);
    case Kind::Loop: raise(ErrorKind::Type, "The loop must have the 'recursive' marker to be called recursively.");
    default: raise(ErrorKind::Type, std::string("'") + tname(fn) + "' object is not callable");
  }
}

// jinja2.runtime.Macro.__call__ followed by the generated macro function.
Val Interp::call_macro(MacroObj* m, Args& a) {
  const MacroDef& d = *m->def;
  if (depth_ >= kMaxMacroDepth) raise(ErrorKind::Recursion, "maximum recursion depth exceeded");
  const size_t argc = d.args.size();
  auto* act = Activation::make(size_t(d.nslots), m->closure);
  Hold hold{act};
  Val* slots = act->slots;
  const size_t npos = std::min(a.pos.size(), argc);
  for (size_t i = 0; i < npos; ++i) slots[d.args[i]->slot] = a.pos[i];
  bool found_caller = false;
  if (npos != argc) {
    for (size_t i = npos; i < argc; ++i) {
      const std::string& name = d.args[i]->name;
      Val v;
      if (a.take(name, v)) slots[d.args[i]->slot] = std::move(v);
      if (name == "caller") found_caller = true;
    }
  } else {
    found_caller = d.explicit_caller;
  }
  Val caller;
  const bool pass_caller = d.caller && !found_caller;
  if (pass_caller) {
    if (!a.take("caller", caller) || caller.is_none()) caller = Val::undef_hint("No caller defined");
  }
  Val kwargs, varargs;
  if (d.catch_kwargs) {
    MapData md;
    for (auto& [k, v] : a.kw) md.set(k, v);
    kwargs = make_dict(std::move(md));
  } else if (!a.kw.empty()) {
    if (a.has("caller"))
      raise(ErrorKind::Type, "macro " + py_repr_str(d.name) +
                                 " was invoked with two values for the special caller argument. This is most likely a "
                                 "bug.");
    raise(ErrorKind::Type,
          "macro " + py_repr_str(d.name) + " takes no keyword argument " + py_repr_str(a.kw[0].first.sv()));
  }
  if (d.catch_varargs) {
    std::vector<Val> rest;
    for (size_t i = argc; i < a.pos.size(); ++i) rest.push_back(a.pos[i]);
    varargs = make_tuple(std::move(rest));
  } else if (a.pos.size() > argc) {
    raise(ErrorKind::Type,
          "macro " + py_repr_str(d.name) + " takes not more than " + std::to_string(argc) + " argument(s)");
  }

  if (pass_caller) act->slots[size_t(d.caller_slot)] = caller;
  if (d.catch_kwargs) act->slots[size_t(d.kwargs_slot)] = kwargs;
  if (d.catch_varargs) act->slots[size_t(d.varargs_slot)] = varargs;
  run_init(d.init, act);
  const size_t first_default = argc - d.defaults.size();
  for (size_t i = 0; i < argc; ++i) {
    const size_t slot = size_t(d.args[i]->slot);
    if (!act->slots[slot].is_missing()) continue;
    Val v = i >= first_default
                ? eval(d.defaults[i - first_default].get(), act)
                : Val::undef_hint_owned("parameter " + py_repr_str(d.args[i]->name) + " was not provided");
    act->slots[slot] = std::move(v);
  }
  ++depth_;
  Flow flow;
  Val r;
  try {
    r = render_body_to_str(d.body, act, &flow);
  } catch (...) {
    --depth_;
    throw;
  }
  --depth_;
  return r;
}

}  // namespace espetero::tmpl
