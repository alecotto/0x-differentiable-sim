"""Static analysis gate. Run before ANY status table is written.

Usage: python check_repo.py [--static]
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FAILS = []


def _collect_defs(tree):
    """Return dict {name: (min_args, max_args)} for all FunctionDef and ClassDef."""
    sigs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            n_pos = len([x for x in a.posonlyargs + a.args if x.arg not in ('self', 'cls')])
            n_kwonly = len(a.kwonlyargs)
            n_defaults = len(a.defaults)
            min_a = max(0, n_pos - n_defaults)
            max_a = 9999 if a.vararg is not None or n_kwonly > 0 else n_pos + n_kwonly
            if a.kwarg is not None:
                max_a = 9999
            sigs[node.name] = (min_a, max_a)
    return sigs


def _load_trees():
    trees = {}
    for dp, dn, fns in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in ('.git', '__pycache__', '.pytest_cache')]
        for fn in fns:
            if not fn.endswith('.py'):
                continue
            fp = os.path.join(dp, fn)
            rel = os.path.relpath(fp, ROOT)
            src = open(fp).read()
            try:
                trees[rel] = (ast.parse(src), src)
            except SyntaxError as e:
                FAILS.append(f"[syntax] {rel}: {e}")
    return trees


def _resolve(mod_name, trees):
    parts = mod_name.split('.')
    fp = os.path.join(ROOT, *parts) + '.py'
    rel = os.path.relpath(fp, ROOT)
    if rel in trees:
        return rel
    fp = os.path.join(ROOT, *parts, '__init__.py')
    rel = os.path.relpath(fp, ROOT)
    if rel in trees:
        return rel
    # try walking up (e.g., diffsim.algo.shac → diffsim/algo/shac.py)
    for i in range(len(parts), 0, -1):
        candidate_rel = os.sep.join(parts[:i]) + '.py'
        if candidate_rel in trees:
            return candidate_rel
        candidate_init = os.sep.join(parts[:i] + ['__init__.py'])
        if candidate_init in trees:
            return candidate_init
    return None


def main():
    trees = _load_trees()

    # Build global symbol table
    symbols = {}
    for rel, (tree, src) in trees.items():
        for name, sig in _collect_defs(tree).items():
            symbols[name] = sig

    for rel, (tree, src) in sorted(trees.items()):
        skip_path = (rel == 'check_repo.py')

        if not skip_path and '/Code/' in src:
            FAILS.append(f"[hardcoded path] {rel}")

        # ---- imported-name existence ----
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            mod = node.module
            if not mod:
                continue
            mod_rel = _resolve(mod, trees)
            if mod_rel is None:
                continue
            mod_tree = trees[mod_rel][0]
            mod_defs = set()
            for n in ast.walk(mod_tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    mod_defs.add(n.name)
                elif isinstance(n, ast.Assign):
                    for t in ast.walk(n):
                        if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                            mod_defs.add(t.id)
                elif isinstance(n, ast.ImportFrom):
                    for a in n.names:
                        mod_defs.add(a.asname or a.name)
                elif isinstance(n, ast.Import):
                    for a in n.names:
                        mod_defs.add((a.asname or a.name).split('.')[0])

            # also check __init__.py re-exports if this IS an __init__
            init_rel = os.path.join(os.path.dirname(mod_rel), '__init__.py').replace(os.sep, '/')
            if init_rel in trees:
                init_tree = trees[init_rel][0]
                for n in ast.walk(init_tree):
                    if isinstance(n, ast.ImportFrom):
                        for a in n.names:
                            mod_defs.add(a.asname or a.name)

            for alias in node.names:
                nm = alias.asname or alias.name
                base = nm.split('.')[0]
                if base not in mod_defs:
                    # might be importing a submodule directly
                    sub_resolve = _resolve(f"{mod}.{base}", trees)
                    if sub_resolve is None:
                        FAILS.append(
                            f"[missing import] {rel}:{node.lineno} "
                            f"'{base}' not found in '{mod}'")

if __name__ == "__main__":
    main()
    if FAILS:
        for f in FAILS:
            print(f"FAIL  {f}", file=sys.stderr)
        print(f"\nFAILED: {len(FAILS)}.", file=sys.stderr)
        sys.exit(1)
    else:
        print("check_repo: clean")
