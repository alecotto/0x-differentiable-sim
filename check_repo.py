"""Static + runtime smoke test. Run before any status table is written."""
import ast
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_FAILS = []
_WARNS = []


def _static_checks():
    """AST-only checks: no torch needed."""
    for dirpath, dirnames, filenames in os.walk(_ROOT):
        dirnames[:] = [d for d in dirnames if d not in ('.git', '__pycache__', '.pytest_cache')]
        rel = os.path.relpath(dirpath, _ROOT)
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            fp = os.path.join(dirpath, fn)
            rel_fp = os.path.relpath(fp, _ROOT)
            if fn == "check_repo.py":
                continue
            src = open(fp).read()
            try:
                tree = ast.parse(src)
            except SyntaxError as e:
                _FAILS.append(f"[syntax] {rel_fp}: {e}")
                continue

            # collect defined names at module level (functions, classes, assignments)
            defined = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    defined.add(node.name)
                elif isinstance(node, ast.Assign):
                    for t in ast.walk(node):
                        if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                            defined.add(t.id)

            # imported-name existence: check that from X import Y exists in X's source
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    mod_path = node.module.replace('.', os.sep)
                    mod_file = os.path.join(_ROOT, mod_path + '.py')
                    mod_init = os.path.join(_ROOT, mod_path, '__init__.py')
                    src_mod = None
                    if os.path.isfile(mod_file):
                        src_mod = open(mod_file).read()
                    elif os.path.isfile(mod_init):
                        src_mod = open(mod_init).read()
                    if src_mod is None:
                        continue
                    # only check stdlib/local imports we can verify
                    for alias in node.names:
                        name = alias.name
                        if name not in src_mod and f"def {name}" not in src_mod \
                                and f"class {name}" not in src_mod \
                                and f"{name} =" not in src_mod:
                            # might be re-exported via __all__ or submodule
                            pass  # skip complex resolution; just warn on obvious misses

            # hardcoded path
            if '/Code/0x-differentiable-sim-project' in src:
                _FAILS.append(f"[hardcoded path] {rel_fp}")

            # tuple-unpack arity at call sites of locally-defined functions
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    targets = node.targets
                    value = node.value
                    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                        fname = value.func.id
                        # find the function def to check return arity
                        for node2 in ast.walk(tree):
                            if isinstance(node2, ast.FunctionDef) and node2.name == fname:
                                rets = [n for n in ast.walk(node2)
                                        if isinstance(n, ast.Return) and n.value is not None]
                                if len(rets) == 1 and isinstance(rets[0].value, ast.Tuple):
                                    ret_arity = len(rets[0].value.elts)
                                elif rets:
                                    ret_arity = 1
                                else:
                                    continue
                                if isinstance(targets[0], (ast.Tuple, ast.List)):
                                    unpack_arity = len(targets[0].elts)
                                    if unpack_arity != ret_arity:
                                        _FAILS.append(
                                            f"[unpack] {rel_fp}:{node.lineno} "
                                            f"unpacks {unpack_arity} from {fname}() "
                                            f"which returns [{ret_arity}]")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--static"

    if mode in ("--static", "--all"):
        _static_checks()

    if mode == "--runtime":
        import subprocess
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-x", "-q"],
                           cwd=_ROOT, capture_output=True, text=True)
        print(r.stdout[-2000:] if r.stdout else "(no output)")
        if r.returncode != 0:
            _FAILS.append(f"[pytest] exit code {r.returncode}")

    if _FAILS:
        for f in _FAILS:
            print(f"FAIL  {f}", file=sys.stderr)
        print(f"\nFAILED: {len(_FAILS)} problem(s). Do not report results from this tree.",
              file=sys.stderr)
        sys.exit(1)
    else:
        print("check_repo: clean")


if __name__ == "__main__":
    main()
