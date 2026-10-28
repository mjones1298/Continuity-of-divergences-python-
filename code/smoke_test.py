"""Smoke test for code/: runs a representative fast script from every folder and checks that it reproduces
the shipped numbers.

The whole code/ tree is first copied to a temporary directory and every script is run there, from an
unrelated working directory, so the shipped results are never overwritten and the scripts are shown to find
their modules and data relative to their own location. Each test then compares what the script produced
(a JSON file or its printed output) with the shipped JSON / output files, or with the numbers quoted in the
book. Total runtime is about 8 minutes on 2 cores; every single test takes under 2 minutes.

Usage:  python3 smoke_test.py [-k SUBSTRING] [--keep]      (or: bash smoke_test.sh, or: make code-test)
Exit status 0 iff every test passes.
"""
import argparse, json, math, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

CODE = Path(__file__).resolve().parent
SKIP_KEYS = ('time', 'sec', 'elapsed', 'wall', 't_est', 't_q', 'runtime', 'date')
FLOAT = re.compile(r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?')


# ------------------------------------------------------------------ comparison helpers
def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)          # mpmath values are stored as strings
        except ValueError:
            return None
    return None


def compare_json(new, ref, rtol, atol=1e-12, path='', out=None):
    """Recursively compare two JSON trees; numbers to rtol/atol, strings exactly. Returns list of mismatches."""
    out = [] if out is None else out
    if isinstance(ref, dict):
        if not isinstance(new, dict):
            out.append(f'{path}: type differs'); return out
        for k in ref:
            if any(s in k.lower() for s in SKIP_KEYS):
                continue
            if k not in new:
                out.append(f'{path}/{k}: missing'); continue
            compare_json(new[k], ref[k], rtol, atol, f'{path}/{k}', out)
        return out
    if isinstance(ref, list):
        if not isinstance(new, list) or len(new) != len(ref):
            out.append(f'{path}: list length differs'); return out
        for i, (a, b) in enumerate(zip(new, ref)):
            compare_json(a, b, rtol, atol, f'{path}[{i}]', out)
        return out
    a, b = _num(new), _num(ref)
    if a is not None and b is not None:
        if math.isnan(b):
            if not math.isnan(a):
                out.append(f'{path}: {a} vs nan')
        elif math.isinf(b) or math.isinf(a):
            if a != b:
                out.append(f'{path}: {a} vs {b}')
        elif abs(a - b) > atol + rtol * abs(b):
            out.append(f'{path}: {a!r} vs {b!r}')
        return out
    if new != ref:
        out.append(f'{path}: {new!r} vs {ref!r}')
    return out


def numbers(text):
    return [float(x) for x in FLOAT.findall(text)]


def compare_text(new, ref, rtol, atol=1e-12):
    """Compare all numbers in two printed outputs, in order."""
    a, b = numbers(new), numbers(ref)
    if len(a) != len(b):
        return [f'{len(a)} numbers printed, {len(b)} expected']
    return [f'number {i}: {x!r} vs {y!r}' for i, (x, y) in enumerate(zip(a, b)) if abs(x - y) > atol + rtol * abs(y)]


def expect(text, pairs, rtol):
    """pairs: (regex with one group, expected value). Each regex must match; group compared to value."""
    bad = []
    for rx, val in pairs:
        m = re.search(rx, text)
        if not m:
            bad.append(f'pattern {rx!r} not found'); continue
        x = float(m.group(1).rstrip(',;:)'))
        if abs(x - val) > rtol * abs(val):
            bad.append(f'{rx!r}: {x} vs {val}')
    return bad


# ------------------------------------------------------------------ tests
# Each test: (folder-relative script, args, prepare(tmp), check(tmp, stdout) -> list of problems)
def js(tmp, rel):
    return json.load(open(tmp / rel))


def ref(rel):
    return json.load(open(CODE / rel))


def check_json(rel, rtol):
    return lambda tmp, out: compare_json(js(tmp, rel), ref(rel), rtol)


def check_out(rel_out, rtol):
    return lambda tmp, out: compare_text(out, (CODE / rel_out).read_text(), rtol)


def check_expect(pairs, rtol):
    return lambda tmp, out: expect(out, pairs, rtol)


def remove(*rels):
    def prep(tmp):
        for r in rels:
            (tmp / r).unlink()
    return prep


def check_below(rx, bound):
    def chk(tmp, out):
        m = re.search(rx, out)
        if not m:
            return [f'pattern {rx!r} not found']
        return [] if float(m.group(1)) < bound else [f'{m.group(1)} is not below {bound}']
    return chk


def check_r2(tmp, out):
    # Theorem 4.4.6 is asymptotically attained: |dF|/bound equals sqrt(1-s) exactly
    pairs = re.findall(r'\|dF\|/bound=(\S+), sqrt\(1-s\)=(\S+)', out)
    if len(pairs) < 6:
        return [f'only {len(pairs)} rows printed']
    return [f'{a} vs {b}' for a, b in pairs if abs(float(a) - float(b)) > 1e-11]


def check_t3_subset(tmp, out):
    rows = {r['p']: r['delta_star'] for r in ref('ch01_ch08_noisy_gates/multiqubit_window/results/t3.json')['window']['n2_local_ZZ']['rows']}
    probs = []
    for p in (1e-2, 1e-3):
        m = re.search(rf'p={p:g} delta_star=(\S+)', out)
        if not m:
            probs.append(f'p={p:g}: not printed'); continue
        x = float(m.group(1))
        if abs(x - rows[p]) > 1e-12 * rows[p]:
            probs.append(f'p={p:g}: {x} vs {rows[p]}')
    return probs


T3_SNIPPET = r'''
import sys
sys.path.insert(0, sys.argv[1])
import t3_multiqubit as T
for p in ("1e-2", "1e-3"):
    w = T.pauli_weights_local(p, 2)
    print(f"p={float(p):g} delta_star={float(T.delta_star(w, (3, 3), 2))!r}")
'''

CH = 'ch01_ch08_noisy_gates'
TESTS = [
    # code/common
    ('common/sharp_alpha.py', [], None, check_expect([   # D^# of two binary symmetric channels = classical Renyi value
        (r'BSC t 1/2 D# (\S+)', 0.17435338714), (r'BSC t 3/4 D# (\S+)', 0.14058910574)], 1e-5)),
    # Chapters 1 and 8
    (f'{CH}/adaptive_qfi/t1_adaptive_qfi.py', [], None, check_json(f'{CH}/adaptive_qfi/results/t1.json', 1e-6)),
    (f'{CH}/hierarchy/t4_hierarchy.py', [], None, check_json(f'{CH}/hierarchy/results/t4.json', 1e-6)),
    (f'{CH}/crossover/t2_crossover.py', [], None, check_json(f'{CH}/crossover/results/t2.json', 1e-6)),
    (f'{CH}/crossover/t2b_integrate.py', [], None, lambda tmp, out:
        check_json(f'{CH}/crossover/results/t2b_merged.json', 1e-12)(tmp, out)
        + check_json(f'{CH}/crossover/results/summary.json', 1e-12)(tmp, out)),
    (f'{CH}/multiqubit_window/[t3 subset]', [], None, check_t3_subset),
    (f'{CH}/proof_support/consts.py', [], None, check_expect([
        (r"f\*/4=\s*(\S+)", 0.149587125278), (r"p=0.01: sup_k .*= (\S+)", 0.148832),
        (r"p=0.001: sup_k .*= (\S+)", 0.149512), (r"min over grid of N_min\*eta\^2\*lam~ = (\S+)", 0.02465)], 2e-4)),
    (f'{CH}/proof_support/p2_unitary.py', [], None, check_expect([
        (r"p=0.01: .*p\*sup = (\S+)", 0.384904), (r"p=0.001: .*p\*sup = (\S+)", 0.384900),
        (r"SDP check at unitary optimum: p\*sup_E = (\S+)", 0.3849038)], 2e-6)),
    (f'{CH}/proof_support/p1_modulus_mp.py', [], None, check_expect([   # Section 8.3: slope / (d/(4 lambda))
        (r"n=1 p=0.01: .*4lam/d = (\S+)", 1.0000000), (r"n=1 p=0.001: .*4lam/d = (\S+)", 1.0000034),
        (r"n=2 p=0.01: .*4lam/d = (\S+)", 0.9999816), (r"n=2 p=0.001: .*4lam/d = (\S+)", 0.99999995)], 1e-7)),
    (f'{CH}/proof_support/chk_thmL.py', [], None, check_expect([
        (r"Theorem L: max \|dF\|/bound = (\S+)", 0.045960854107698776),
        (r"delta<=p: (\S+)", 0.1462860443903723)], 1e-8)),
    (f'{CH}/proof_support/p2_resolve.py', ['{tmp}/' + f'{CH}/proof_support/p2_restart0_x.npy', '1e-3'], None,
        check_expect([   # Section 8.3: the p*Sigma = 5.34 of the double-precision search is an artifact
            (r"the search objective\): p\*sup = (\S+)", 5.338649), (r"mp FD h= 1e-30: p\*sup = (\S+)", 0.384601970),
            (r"\(the 5.34 direction\) E, h=1e-20: p\*dF/de\(\+\) = (\S+)", 0.248223646)], 1e-6)),
    (f'{CH}/proof_support/crosscheck_theorems.py', [], None,
        check_json(f'{CH}/proof_support/results/crosscheck_theorems.json', 1e-6)),
    (f'{CH}/amplitude_damping/e2_reset_noise.py', [], None,
        check_json(f'{CH}/amplitude_damping/results/e2.json', 1e-6)),
    (f'{CH}/first_order/m1_chernoff.py', [], None, lambda tmp, out:
        check_json(f'{CH}/first_order/results_chernoff.json', 1e-8)(tmp, out)
        + compare_text((tmp / f'{CH}/first_order/m1_chernoff.out').read_text(),
                       (CODE / f'{CH}/first_order/m1_chernoff.out').read_text(), 1e-6)),
    (f'{CH}/first_order/m1_first_order.py', [], None, check_json(f'{CH}/first_order/results_first_order.json', 1e-8)),
    (f'{CH}/biased_noise/b1_closed_forms.py', [], None, check_json(f'{CH}/biased_noise/results/b1.json', 1e-8)),
    (f'{CH}/certification/validate_hp.py', [], None, check_json(f'{CH}/certification/results/validate_hp.json', 1e-6)),
    (f'{CH}/certification/analyze_s1.py', [], None, check_json(f'{CH}/certification/results/s1_summary.json', 1e-9)),
    (f'{CH}/certification/analyze_s2.py', [], None, check_json(f'{CH}/certification/results/s2_summary.json', 1e-9)),
    (f'{CH}/biased_noise/b2_adaptive_qfi.py', [], remove(f'{CH}/biased_noise/results/b2.json'),
        check_json(f'{CH}/biased_noise/results/b2.json', 1e-6)),
    (f'{CH}/figures/make_fig4_cert.py', [], None, lambda tmp, out:
        [] if (tmp / f'{CH}/figures/fig4_cert.pdf').stat().st_size > 10000 else ['fig4_cert.pdf not written']),
    (f'{CH}/noise_floor/e8c_interior_modulus.py', [], None, check_json(f'{CH}/noise_floor/results/e8c.json', 1e-6)),
    (f'{CH}/noise_floor/e6_figures.py', [], None, lambda tmp, out:
        [] if (tmp / f'{CH}/noise_floor/figures/e6_noise_floor.png').exists() else ['figure not written']),
    # Chapter 4
    (f'ch04_continuity/pair_bounds/e1_jones_cor33.py', [], None,
        check_json('ch04_continuity/pair_bounds/results/e1.json', 1e-5)),
    (f'ch04_continuity/dichotomy/chk_p3b.py', [], None, check_expect([(r"max F/bound = (\S+)", 0.31083091768549437)], 1e-6)),
    (f'ch04_continuity/v6_verification/r2.py', [], None, check_r2),
    # Chapter 5
    ('ch05_shadows/hockey_lipschitz/e5_hockey.py', [], None, lambda tmp, out: compare_json(
        js(tmp, 'ch05_shadows/hockey_lipschitz/results/e5.json'), ref('ch05_shadows/hockey_lipschitz/results/e5.json'), 1e-5, 1e-8)),
    # Chapter 7
    ('ch07_chernoff/ad_table.py', [], None, check_expect([
        (r"D=0.10\s+xi_prod=(\S+)", 5.077e-3), (r"D=0.03\s+xi_prod=(\S+)", 4.506e-4),
        (r"D=0.01\s+xi_prod=(\S+)", 5.001e-5), (r"D=0.10 .*-2ln mu=(\S+)", 2.252e-2),
        (r"D=0.30 n=3 adapt=(\S+)", 0.2815), (r"D=0.30 n=3 adapt=\S+ par=(\S+)", 0.2854),
        (r"D=0.30 n=3 .*\|1>=(\S+)", 0.3020)], 1e-3)),
    # independent checks and referee re-computations
    ('checks/independent_C1_C7/c2.py', [], None, check_below(r"max \|F-pred\| over grid (\S+)", 1e-50)),
    ('checks/independent_K1_K4/k2.py', [], None, check_expect([
        (r"p 0.1 delta\* \S+ delta\*/sqrt\(p\) (\S+)", 1.0719063), (r"p 0.01 delta\* \S+ delta\*/sqrt\(p\) (\S+)", 0.97800488),
        (r"p 0.001 delta\* \S+ delta\*/sqrt\(p\) (\S+)", 0.96957166), (r"p 1e-4 delta\* \S+ delta\*/sqrt\(p\) (\S+)", 0.96873687),
        (r"limit delta\*/sqrt\(p\) = (\S+)", 0.9686442097)], 1e-7)),
    ('checks/referee_letter/check4_ad.py', [], None, check_out('checks/referee_letter/check4.out', 1e-9)),
    ('checks/referee_letter/check5_constants.py', [], None, check_out('checks/referee_letter/check5.out', 1e-9)),
    ('checks/referee_letter/check1b_L2.py', [], None, check_out('checks/referee_letter/check1b.out', 1e-9)),
    ('checks/referee_prereview/rld.py', [], None, check_expect([(r"p=1e-04: .*F_RLD\*p=(\S+)", 1.000)], 1e-3)),
]


def run(tmp, script, args):
    cwd = tempfile.mkdtemp()            # unrelated working directory
    try:
        if script.endswith('[t3 subset]'):
            cmd = [sys.executable, '-c', T3_SNIPPET, str(tmp / CH / 'multiqubit_window')]
        else:
            cmd = [sys.executable, str(tmp / script)] + [x.replace('{tmp}', str(tmp)) for x in args]
        env = dict(os.environ, MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1')
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300, env=env)
        return r.returncode, r.stdout, r.stderr
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', default='', help='run only tests whose script path contains this string')
    ap.add_argument('--keep', action='store_true', help='keep the temporary copy and print its location')
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix='code_smoke_'))
    shutil.copytree(CODE, tmp, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    failures = 0
    print(f'{"script":<62} {"status":<6} {"time":>7}')
    try:
        for script, args, prep, check in TESTS:
            if a.k not in script:
                continue
            if prep:
                prep(tmp)
            t0 = time.time()
            rc, out, err = run(tmp, script, args)
            dt = time.time() - t0
            probs = [f'exit status {rc}: ' + err.strip().splitlines()[-1] if err.strip() else f'exit status {rc}'] if rc else []
            if not rc and check:
                try:
                    probs += check(tmp, out)
                except Exception as e:     # a missing output file etc.
                    probs.append(f'check failed: {e!r}')
            status = 'ok' if not probs else 'FAIL'
            failures += bool(probs)
            print(f'{script:<62} {status:<6} {dt:6.1f}s', flush=True)
            for pr in probs[:5]:
                print(f'    {pr}')
            if len(probs) > 5:
                print(f'    ... {len(probs) - 5} more')
    finally:
        if a.keep:
            print('temporary copy kept at', tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f'{failures} failure(s)')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
