#!/usr/bin/env python3
"""
scripts/verify_model_parity.py - True model parity evaluator (Milestone 4 / F18), rewritten 2026-11-18.

What changed vs. the previous revision (honesty fixes):
  * The old `run_reference_inference` was a stub (zero logits, empty token list) and the test cases compared Tenzo against
    token lists hard-coded from unknown provenance. Both are gone. References are now REAL dumps read from
    results/reference/<source>/<prompt_id>.{ids.txt,logits.f32,greedy.txt}, produced by
      - tools/reference/hf_ref_dump.py      (source "hf":    HF transformers BitNetForCausalLM, ground truth)
      - tools/reference/bitnet_ref_dump.cpp (source "llama": bitnet.cpp / libllama i2_s GGUF)
  * Tenzo is driven with the REFERENCE token ids (tenzo-inference --ids-file), so tokenizer differences cannot hide or
    create logit differences; tokenization is compared separately (--check-tokenization).
  * A missing reference file is an error (never silently replaced by zeros).

Metrics per prompt and source: cosine, max_abs, mae, rmse, top-1, top-5 overlap, KL(ref||tenzo) over softmax, and the
length of the greedy-token prefix that matches before the first divergence.

Gate modes:
  * v1 (default, unchanged): cosine > 0.999 and max_abs <= 0.25 on first-step logits, top-1 equal on all prompts;
    greedy agreement is reported (and gated only with --gate-greedy).
  * v2 (reference-calibrated): calibrated to the empirical fp32-vs-bf16 noise floor between the two ground-truth
    references (tools/reference/golden_bitnet.py fp32 vs HF transformers bf16 across the 7 prompts:
    max_abs = 0.532..1.634 [0/7 pass <= 0.25], cosine = 0.9932..0.99992 [1 - cos = 7.56e-5..6.76e-3]):
      1. Per prompt p: max_abs(Tenzo, golden) <= 1.5 * max_abs(HF bf16, golden)
         and (1 - cos(Tenzo, golden)) <= 1.5 * (1 - cos(HF bf16, golden)).
      2. top-1(Tenzo) == top-1(golden) on >= 6 of 7 prompts.
      3. Paired PPL ratio Tenzo / golden on 16 WikiText-2 EN windows: 95% bootstrap CI inside [0.995, 1.010].

Exit code is non-zero if any gated check fails.
"""
import sys
import os
import json
import math
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCAB = 128256


def check_for_nan_inf(arr: np.ndarray, name: str) -> None:
    """Explicitly reject NaN or Inf values in logits."""
    if not np.isfinite(arr).all():
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        raise ValueError(f"Numerical error: {name} contains {nan_count} NaNs and {inf_count} Infs!")


def compute_logits_metrics(engine_logits: np.ndarray, ref_logits: np.ndarray) -> Dict[str, Any]:
    """
    Computes rigorous numerical metrics between engine logits and reference logits:
    - Cosine similarity
    - Max absolute difference
    - Mean absolute error (MAE)
    - Root mean square error (RMSE)
    - Top-1, Top-5, Top-10 agreement
    """
    check_for_nan_inf(engine_logits, "Engine logits")
    check_for_nan_inf(ref_logits, "Reference logits")

    if engine_logits.shape != ref_logits.shape:
        raise ValueError(f"Shape mismatch: engine logits {engine_logits.shape} vs reference {ref_logits.shape}")

    u = engine_logits.astype(np.float64)
    v = ref_logits.astype(np.float64)

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u < 1e-12 or norm_v < 1e-12:
        raise ValueError("Degenerate near-zero logits vector detected!")

    cosine_sim = float(np.dot(u, v) / (norm_u * norm_v))
    max_abs_diff = float(np.max(np.abs(u - v)))
    mae = float(np.mean(np.abs(u - v)))
    rmse = float(np.sqrt(np.mean((u - v) ** 2)))

    # Top-K agreement
    engine_top1 = int(np.argmax(engine_logits))
    ref_top1 = int(np.argmax(ref_logits))

    engine_top5 = set(np.argsort(engine_logits)[-5:])
    ref_top5 = set(np.argsort(ref_logits)[-5:])
    top5_overlap = len(engine_top5.intersection(ref_top5)) / 5.0

    engine_top10 = set(np.argsort(engine_logits)[-10:])
    ref_top10 = set(np.argsort(ref_logits)[-10:])
    top10_overlap = len(engine_top10.intersection(ref_top10)) / 10.0

    return {
        "cosine_similarity": cosine_sim,
        "max_abs_diff": max_abs_diff,
        "mae": mae,
        "rmse": rmse,
        "engine_top1": engine_top1,
        "ref_top1": ref_top1,
        "top1_match": (engine_top1 == ref_top1),
        "top5_agreement": top5_overlap,
        "top10_agreement": top10_overlap,
    }


def softmax64(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def kl_divergence(ref_logits: np.ndarray, eng_logits: np.ndarray) -> float:
    """KL(ref || engine) between the two softmax distributions (nats)."""
    p = softmax64(ref_logits)
    q = softmax64(eng_logits)
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(np.maximum(q[mask], 1e-300)))))


def greedy_prefix_match(a: List[int], b: List[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def in_docker() -> bool:
    return Path("/app/cmake-build-debug/tenzo-inference").is_file() and Path("/.dockerenv").exists()


def _container_path(p: Path) -> str:
    """Translate a host path under the repo to the /app mount used by the dev container."""
    p = Path(p).resolve()
    if in_docker():
        return str(p)
    try:
        return "/app/" + str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        raise ValueError(f"{p} is outside the repository; the container only mounts the repo at /app")


def run_tenzo(args: List[str], timeout_sec: int = 900) -> Tuple[int, str]:
    """Run tenzo-inference (inside the dev container unless already in it). `args` may contain absolute host paths
    already translated with _container_path()."""
    omp = ["-e", "OMP_PLACES=cores", "-e", "OMP_PROC_BIND=spread"]
    if in_docker():
        cmd = ["/app/cmake-build-debug/tenzo-inference"] + args
    else:
        cmd = ["docker", "compose", "-f", str(PROJECT_ROOT / "docker-compose.yml"), "run", "--rm"] + omp + \
              ["dev", "/app/cmake-build-debug/tenzo-inference"] + args
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, cwd=str(PROJECT_ROOT))
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def load_prompts(path: Path) -> List[Dict[str, str]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_reference(ref_dir: Path, source: str, pid: str) -> Dict[str, Any]:
    """Read a real reference dump. Raises FileNotFoundError if any piece is missing (no stubs)."""
    d = Path(ref_dir) / source
    ids_p, lg_p, gr_p = d / f"{pid}.ids.txt", d / f"{pid}.logits.f32", d / f"{pid}.greedy.txt"
    for f in (ids_p, lg_p, gr_p):
        if not f.is_file():
            raise FileNotFoundError(f"reference dump missing: {f} (regenerate with tools/reference/*)")
    logits = np.fromfile(lg_p, dtype=np.float32)
    if logits.size != VOCAB:
        raise ValueError(f"{lg_p}: expected {VOCAB} logits, got {logits.size}")
    return {"ids": [int(x) for x in ids_p.read_text().split()], "logits": logits,
            "greedy": [int(x) for x in gr_p.read_text().split()]}


def run_tenzo_dump(model_dir: str, ids: List[int], kv_quant: str, steps: int, workdir: Path) -> Dict[str, Any]:
    """Run Tenzo on explicit token ids; returns first-step logits and greedy ids."""
    workdir.mkdir(parents=True, exist_ok=True)
    ids_file = workdir / "ids.txt"
    ids_file.write_text(" ".join(map(str, ids)) + "\n")
    prefix = workdir / "tenzo"
    for suffix in (".logits.f32", ".greedy.txt"):
        Path(str(prefix) + suffix).unlink(missing_ok=True)
    m = model_dir if (in_docker() or model_dir.startswith("/app")) else "/app/" + model_dir
    rc, out = run_tenzo(["-m", m, "--kv-quant", kv_quant, "--ids-file", _container_path(ids_file),
                         "--dump-prefix", _container_path(prefix), "--greedy-steps", str(steps),
                         "-c", str(max(1024, len(ids) + steps + 8))])
    res: Dict[str, Any] = {"rc": rc, "output_tail": out[-600:]}
    lp, gp = Path(str(prefix) + ".logits.f32"), Path(str(prefix) + ".greedy.txt")
    if rc == 0 and lp.is_file() and gp.is_file():
        res["logits"] = np.fromfile(lp, dtype=np.float32)
        res["greedy"] = [int(x) for x in gp.read_text().split()]
    return res


def compare_one(tenzo: Dict[str, Any], ref: Dict[str, Any], cos_min: float, maxdiff: float) -> Dict[str, Any]:
    m = compute_logits_metrics(tenzo["logits"], ref["logits"])
    m["kl_ref_to_tenzo"] = kl_divergence(ref["logits"], tenzo["logits"])
    m["greedy_prefix_match"] = greedy_prefix_match(tenzo["greedy"], ref["greedy"])
    m["greedy_len"] = len(ref["greedy"])
    m["gate_cosine"] = m["cosine_similarity"] > cos_min
    m["gate_max_abs"] = m["max_abs_diff"] <= maxdiff
    m["gate_top1"] = bool(m["top1_match"])
    m["gate_pass"] = bool(m["gate_cosine"] and m["gate_max_abs"] and m["gate_top1"])
    return m


def evaluate_v2_prompt(tenzo_vs_golden: Dict[str, Any], hf_vs_golden: Dict[str, Any]) -> Dict[str, Any]:
    """Reference-calibrated v2 prompt checks against golden fp32, scaled from HF bf16 vs golden fp32."""
    dcos_tenzo = 1.0 - float(tenzo_vs_golden["cosine_similarity"])
    dcos_ref = 1.0 - float(hf_vs_golden["cosine_similarity"])
    mabs_tenzo = float(tenzo_vs_golden["max_abs_diff"])
    mabs_ref = float(hf_vs_golden["max_abs_diff"])
    gate_max_abs_v2 = bool(mabs_tenzo <= 1.5 * mabs_ref)
    gate_cosine_v2 = bool(dcos_tenzo <= 1.5 * dcos_ref)
    return {
        "one_minus_cos_tenzo": dcos_tenzo,
        "one_minus_cos_ref": dcos_ref,
        "one_minus_cos_limit": 1.5 * dcos_ref,
        "ref_max_abs": mabs_ref,
        "max_abs_limit": 1.5 * mabs_ref,
        "gate_max_abs_v2": gate_max_abs_v2,
        "gate_cosine_v2": gate_cosine_v2,
        "gate_prompt_v2": bool(gate_max_abs_v2 and gate_cosine_v2),
    }


def load_score_windows(path: Path) -> Dict[int, Tuple[int, float]]:
    """Parse `w <id> <n_tokens> <sum_nll>` lines from a score file."""
    if not path.is_file():
        raise FileNotFoundError(f"score file missing: {path}")
    out: Dict[int, Tuple[int, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0] == "w":
            out[int(parts[1])] = (int(parts[2]), float(parts[3]))
    if not out:
        raise ValueError(f"no 'w' lines in score file: {path}")
    return out


def paired_ppl_bootstrap_ci(
    cand_scores: Dict[int, Tuple[int, float]],
    ref_scores: Dict[int, Tuple[int, float]],
    n_draws: int = 10000,
    seed: int = 0,
    lo_bound: float = 0.995,
    hi_bound: float = 1.010,
) -> Dict[str, Any]:
    """Paired bootstrap 95% CI of PPL(cand) / PPL(ref) across identical windows."""
    ids = sorted(set(cand_scores.keys()) & set(ref_scores.keys()))
    if len(ids) != len(ref_scores) or len(ids) == 0:
        raise ValueError(f"window mismatch: cand={len(cand_scores)} ref={len(ref_scores)} common={len(ids)}")
    cn = np.array([cand_scores[i][0] for i in ids], dtype=np.float64)
    cl = np.array([cand_scores[i][1] for i in ids], dtype=np.float64)
    rn = np.array([ref_scores[i][0] for i in ids], dtype=np.float64)
    rl = np.array([ref_scores[i][1] for i in ids], dtype=np.float64)
    if not np.array_equal(cn, rn):
        raise ValueError("token count mismatch across windows")
    cand_ppl = float(math.exp(cl.sum() / cn.sum()))
    ref_ppl = float(math.exp(rl.sum() / rn.sum()))
    ratio = cand_ppl / ref_ppl
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, len(ids), size=(n_draws, len(ids)))
    ratios = np.exp(cl[idx].sum(axis=1) / cn[idx].sum(axis=1)) / np.exp(rl[idx].sum(axis=1) / rn[idx].sum(axis=1))
    ci_lo, ci_hi = float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))
    return {
        "n_windows": len(ids),
        "cand_ppl": cand_ppl,
        "ref_ppl": ref_ppl,
        "ratio": ratio,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "gate_ppl_ci_v2": bool(ci_lo >= lo_bound and ci_hi <= hi_bound),
    }


def run_tenzo_score(model_dir: str, kv_quant: str, windows_file: Path, out_score: Path) -> None:
    """Run tenzo-inference --score on 16 EN windows inside the dev container."""
    m = model_dir if (in_docker() or model_dir.startswith("/app")) else "/app/" + model_dir
    out_score.parent.mkdir(parents=True, exist_ok=True)
    rc, out = run_tenzo([
        "-m", m, "--kv-quant", kv_quant, "-c", "512",
        "--score", _container_path(windows_file),
        "--score-out", _container_path(out_score),
    ], timeout_sec=3600)
    if rc != 0:
        raise RuntimeError(f"tenzo-inference --score failed (rc={rc}):\n{out[-600:]}")


def check_tokenization(model_dir: str, prompts: List[Dict[str, str]], ref_dir: Path, sources: List[str]) -> List[Dict[str, Any]]:
    """Compare Tenzo's tokenizer (tenzo-inference --tokenize-only) with every reference tokenizer."""
    m = model_dir if (in_docker() or model_dir.startswith("/app")) else "/app/" + model_dir
    rows = []
    for p in prompts:
        rc, out = run_tenzo(["-m", m, "--tokenize-only", "-p", p["text"]], timeout_sec=120)
        line = [l for l in out.splitlines() if l.startswith("TOKENS:")]
        t_ids = [int(x) for x in line[0].split()[1:]] if (rc == 0 and line) else None
        row = {"prompt_id": p["id"], "tenzo_ids": t_ids}
        for src in sources:
            f = Path(ref_dir) / src / f"{p['id']}.ids_native.txt"      # the engine's OWN tokenisation of the text
            if not f.is_file():
                f = Path(ref_dir) / src / f"{p['id']}.ids.txt"
            if f.is_file() and t_ids is not None:
                r_ids = [int(x) for x in f.read_text().split()]
                n = greedy_prefix_match(t_ids, r_ids)
                row[src] = {"ref_len": len(r_ids), "tenzo_len": len(t_ids), "identical": t_ids == r_ids,
                            "first_divergence": None if t_ids == r_ids else n}
        rows.append(row)
    return rows


def negative_control_truncated(model_dir: str) -> Dict[str, Any]:
    """Engine must exit non-zero on truncated weights (F16). Uses a tiny truncated copy (1 MB) under the repo tree."""
    src = PROJECT_ROOT / model_dir
    tmp = Path(tempfile.mkdtemp(prefix="parity_trunc_", dir=str(PROJECT_ROOT / "results")))
    try:
        for f in ("config.json", "model.mlir", "tokenizer.vocab"):
            shutil.copy(src / f, tmp / f)
        if (src / "tokenizer.json").is_file():
            shutil.copy(src / "tokenizer.json", tmp / "tokenizer.json")
        with open(src / "weights.bin", "rb") as fi, open(tmp / "weights.bin", "wb") as fo:
            fo.write(fi.read(1 << 20))
        rc, out = run_tenzo(["-m", _container_path(tmp), "-p", "Test prompt", "-n", "1", "-t", "0"], timeout_sec=120)
        return {"name": "truncated_weights", "rc": rc, "pass": rc != 0}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_self_test() -> bool:
    """
    Executes unit self-test of the evaluator's mathematical comparator and verification logic.
    """
    print("\n=======================================================")
    print("▶ Running Parity Evaluator Self-Test (R5 Unit Validation)")
    print("=======================================================")

    vocab_size = 128256
    rng = np.random.RandomState(42)

    # 1. Test NaN / Inf rejection
    clean_vec = rng.randn(vocab_size).astype(np.float32)
    check_for_nan_inf(clean_vec, "Clean Vector")

    nan_vec = clean_vec.copy()
    nan_vec[100] = np.nan
    try:
        check_for_nan_inf(nan_vec, "NaN Vector")
        print("❌ Self-test FAILED: check_for_nan_inf did not catch NaN!", file=sys.stderr)
        return False
    except ValueError as e:
        assert "NaNs" in str(e)
    print("  ✅ Self-test: NaN detection and rejection verified.")

    inf_vec = clean_vec.copy()
    inf_vec[200] = np.inf
    try:
        check_for_nan_inf(inf_vec, "Inf Vector")
        print("❌ Self-test FAILED: check_for_nan_inf did not catch Inf!", file=sys.stderr)
        return False
    except ValueError as e:
        assert "Infs" in str(e)
    print("  ✅ Self-test: Inf detection and rejection verified.")

    # 2. Test identical vector metrics
    metrics_id = compute_logits_metrics(clean_vec, clean_vec)
    assert abs(metrics_id["cosine_similarity"] - 1.0) < 1e-6
    assert metrics_id["max_abs_diff"] == 0.0
    assert metrics_id["top1_match"] is True
    assert metrics_id["top5_agreement"] == 1.0
    print("  ✅ Self-test: Identical vector metrics verified (cosine=1.0, max_diff=0.0, top1=True).")

    # 3. Test small bounded noise (cosine > 0.999, max_abs_diff <= 0.25)
    noise = (rng.randn(vocab_size) * 0.01).astype(np.float32)
    perturbed = clean_vec + noise
    metrics_perturbed = compute_logits_metrics(clean_vec, perturbed)
    assert metrics_perturbed["cosine_similarity"] > 0.999
    assert metrics_perturbed["max_abs_diff"] <= 0.25
    print(f"  ✅ Self-test: Bounded noise parity verified (cosine={metrics_perturbed['cosine_similarity']:.6f}, max_diff={metrics_perturbed['max_abs_diff']:.4f}).")

    # 4. Test large divergence detection
    divergent = rng.randn(vocab_size).astype(np.float32)
    metrics_div = compute_logits_metrics(clean_vec, divergent)
    assert metrics_div["cosine_similarity"] < 0.999
    print(f"  ✅ Self-test: Divergence correctly detected (cosine={metrics_div['cosine_similarity']:.4f} < 0.999).")

    # 5. Test degenerate zero vector rejection
    zero_vec = np.zeros(vocab_size, dtype=np.float32)
    try:
        compute_logits_metrics(clean_vec, zero_vec)
        print("❌ Self-test FAILED: Zero vector not rejected!", file=sys.stderr)
        return False
    except ValueError as e:
        assert "Degenerate" in str(e)
    print("  ✅ Self-test: Degenerate near-zero vector rejection verified.")

    # 6. KL divergence: zero for identical, positive for perturbed, symmetric-safe
    assert kl_divergence(clean_vec, clean_vec) < 1e-12
    assert kl_divergence(clean_vec, divergent) > 0.1
    print("  ✅ Self-test: KL divergence verified.")

    # 7. Greedy prefix matcher
    assert greedy_prefix_match([1, 2, 3], [1, 2, 9]) == 2 and greedy_prefix_match([1], [1]) == 1 and greedy_prefix_match([2], [1]) == 0
    print("  ✅ Self-test: greedy prefix matcher verified.")

    # 8. A missing reference must raise, never return zeros
    try:
        load_reference(Path("/nonexistent"), "hf", "x")
        print("❌ Self-test FAILED: missing reference did not raise!", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("  ✅ Self-test: missing reference raises (no stub fallback).")

    # 9. Gate v2 helpers: reference-calibrated prompt limits and paired PPL bootstrap CI
    ref_v2 = compute_logits_metrics(clean_vec, clean_vec + (rng.randn(vocab_size) * 0.05).astype(np.float32))
    t_pass = compute_logits_metrics(clean_vec, clean_vec + (rng.randn(vocab_size) * 0.04).astype(np.float32))
    t_fail = compute_logits_metrics(clean_vec, clean_vec + (rng.randn(vocab_size) * 0.50).astype(np.float32))
    assert evaluate_v2_prompt(t_pass, ref_v2)["gate_prompt_v2"] is True
    assert evaluate_v2_prompt(t_fail, ref_v2)["gate_prompt_v2"] is False
    w_ref = {i: (511, 1500.0 + 10.0 * i) for i in range(16)}
    w_ok = {i: (511, 1501.0 + 10.0 * i) for i in range(16)}
    w_bad = {i: (511, 1530.0 + 10.0 * i) for i in range(16)}
    assert paired_ppl_bootstrap_ci(w_ok, w_ref, n_draws=200)["gate_ppl_ci_v2"] is True
    assert paired_ppl_bootstrap_ci(w_bad, w_ref, n_draws=200)["gate_ppl_ci_v2"] is False
    print("  ✅ Self-test: gate v2 reference-calibrated limits and PPL bootstrap CI verified.")

    print("=======================================================")
    print("🎉 ALL PARITY EVALUATOR SELF-TESTS PASSED!")
    print("=======================================================")
    return True


def main():
    ap = argparse.ArgumentParser(description="Tenzo model parity evaluator (real references, 2026-11-18)")
    ap.add_argument("-m", "--model-dir", default="models/bitnet-1.58b")
    ap.add_argument("--ref-dir", default=str(PROJECT_ROOT / "results" / "reference"))
    ap.add_argument("--prompts", default=str(PROJECT_ROOT / "tools" / "reference" / "prompts.json"))
    ap.add_argument("--sources", default="hf,llama", help="comma list of reference sources under --ref-dir")
    ap.add_argument("--gate-source", default="hf", help="reference source that the exit code is gated on")
    ap.add_argument("--kv-quant", default="int8_fused")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--tolerance", type=float, default=0.999, help="Minimum cosine similarity")
    ap.add_argument("--max-diff", type=float, default=0.25, help="Maximum absolute logit difference")
    ap.add_argument("--gate-greedy", action="store_true", help="also require the full greedy sequence to match")
    ap.add_argument("--check-tokenization", action="store_true")
    ap.add_argument("--only-tokenization", action="store_true", help="compare tokenisation only (implies --check-tokenization)")
    ap.add_argument("--negative-control", action="store_true", help="run the truncated-weights negative control")
    ap.add_argument("--gate", choices=["v1", "v2"], default="v1",
                    help="v1: legacy strict cos>0.999 & max_abs<=0.25; v2: reference-calibrated vs golden/HF")
    ap.add_argument("--score-file", default=None, help="16-window EN score file for --gate v2 (runs --score if omitted)")
    ap.add_argument("--golden-score-file", default=str(PROJECT_ROOT / "results" / "reference" / "golden" / "nll_golden_float32.txt"))
    ap.add_argument("--windows-file", default=str(PROJECT_ROOT / "results" / "reference" / "windows.ids"))
    ap.add_argument("--out", default=None, help="write JSON results here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if run_self_test() else 1)

    ref_dir = Path(args.ref_dir)
    sources = [s for s in args.sources.split(",") if s]
    if args.gate == "v2":
        for req in ("golden", "hf"):
            if req not in sources:
                sources.append(req)
    prompts = load_prompts(Path(args.prompts))
    results: Dict[str, Any] = {"kv_quant": args.kv_quant, "gate_mode": args.gate, "prompts": {},
                               "tolerance_cos": args.tolerance, "tolerance_max_abs": args.max_diff}
    ok = True

    if args.check_tokenization or args.only_tokenization:
        results["tokenization"] = check_tokenization(args.model_dir, prompts, ref_dir, sources)
        for row in results["tokenization"]:
            for src in sources:
                if src in row:
                    print(f"[tokenization] {row['prompt_id']:<14} vs {src:<5}: identical={row[src]['identical']} "
                          f"first_divergence={row[src]['first_divergence']} (tenzo {row[src]['tenzo_len']} / ref {row[src]['ref_len']} tokens)")

    if args.only_tokenization:
        if args.out:
            Path(args.out).write_text(json.dumps(results, indent=1))
        sys.exit(0)

    work = Path(tempfile.mkdtemp(prefix="parity_", dir=str(PROJECT_ROOT / "results")))
    try:
        for p in prompts:
            pid = p["id"]
            try:
                refs = {s: load_reference(ref_dir, s, pid) for s in sources if (ref_dir / s / f"{pid}.logits.f32").is_file()}
            except (FileNotFoundError, ValueError) as e:
                print(f"❌ {pid}: {e}", file=sys.stderr)
                ok = False
                continue
            if args.gate_source not in refs:
                print(f"❌ {pid}: gate reference '{args.gate_source}' missing", file=sys.stderr)
                ok = False
                continue
            ids = refs[args.gate_source]["ids"]
            t = run_tenzo_dump(args.model_dir, ids, args.kv_quant, args.steps, work / pid)
            if "logits" not in t:
                print(f"❌ {pid}: tenzo run failed rc={t['rc']}\n{t['output_tail']}", file=sys.stderr)
                ok = False
                continue
            entry = {"n_tokens": len(ids)}
            for s, r in refs.items():
                m = compare_one(t, r, args.tolerance, args.max_diff)
                entry[s] = m
                if args.gate == "v1":
                    tag = "PASS" if m["gate_pass"] else "FAIL"
                    print(f"[{tag}] {pid:<14} vs {s:<5} cos={m['cosine_similarity']:.6f} max_abs={m['max_abs_diff']:.4f} "
                          f"top1={m['top1_match']} top5={m['top5_agreement']:.1f} KL={m['kl_ref_to_tenzo']:.4f} "
                          f"greedy={m['greedy_prefix_match']}/{m['greedy_len']}")
                    if s == args.gate_source:
                        if not m["gate_pass"]:
                            ok = False
                        if args.gate_greedy and m["greedy_prefix_match"] != m["greedy_len"]:
                            ok = False
            if args.gate == "v2":
                ref_cmp = compute_logits_metrics(refs["hf"]["logits"], refs["golden"]["logits"])
                v2 = evaluate_v2_prompt(entry["golden"], ref_cmp)
                entry["v2_golden_vs_hf_ref"] = ref_cmp
                entry["v2"] = v2
                g_m = entry["golden"]
                tag = "PASS" if v2["gate_prompt_v2"] else "FAIL"
                print(f"[{tag}] {pid:<14} vs golden (v2): max_abs={g_m['max_abs_diff']:.4f}<=1.5*{v2['ref_max_abs']:.4f}({v2['max_abs_limit']:.4f}) "
                      f"1-cos={v2['one_minus_cos_tenzo']:.3e}<=1.5*{v2['one_minus_cos_ref']:.3e}({v2['one_minus_cos_limit']:.3e}) "
                      f"top1={g_m['top1_match']} KL={g_m['kl_ref_to_tenzo']:.4f} greedy={g_m['greedy_prefix_match']}/{g_m['greedy_len']}")
                if not v2["gate_prompt_v2"]:
                    ok = False
                if args.gate_greedy and g_m["greedy_prefix_match"] != g_m["greedy_len"]:
                    ok = False
            results["prompts"][pid] = entry
        if args.gate == "v2":
            top1_hits = sum(1 for e in results["prompts"].values() if e.get("golden", {}).get("top1_match"))
            n_prompts = len(results["prompts"])
            min_top1 = 6 if n_prompts == 7 else max(1, int(math.ceil(6.0 * n_prompts / 7.0)))
            gate_top1_v2 = (top1_hits >= min_top1) and (n_prompts == len(prompts))
            results["v2_top1"] = {"hits": top1_hits, "total": n_prompts, "required": min_top1, "pass": gate_top1_v2}
            print(f"[{'PASS' if gate_top1_v2 else 'FAIL'}] v2 top-1 aggregate: {top1_hits}/{n_prompts} (required >= {min_top1})")
            ok = ok and gate_top1_v2

            if args.score_file:
                score_path = Path(args.score_file)
            else:
                score_path = work / f"score_{args.kv_quant}.txt"
                run_tenzo_score(args.model_dir, args.kv_quant, Path(args.windows_file), score_path)
            ppl_res = paired_ppl_bootstrap_ci(load_score_windows(score_path), load_score_windows(Path(args.golden_score_file)))
            ppl_res["score_file"] = str(score_path)
            results["v2_ppl_ci"] = ppl_res
            print(f"[{'PASS' if ppl_res['gate_ppl_ci_v2'] else 'FAIL'}] v2 EN16 PPL ratio: "
                  f"{ppl_res['cand_ppl']:.3f}/{ppl_res['ref_ppl']:.3f} = {ppl_res['ratio']:.4f} "
                  f"95% CI [{ppl_res['ci_lo']:.4f}, {ppl_res['ci_hi']:.4f}] in [0.995, 1.010]")
            ok = ok and ppl_res["gate_ppl_ci_v2"]
    finally:
        shutil.rmtree(work, ignore_errors=True)

    if args.negative_control:
        nc = negative_control_truncated(args.model_dir)
        results["negative_control"] = nc
        print(f"[{'PASS' if nc['pass'] else 'FAIL'}] negative control {nc['name']}: rc={nc['rc']}")
        ok = ok and nc["pass"]

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o)))
    print("PARITY RESULT:", "ALL GATES PASSED" if ok else "GATE FAILURES (see numbers above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
