"""Reproducible Inference Engine Latency and Throughput Benchmarking Script.

Executes controlled, timed evaluations of single-record and batch inference modes,
documenting warm-up protocols, execution environments, measurement boundaries,
and percentile distributions.
"""

import json
import platform
import subprocess
import time

import numpy as np
import psutil

from predictlab.config import get_settings
from predictlab.inference.engine import InferenceEngine


def get_cpu_model() -> str:
    """Retrieve the exact CPU model name across Windows/Linux."""
    if platform.system() == "Windows":
        try:
            cmd = ["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"]
            out = subprocess.check_output(cmd, text=True).strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


def get_environment_info() -> dict:
    """Capture execution environment metadata."""
    mem = psutil.virtual_memory()
    import pydantic
    import sklearn

    return {
        "operating_system": platform.platform(),
        "os_release": platform.version(),
        "cpu_model": get_cpu_model(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(mem.total / (1024**3), 2),
        "available_ram_gb": round(mem.available / (1024**3), 2),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "pydantic_version": pydantic.__version__,
        "numpy_version": np.__version__,
    }


def compute_benchmark_stats(latencies_ms: list[float], batch_size: int = 1) -> dict:
    """Compute distribution metrics from a list of measured latency times in milliseconds."""
    arr = np.array(latencies_ms)
    med = float(np.median(arr))
    mean = float(np.mean(arr))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    # Normalized per-record stats
    per_rec_med = med / batch_size
    per_rec_mean = mean / batch_size
    per_rec_p95 = p95 / batch_size
    per_rec_p99 = p99 / batch_size

    # Throughput (records / sec)
    throughput = (batch_size / (med / 1000.0)) if med > 0 else 0.0

    return {
        "batch_size": batch_size,
        "repetitions": len(latencies_ms),
        "total_batch_latency_ms": {
            "median": round(med, 2),
            "mean": round(mean, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
        },
        "normalized_per_record_latency_ms": {
            "median": round(per_rec_med, 4),
            "mean": round(per_rec_mean, 4),
            "p95": round(per_rec_p95, 4),
            "p99": round(per_rec_p99, 4),
        },
        "throughput_records_per_sec": round(throughput, 1),
    }


def main() -> None:
    settings = get_settings()
    exp_dir = settings.reports_dir / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=======================================================")
    print(" PREDICTLAB — INFERENCE ENGINE BENCHMARK SUITE")
    print("=======================================================")

    env_info = get_environment_info()
    print("\nExecution Environment:")
    for k, v in env_info.items():
        print(f"  {k:25s}: {v}")

    # Initialize InferenceEngine
    engine = InferenceEngine()

    sample_record = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": 412.5,
    }

    # 1. Warm-up Phase
    print("\n[Phase 1] Executing warm-up iterations (5 unmeasured cycles)...")
    for _ in range(5):
        engine.predict_single(sample_record, include_explanation=False)
        engine.predict_single(sample_record, include_explanation=True)
        engine.predict_batch([sample_record] * 20)

    results = {
        "environment": env_info,
        "measurement_methodology": {
            "timer": "time.perf_counter() (monotonic high-resolution timer)",
            "warmup_cycles": 5,
            "scope_single_record": (
                "Includes full input validation (Pydantic CustomerInputRecord), DataFrame wrapping, "
                "scikit-learn ColumnTransformer preprocessing, 5-fold calibrated ensemble prediction, "
                "risk tier assignment, and SinglePredictionOutput serialization."
            ),
            "scope_single_record_shap": (
                "Includes single-record pipeline plus 5-estimator TreeSHAP margin attribution traversal, "
                "feature unwrapping, and CustomerLocalExplanation construction."
            ),
            "scope_batch": (
                "Includes batch validation, atomic/per-record checking, vectorized DataFrame preprocessing, "
                "calibrated ensemble prediction across 5 estimators, risk tier assignment, and "
                "BatchPredictionOutput packaging."
            ),
            "note_on_latencies": (
                "Normalized per-record batch latency reflects amortized throughput in vectorized batch jobs "
                "and is NOT equivalent to single-record interactive request latency, which is dominated by "
                "per-request object instantiation and validation overhead."
            ),
        },
        "benchmarks": {},
    }

    # 2. Benchmark Single Record (No SHAP)
    print("\n[Phase 2] Benchmarking Single Record (no SHAP, 50 repetitions)...")
    single_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        engine.predict_single(sample_record, include_explanation=False)
        t1 = time.perf_counter()
        single_times.append((t1 - t0) * 1000.0)
    results["benchmarks"]["single_record_no_shap"] = compute_benchmark_stats(single_times, batch_size=1)

    # 3. Benchmark Single Record (With TreeSHAP)
    print("\n[Phase 3] Benchmarking Single Record with 5-Estimator TreeSHAP (20 repetitions)...")
    single_shap_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        engine.predict_single(sample_record, include_explanation=True)
        t1 = time.perf_counter()
        single_shap_times.append((t1 - t0) * 1000.0)
    results["benchmarks"]["single_record_with_shap"] = compute_benchmark_stats(single_shap_times, batch_size=1)

    # 4. Benchmark Batch 100
    print("\n[Phase 4] Benchmarking Batch 100 Records (30 repetitions)...")
    batch_100_payload = [sample_record] * 100
    batch_100_times = []
    for _ in range(30):
        t0 = time.perf_counter()
        engine.predict_batch(batch_100_payload)
        t1 = time.perf_counter()
        batch_100_times.append((t1 - t0) * 1000.0)
    results["benchmarks"]["batch_100_records"] = compute_benchmark_stats(batch_100_times, batch_size=100)

    # 5. Benchmark Batch 1,000
    print("\n[Phase 5] Benchmarking Batch 1,000 Records (30 repetitions)...")
    batch_1000_payload = [sample_record] * 1000
    batch_1000_times = []
    for _ in range(30):
        t0 = time.perf_counter()
        engine.predict_batch(batch_1000_payload)
        t1 = time.perf_counter()
        batch_1000_times.append((t1 - t0) * 1000.0)
    results["benchmarks"]["batch_1000_records"] = compute_benchmark_stats(batch_1000_times, batch_size=1000)

    # Print Formatted Results
    print("\n" + "=" * 55)
    print(" BENCHMARK SUMMARY TABLE")
    print("=" * 55)
    for name, data in results["benchmarks"].items():
        tb = data["total_batch_latency_ms"]
        pr = data["normalized_per_record_latency_ms"]
        print(f"\nWorkload: {name} (N = {data['batch_size']}, {data['repetitions']} repeats)")
        print(f"  Total Latency   -> Median: {tb['median']} ms | Mean: {tb['mean']} ms | P95: {tb['p95']} ms | P99: {tb['p99']} ms")
        if data["batch_size"] > 1:
            print(f"  Per-Record      -> Median: {pr['median']} ms | Mean: {pr['mean']} ms | P95: {pr['p95']} ms")
        print(f"  Throughput      -> {data['throughput_records_per_sec']} records/sec")

    # Serialize JSON
    out_path = exp_dir / "inference_benchmarks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
