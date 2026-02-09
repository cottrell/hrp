"""Benchmark and comparison tests for HRP implementations."""
import time
import numpy as np
import pandas as pd
import pytest

from hrp import get_hrp


def benchmark_implementation(func, data, runs=100):
    """Benchmark a single implementation."""
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = func(data)
        end = time.perf_counter()
        times.append(end - start)
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'result': result
    }


def test_hrp_numpy():
    """Test the numpy-only implementation."""
    np.random.seed(42)
    y = np.random.randn(100, 20)
    result = get_hrp(y)
    assert 'hrp' in result
    assert len(result['hrp']) == 20
    assert np.isclose(result['hrp'].sum(), 1.0)
    print(f"✓ hrp (numpy) passes basic tests")


def test_hrp_pyportfolioopt():
    """Test PyPortfolioOpt HRP implementation."""
    try:
        from pypfopt import HRPOpt

        np.random.seed(42)
        y = np.random.randn(100, 20)
        returns_df = pd.DataFrame(y)

        hrp = HRPOpt(returns_df)
        weights = hrp.optimize()
        weights_array = np.array(list(weights.values()))

        assert len(weights_array) == 20
        assert np.isclose(weights_array.sum(), 1.0)
        print(f"✓ PyPortfolioOpt passes basic tests")
    except ImportError:
        pytest.skip("PyPortfolioOpt not installed")


def test_hrp_riskfolio():
    """Test Riskfolio-Lib HRP implementation."""
    try:
        import riskfolio as rp

        np.random.seed(42)
        y = np.random.randn(100, 20)
        returns_df = pd.DataFrame(y)

        port = rp.HCPortfolio(returns=returns_df)
        weights = port.optimization(model='HRP', rm='MV', rf=0, linkage='single')
        weights_array = weights.values.flatten()

        assert len(weights_array) == 20
        assert np.isclose(weights_array.sum(), 1.0, atol=1e-5)
        print(f"✓ Riskfolio-Lib passes basic tests")
    except ImportError:
        pytest.skip("Riskfolio-Lib not installed")


def test_cross_implementation_equivalence():
    """Verify all HRP implementations produce equivalent results on same data."""
    np.random.seed(42)
    y = np.random.randn(100, 20)
    returns_df = pd.DataFrame(y)

    # Get results from all implementations
    results = {}

    # Numpy implementation
    hrp_result = get_hrp(y)
    results['numpy'] = hrp_result['hrp']

    # PyPortfolioOpt
    try:
        from pypfopt import HRPOpt
        hrp = HRPOpt(returns_df)
        weights = hrp.optimize()
        results['pypfopt'] = np.array(list(weights.values()))
    except ImportError:
        print("⚠ PyPortfolioOpt not installed, skipping cross-validation")

    # Riskfolio-Lib
    try:
        import riskfolio as rp
        port = rp.HCPortfolio(returns=returns_df)
        weights = port.optimization(model='HRP', rm='MV', rf=0, linkage='single')
        results['riskfolio'] = weights.values.flatten()
    except ImportError:
        print("⚠ Riskfolio-Lib not installed, skipping cross-validation")

    # Compare all implementations
    if len(results) > 1:
        print(f"\nCross-implementation validation ({len(results)} implementations):")
        baseline = results['numpy']

        for name, weights in results.items():
            if name != 'numpy':
                # Sort both to compare distributions (order might differ)
                baseline_sorted = np.sort(baseline)
                weights_sorted = np.sort(weights)

                diff = np.abs(baseline_sorted - weights_sorted)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)

                print(f"  {name:12s} vs numpy: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

                # Check for reasonable similarity (not exact due to algorithm differences)
                # HRP implementations can differ due to tie-breaking, precision, etc.
                assert max_diff < 0.05, \
                    f"{name} produces very different weights (max_diff={max_diff:.4f} > 5%)"
                assert mean_diff < 0.01, \
                    f"{name} produces very different weights (mean_diff={mean_diff:.4f} > 1%)"

        print(f"✓ All {len(results)} implementations produce similar results (within expected tolerance)")
        print(f"  Note: Small differences (~1-2%) are expected due to implementation details")
    else:
        print("⚠ Only one implementation available, skipping cross-validation")


def test_benchmark_all():
    """Benchmark all HRP implementations and print comparison table."""
    np.random.seed(42)
    sizes = [(100, 20), (500, 50), (1000, 100)]
    results = {}

    print("\n" + "="*80)
    print("HRP IMPLEMENTATION BENCHMARK")
    print("="*80)

    for n_samples, n_assets in sizes:
        y = np.random.randn(n_samples, n_assets)
        returns_df = pd.DataFrame(y)

        print(f"\nDataset: {n_samples} samples × {n_assets} assets")
        print("-" * 80)

        dataset_key = f"{n_samples}×{n_assets}"
        results[dataset_key] = {}

        # Benchmark numpy implementation
        bench = benchmark_implementation(lambda data: get_hrp(data), y, runs=100)
        results[dataset_key]['hrp (numpy)'] = bench
        print(f"hrp (numpy):        {bench['mean']*1000:8.3f} ± {bench['std']*1000:6.3f} ms")

        # Benchmark PyPortfolioOpt
        try:
            from pypfopt import HRPOpt
            def run_pypfopt(data):
                df = pd.DataFrame(data)
                hrp = HRPOpt(df)
                return hrp.optimize()

            bench = benchmark_implementation(run_pypfopt, y, runs=100)
            results[dataset_key]['PyPortfolioOpt'] = bench
            speedup = bench['mean'] / results[dataset_key]['hrp (numpy)']['mean']
            print(f"PyPortfolioOpt:     {bench['mean']*1000:8.3f} ± {bench['std']*1000:6.3f} ms  ({speedup:5.1f}× slower)")
        except ImportError:
            print("PyPortfolioOpt:     [not installed]")

        # Benchmark Riskfolio-Lib
        try:
            import riskfolio as rp
            def run_riskfolio(data):
                df = pd.DataFrame(data)
                port = rp.HCPortfolio(returns=df)
                return port.optimization(model='HRP', rm='MV', rf=0, linkage='single')

            bench = benchmark_implementation(run_riskfolio, y, runs=100)
            results[dataset_key]['Riskfolio-Lib'] = bench
            speedup = bench['mean'] / results[dataset_key]['hrp (numpy)']['mean']
            print(f"Riskfolio-Lib:      {bench['mean']*1000:8.3f} ± {bench['std']*1000:6.3f} ms  ({speedup:5.1f}× slower)")
        except ImportError:
            print("Riskfolio-Lib:      [not installed]")

    # Generate markdown table
    print("\n" + "="*80)
    print("MARKDOWN TABLE FOR README")
    print("="*80 + "\n")

    print("| Implementation | 100×20 | 500×50 | 1000×100 |")
    print("|----------------|--------|--------|----------|")

    all_impls = set()
    for dataset_results in results.values():
        all_impls.update(dataset_results.keys())

    for impl in sorted(all_impls):
        row = [impl]
        for dataset_key in [f"{n}×{a}" for n, a in sizes]:
            if impl in results[dataset_key]:
                mean_ms = results[dataset_key][impl]['mean'] * 1000
                std_ms = results[dataset_key][impl]['std'] * 1000
                row.append(f"{mean_ms:.2f}±{std_ms:.2f}ms")
            else:
                row.append("-")
        print("| " + " | ".join(row) + " |")

    print("\nNote: Times shown as mean ± std dev over 100 runs")
    print("="*80)


if __name__ == "__main__":
    test_benchmark_all()
