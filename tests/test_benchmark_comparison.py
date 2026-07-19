#!/usr/bin/env python
"""Test benchmark comparison."""

import numpy as np
from analysis.benchmark_comparison import (
    american_to_prob,
    calculate_nfelo_prob,
    calculate_spread_baseline_prob,
    calculate_home_favorite_baseline,
    mcnemar_test,
    delong_test,
    paired_t_test,
    calculate_benchmark_metrics,
    compare_models,
)

def test_american_to_prob():
    """Test American odds to probability conversion."""
    print("Testing American odds conversion...")

    # Favorite odds
    assert abs(american_to_prob(-150) - 0.6000) < 0.01
    assert abs(american_to_prob(-200) - 0.6667) < 0.01
    assert abs(american_to_prob(-110) - 0.5238) < 0.01

    # Underdog odds
    assert abs(american_to_prob(+150) - 0.4000) < 0.01
    assert abs(american_to_prob(+200) - 0.3333) < 0.01
    assert abs(american_to_prob(+100) - 0.5000) < 0.01

    print("  [PASS] American odds conversion")


def test_nfelo_prob():
    """Test nfelo probability calculation."""
    print("\nTesting nfelo probability...")

    # Equal teams at home
    prob = calculate_nfelo_prob(1500, 1500, home_field=65)
    print(f"  Equal teams (1500 vs 1500): {prob:.3f}")
    assert 0.55 < prob < 0.65  # Home field advantage

    # Strong home team
    prob = calculate_nfelo_prob(1600, 1400, home_field=65)
    print(f"  Strong home (1600 vs 1400): {prob:.3f}")
    assert 0.75 < prob < 0.85

    # Weak home team
    prob = calculate_nfelo_prob(1400, 1600, home_field=65)
    print(f"  Weak home (1400 vs 1600): {prob:.3f}")
    assert 0.25 < prob < 0.35

    print("  [PASS] nfelo probability")


def test_spread_baseline():
    """Test spread baseline probability."""
    print("\nTesting spread baseline...")

    # Home favored by 7
    prob = calculate_spread_baseline_prob(-7.0)
    print(f"  Home -7: {prob:.3f}")
    assert 0.68 < prob < 0.74  # ~71%

    # Even game
    prob = calculate_spread_baseline_prob(0.0)
    print(f"  Even (0): {prob:.3f}")
    assert abs(prob - 0.50) < 0.01

    # Home underdog by 3
    prob = calculate_spread_baseline_prob(+3.0)
    print(f"  Home +3: {prob:.3f}")
    assert 0.38 < prob < 0.44  # ~41%

    print("  [PASS] Spread baseline")


def test_home_favorite_baseline():
    """Test home favorite baseline."""
    print("\nTesting home favorite baseline...")

    # Home favored
    prob = calculate_home_favorite_baseline(-7.0)
    assert prob == 1.0

    # Home underdog
    prob = calculate_home_favorite_baseline(+3.0)
    assert prob == 0.0

    # Even (pick home)
    prob = calculate_home_favorite_baseline(0.0)
    assert prob == 0.0  # Not favored

    print("  [PASS] Home favorite baseline")


def test_mcnemar():
    """Test McNemar test."""
    print("\nTesting McNemar test...")

    # Create test data
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    pred_a = np.array([1, 1, 1, 0, 0, 0, 0, 1])  # 6/8 correct
    pred_b = np.array([1, 1, 0, 0, 0, 0, 1, 1])  # 5/8 correct

    stat, p_value = mcnemar_test(y_true, pred_a, pred_b)

    print(f"  Statistic: {stat:.4f}, P-value: {p_value:.4f}")
    assert 0 <= p_value <= 1

    # Identical predictions should have p=1
    stat, p_value = mcnemar_test(y_true, pred_a, pred_a)
    print(f"  Identical predictions: p={p_value:.4f}")
    assert p_value == 1.0

    print("  [PASS] McNemar test")


def test_delong():
    """Test DeLong test."""
    print("\nTesting DeLong test...")

    # Create test data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    prob_a = np.random.uniform(0.2, 0.8, 100)
    prob_b = prob_a + np.random.normal(0, 0.1, 100)
    prob_b = np.clip(prob_b, 0, 1)

    stat, p_value = delong_test(y_true, prob_a, prob_b)

    print(f"  Z-statistic: {stat:.4f}, P-value: {p_value:.4f}")
    assert 0 <= p_value <= 1

    print("  [PASS] DeLong test")


def test_paired_t():
    """Test paired t-test."""
    print("\nTesting paired t-test...")

    # Create test data
    np.random.seed(42)
    metric_a = np.random.normal(0.20, 0.05, 100)
    metric_b = metric_a + np.random.normal(0.01, 0.02, 100)

    stat, p_value = paired_t_test(metric_a, metric_b)

    print(f"  T-statistic: {stat:.4f}, P-value: {p_value:.4f}")
    assert 0 <= p_value <= 1

    # Identical metrics should have p=NaN (no variance)
    stat, p_value = paired_t_test(metric_a, metric_a)
    print(f"  Identical metrics: p={p_value}")
    assert np.isnan(p_value)  # No variance in differences

    print("  [PASS] Paired t-test")


def test_benchmark_metrics():
    """Test benchmark metrics calculation."""
    print("\nTesting benchmark metrics...")

    # Create test data
    np.random.seed(42)
    n = 200
    y_true = np.random.randint(0, 2, n)
    y_prob = np.random.uniform(0.2, 0.8, n)

    metrics = calculate_benchmark_metrics(y_true, y_prob, "Test Model")

    print(f"  Model: {metrics.name}")
    print(f"  N: {metrics.n_predictions}")
    print(f"  Accuracy: {metrics.accuracy:.3f}")
    print(f"  AUC: {metrics.auc:.3f}")
    print(f"  Brier: {metrics.brier_score:.4f}")
    print(f"  LogLoss: {metrics.log_loss:.4f}")

    assert metrics.n_predictions == n
    assert 0 <= metrics.accuracy <= 1
    assert 0 <= metrics.auc <= 1
    assert 0 <= metrics.brier_score <= 1
    assert metrics.log_loss > 0

    print("  [PASS] Benchmark metrics")


def test_compare_models():
    """Test model comparison."""
    print("\nTesting model comparison...")

    # Create test data
    np.random.seed(42)
    n = 200
    y_true = np.random.randint(0, 2, n)
    prob_a = np.random.uniform(0.2, 0.8, n)
    prob_b = prob_a + np.random.normal(0, 0.05, n)
    prob_b = np.clip(prob_b, 0, 1)

    results = compare_models(y_true, prob_a, prob_b, "Model A", "Model B")

    print(f"  Comparisons: {len(results)}")

    for result in results:
        print(f"    {result.metric}: diff={result.difference:+.4f}, "
              f"p={result.p_value:.4f}, sig={result.significant}")

    assert len(results) >= 2  # At least accuracy and Brier
    assert all(0 <= r.p_value <= 1 for r in results)

    print("  [PASS] Model comparison")


def test_module_functions():
    """Test that all module functions are available."""
    print("\nTesting module functions...")

    from analysis.benchmark_comparison import (
        calculate_vegas_prob,
        load_predictions,
        run_benchmark_comparison,
        generate_comparison_report,
    )

    print("  Functions available:")
    print("    - american_to_prob")
    print("    - calculate_nfelo_prob")
    print("    - calculate_vegas_prob")
    print("    - calculate_spread_baseline_prob")
    print("    - calculate_home_favorite_baseline")
    print("    - mcnemar_test")
    print("    - delong_test")
    print("    - paired_t_test")
    print("    - calculate_benchmark_metrics")
    print("    - compare_models")
    print("    - load_predictions")
    print("    - run_benchmark_comparison")
    print("    - generate_comparison_report")

    print("  [PASS] Module functions available")
