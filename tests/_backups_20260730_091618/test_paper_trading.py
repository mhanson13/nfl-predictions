#!/usr/bin/env python
"""Test paper trading simulator."""

from analysis.paper_trading import (
    american_to_decimal,
    calculate_kelly_fraction,
    PaperTradingStrategy,
    PaperTradingSimulator,
    STRATEGIES,
)

def test_american_to_decimal():
    """Test American to decimal odds conversion."""
    print("Testing American to decimal conversion...")

    # Favorite odds
    assert abs(american_to_decimal(-150) - 1.6667) < 0.01
    assert abs(american_to_decimal(-200) - 1.50) < 0.01
    assert abs(american_to_decimal(-110) - 1.9091) < 0.01

    # Underdog odds
    assert abs(american_to_decimal(+150) - 2.50) < 0.01
    assert abs(american_to_decimal(+200) - 3.00) < 0.01
    assert abs(american_to_decimal(+100) - 2.00) < 0.01

    print("  [PASS] American to decimal conversion")

def test_kelly_criterion():
    """Test Kelly criterion calculation."""
    print("\nTesting Kelly criterion...")

    # Example: 60% win prob, 2.0 decimal odds (even money)
    kelly = calculate_kelly_fraction(0.60, 2.0, 1.0)  # Full Kelly
    print(f"  60% win prob, 2.0 odds, full Kelly: {kelly:.2%}")
    assert abs(kelly - 0.20) < 0.01  # Should be 20%

    # Quarter Kelly
    quarter_kelly = calculate_kelly_fraction(0.60, 2.0, 0.25)
    print(f"  60% win prob, 2.0 odds, quarter Kelly: {quarter_kelly:.2%}")
    assert abs(quarter_kelly - 0.05) < 0.01  # Should be 5%

    # No edge (50% win prob, 2.0 odds)
    no_edge = calculate_kelly_fraction(0.50, 2.0, 1.0)
    print(f"  50% win prob, 2.0 odds (no edge): {no_edge:.2%}")
    assert no_edge == 0.0  # Should be 0%

    print("  [PASS] Kelly criterion calculation")

def test_strategies():
    """Test predefined strategies."""
    print("\nTesting predefined strategies...")

    assert "conservative" in STRATEGIES
    assert "moderate" in STRATEGIES
    assert "kelly" in STRATEGIES
    assert "aggressive" in STRATEGIES

    # Check conservative is most cautious
    conservative = STRATEGIES["conservative"]
    aggressive = STRATEGIES["aggressive"]

    assert conservative.kelly_fraction < aggressive.kelly_fraction
    assert conservative.min_clv > aggressive.min_clv
    assert conservative.max_bet_pct < aggressive.max_bet_pct

    print("  Strategies:")
    for name, strategy in STRATEGIES.items():
        print(f"    {name:12s}: Kelly={strategy.kelly_fraction:.0%}, "
              f"MinCLV={strategy.min_clv:.1%}, MaxBet={strategy.max_bet_pct:.1%}")

    print("  [PASS] Predefined strategies")

def test_simulator():
    """Test paper trading simulator."""
    print("\nTesting paper trading simulator...")

    sim = PaperTradingSimulator(
        initial_bankroll=10000.0,
        strategy=STRATEGIES["moderate"]
    )

    assert sim.bankroll == 10000.0
    assert len(sim.bets) == 0

    # Place a winning bet
    bet = sim.place_bet(
        game_id="test_001",
        season=2025,
        week=1,
        bet_type="moneyline",
        side="home",
        team="KC",
        model_prob=0.65,
        closing_odds=-150,
        closing_prob=0.60,
        clv=0.05,
        won=True
    )

    assert bet is not None, "Bet should be placed (meets criteria)"
    assert bet.won == True
    assert bet.profit > 0
    assert sim.bankroll > 10000.0

    print(f"  Placed winning bet: stake=${bet.stake:.2f}, profit=${bet.profit:.2f}")
    print(f"  Bankroll after: ${sim.bankroll:.2f}")

    # Place a losing bet
    bet2 = sim.place_bet(
        game_id="test_002",
        season=2025,
        week=1,
        bet_type="moneyline",
        side="away",
        team="BUF",
        model_prob=0.55,
        closing_odds=+150,
        closing_prob=0.40,
        clv=0.15,
        won=False
    )

    assert bet2 is not None
    assert bet2.won == False
    assert bet2.profit < 0

    print(f"  Placed losing bet: stake=${bet2.stake:.2f}, profit=${bet2.profit:.2f}")
    print(f"  Bankroll after: ${sim.bankroll:.2f}")

    # Get metrics
    metrics = sim.get_metrics()
    assert metrics["n_bets"] == 2
    assert 0 < metrics["win_rate"] < 1

    print(f"  Metrics: {metrics['n_bets']} bets, win rate {metrics['win_rate']:.1%}")

    print("  [PASS] Paper trading simulator")

def test_module_functions():
    """Test that all module functions are available."""
    print("\nTesting module functions...")

    from analysis.paper_trading import (
        load_clv_data,
        simulate_season,
        compare_strategies,
        generate_report,
    )

    print("  Functions available:")
    print("    - load_clv_data")
    print("    - simulate_season")
    print("    - compare_strategies")
    print("    - generate_report")

    print("  [PASS] Module functions available")
