# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Paper Trading Simulator

Simulates betting with proper bankroll management (Kelly criterion) and tracks
performance metrics over time. Enables testing different strategies without risking
real money.

Key Features:
- Kelly criterion bet sizing
- Multiple strategies (conservative, aggressive, Kelly optimal)
- ROI, Sharpe ratio, max drawdown tracking
- Bankroll curve visualization
- Risk metrics and performance analysis

Usage:
    # Simulate bets for a season
    python -m analysis.paper_trading --season 2025 --strategy kelly
    
    # Compare strategies
    python -m analysis.paper_trading --season 2025 --compare-strategies
    
    # Generate performance report
    python -m analysis.paper_trading --season 2025 --generate-report
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Paths
CLV_TRACKING_DIR = Path("analysis/clv_tracking")
PAPER_TRADING_DIR = Path("analysis/paper_trading")
PAPER_TRADING_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BetResult:
    """Result of a single bet."""
    game_id: str
    season: int
    week: int
    bet_type: str
    side: str
    team: str
    stake: float
    odds: float
    model_prob: float
    closing_prob: float
    clv: float
    won: bool
    profit: float
    bankroll_after: float


@dataclass
class PaperTradingStrategy:
    """Paper trading strategy configuration."""
    name: str
    kelly_fraction: float  # Fraction of Kelly to bet (0.25 = quarter Kelly)
    min_clv: float  # Minimum CLV to place bet
    min_edge: float  # Minimum edge (model_prob - closing_prob)
    max_bet_pct: float  # Maximum bet as % of bankroll
    
    def __str__(self) -> str:
        return f"{self.name} (Kelly={self.kelly_fraction}, MinCLV={self.min_clv}, MaxBet={self.max_bet_pct:.1%})"


# Predefined strategies
STRATEGIES = {
    "conservative": PaperTradingStrategy(
        name="Conservative",
        kelly_fraction=0.10,  # 10% of Kelly
        min_clv=0.03,  # 3% minimum CLV
        min_edge=0.03,
        max_bet_pct=0.02,  # Max 2% of bankroll
    ),
    "moderate": PaperTradingStrategy(
        name="Moderate",
        kelly_fraction=0.25,  # Quarter Kelly
        min_clv=0.02,  # 2% minimum CLV
        min_edge=0.02,
        max_bet_pct=0.05,  # Max 5% of bankroll
    ),
    "kelly": PaperTradingStrategy(
        name="Kelly Optimal",
        kelly_fraction=0.50,  # Half Kelly
        min_clv=0.01,  # 1% minimum CLV
        min_edge=0.01,
        max_bet_pct=0.10,  # Max 10% of bankroll
    ),
    "aggressive": PaperTradingStrategy(
        name="Aggressive",
        kelly_fraction=1.00,  # Full Kelly
        min_clv=0.00,  # No minimum CLV
        min_edge=0.00,
        max_bet_pct=0.15,  # Max 15% of bankroll
    ),
}


def american_to_decimal(odds: float) -> float:
    """Convert American odds to decimal odds."""
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.0


def calculate_kelly_fraction(
    win_prob: float,
    decimal_odds: float,
    kelly_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly criterion bet size.
    
    Args:
        win_prob: Probability of winning (0-1)
        decimal_odds: Decimal odds (e.g., 2.0 for +100)
        kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
    
    Returns:
        Bet size as fraction of bankroll
    """
    if pd.isna(win_prob) or pd.isna(decimal_odds):
        return 0.0
    
    # Kelly formula: f = (bp - q) / b
    # where b = decimal_odds - 1, p = win_prob, q = 1 - win_prob
    b = decimal_odds - 1.0
    p = win_prob
    q = 1.0 - p
    
    if b <= 0:
        return 0.0
    
    kelly = (b * p - q) / b
    
    # Apply fraction and ensure non-negative
    return max(0.0, kelly * kelly_fraction)


class PaperTradingSimulator:
    """Simulates paper trading with bankroll management."""
    
    def __init__(
        self,
        initial_bankroll: float = 10000.0,
        strategy: PaperTradingStrategy = STRATEGIES["moderate"]
    ):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.strategy = strategy
        self.bets: List[BetResult] = []
        self.bankroll_history: List[float] = [initial_bankroll]
    
    def should_bet(self, clv: float, edge: float) -> bool:
        """Determine if bet meets strategy criteria."""
        return clv >= self.strategy.min_clv and edge >= self.strategy.min_edge
    
    def calculate_stake(self, model_prob: float, decimal_odds: float) -> float:
        """Calculate bet stake using Kelly criterion."""
        kelly_size = calculate_kelly_fraction(
            model_prob,
            decimal_odds,
            self.strategy.kelly_fraction
        )
        
        # Apply maximum bet constraint
        kelly_size = min(kelly_size, self.strategy.max_bet_pct)
        
        # Calculate stake
        stake = self.bankroll * kelly_size
        
        return stake
    
    def place_bet(
        self,
        game_id: str,
        season: int,
        week: int,
        bet_type: str,
        side: str,
        team: str,
        model_prob: float,
        closing_odds: float,
        closing_prob: float,
        clv: float,
        won: bool
    ) -> Optional[BetResult]:
        """
        Place a bet and update bankroll.
        
        Returns:
            BetResult if bet was placed, None if bet was skipped
        """
        edge = model_prob - closing_prob
        
        # Check if bet meets criteria
        if not self.should_bet(clv, edge):
            return None
        
        # Calculate stake
        decimal_odds = american_to_decimal(closing_odds)
        stake = self.calculate_stake(model_prob, decimal_odds)
        
        if stake <= 0:
            return None
        
        # Calculate profit/loss
        if won:
            profit = stake * (decimal_odds - 1.0)
        else:
            profit = -stake
        
        # Update bankroll
        self.bankroll += profit
        self.bankroll_history.append(self.bankroll)
        
        # Record bet
        bet = BetResult(
            game_id=game_id,
            season=season,
            week=week,
            bet_type=bet_type,
            side=side,
            team=team,
            stake=stake,
            odds=closing_odds,
            model_prob=model_prob,
            closing_prob=closing_prob,
            clv=clv,
            won=won,
            profit=profit,
            bankroll_after=self.bankroll
        )
        self.bets.append(bet)
        
        return bet
    
    def get_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not self.bets:
            return {
                "n_bets": 0,
                "total_staked": 0.0,
                "total_profit": 0.0,
                "roi": 0.0,
                "win_rate": 0.0,
                "avg_stake": 0.0,
                "avg_odds": 0.0,
                "avg_clv": 0.0,
                "final_bankroll": self.initial_bankroll,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
            }
        
        total_staked = sum(bet.stake for bet in self.bets)
        total_profit = sum(bet.profit for bet in self.bets)
        wins = sum(1 for bet in self.bets if bet.won)
        
        # Calculate max drawdown
        peak = self.initial_bankroll
        max_dd = 0.0
        for bankroll in self.bankroll_history:
            if bankroll > peak:
                peak = bankroll
            dd = (peak - bankroll) / peak
            if dd > max_dd:
                max_dd = dd
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        for i in range(1, len(self.bankroll_history)):
            ret = (self.bankroll_history[i] - self.bankroll_history[i-1]) / self.bankroll_history[i-1]
            returns.append(ret)
        
        if returns:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (mean_return / std_return * np.sqrt(len(returns))) if std_return > 0 else 0.0
        else:
            sharpe = 0.0
        
        return {
            "n_bets": len(self.bets),
            "total_staked": total_staked,
            "total_profit": total_profit,
            "roi": total_profit / total_staked if total_staked > 0 else 0.0,
            "win_rate": wins / len(self.bets),
            "avg_stake": total_staked / len(self.bets),
            "avg_odds": np.mean([bet.odds for bet in self.bets]),
            "avg_clv": np.mean([bet.clv for bet in self.bets]),
            "final_bankroll": self.bankroll,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
        }


def load_clv_data(season: int, bet_type: str = "moneyline") -> pd.DataFrame:
    """Load CLV data for simulation."""
    if bet_type == "moneyline":
        clv_file = CLV_TRACKING_DIR / f"{season}_moneyline_clv.csv"
    else:
        clv_file = CLV_TRACKING_DIR / f"{season}_spread_clv.csv"
    
    if not clv_file.exists():
        raise FileNotFoundError(f"CLV data not found: {clv_file}. Run clv_tracker first.")
    
    return pd.read_csv(clv_file)


def simulate_season(
    season: int,
    strategy: PaperTradingStrategy,
    initial_bankroll: float = 10000.0,
    bet_type: str = "moneyline"
) -> PaperTradingSimulator:
    """
    Simulate paper trading for a season.
    
    Args:
        season: NFL season
        strategy: Trading strategy
        initial_bankroll: Starting bankroll
        bet_type: Type of bets (moneyline or spread)
    
    Returns:
        PaperTradingSimulator with results
    """
    # Load CLV data
    clv_df = load_clv_data(season, bet_type)
    
    # Initialize simulator
    sim = PaperTradingSimulator(initial_bankroll, strategy)
    
    # Simulate bets
    for _, row in clv_df.iterrows():
        if bet_type == "moneyline":
            won = bool(row.get("actual_win", 0))
        else:
            won = bool(row.get("actual_cover", 0))
        
        sim.place_bet(
            game_id=row["game_id"],
            season=row["season"],
            week=row["week"],
            bet_type=row["bet_type"],
            side=row["side"],
            team=row["team"],
            model_prob=row["model_prob"],
            closing_odds=row["closing_odds"],
            closing_prob=row["closing_prob"],
            clv=row["clv"],
            won=won
        )
    
    return sim


def compare_strategies(
    season: int,
    initial_bankroll: float = 10000.0,
    bet_type: str = "moneyline"
) -> Dict[str, Dict[str, Any]]:
    """
    Compare all strategies for a season.
    
    Returns:
        Dictionary mapping strategy name to metrics
    """
    results = {}
    
    for strategy_name, strategy in STRATEGIES.items():
        sim = simulate_season(season, strategy, initial_bankroll, bet_type)
        results[strategy_name] = {
            "strategy": str(strategy),
            "metrics": sim.get_metrics(),
            "bets": [vars(bet) for bet in sim.bets],
            "bankroll_history": sim.bankroll_history,
        }
    
    return results


def generate_report(season: int, bet_type: str = "moneyline") -> Path:
    """Generate paper trading report."""
    results = compare_strategies(season, bet_type=bet_type)
    
    # Save detailed results
    results_file = PAPER_TRADING_DIR / f"{season}_{bet_type}_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Generate markdown report
    report_lines = [
        f"# Paper Trading Report",
        f"",
        f"**Season:** {season}  ",
        f"**Bet Type:** {bet_type.title()}  ",
        f"**Initial Bankroll:** $10,000  ",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}  ",
        f"",
        f"---",
        f"",
        f"## Strategy Comparison",
        f"",
        f"| Strategy | Bets | Win Rate | ROI | Final Bankroll | Max DD | Sharpe |",
        f"|----------|------|----------|-----|----------------|--------|--------|",
    ]
    
    for strategy_name in ["conservative", "moderate", "kelly", "aggressive"]:
        metrics = results[strategy_name]["metrics"]
        report_lines.append(
            f"| {STRATEGIES[strategy_name].name} | "
            f"{metrics['n_bets']} | "
            f"{metrics['win_rate']:.1%} | "
            f"{metrics['roi']:.1%} | "
            f"${metrics['final_bankroll']:,.0f} | "
            f"{metrics['max_drawdown']:.1%} | "
            f"{metrics['sharpe_ratio']:.2f} |"
        )
    
    report_lines.extend([
        f"",
        f"---",
        f"",
        f"## Detailed Metrics",
        f"",
    ])
    
    for strategy_name in ["conservative", "moderate", "kelly", "aggressive"]:
        strategy = STRATEGIES[strategy_name]
        metrics = results[strategy_name]["metrics"]
        
        report_lines.extend([
            f"### {strategy.name}",
            f"",
            f"**Configuration:**",
            f"- Kelly Fraction: {strategy.kelly_fraction:.0%}",
            f"- Min CLV: {strategy.min_clv:.1%}",
            f"- Max Bet: {strategy.max_bet_pct:.1%} of bankroll",
            f"",
            f"**Performance:**",
            f"- Total Bets: {metrics['n_bets']}",
            f"- Win Rate: {metrics['win_rate']:.1%}",
            f"- Average CLV: {metrics['avg_clv']:.2%}",
            f"- Total Staked: ${metrics['total_staked']:,.0f}",
            f"- Total Profit: ${metrics['total_profit']:,.0f}",
            f"- ROI: {metrics['roi']:.1%}",
            f"- Final Bankroll: ${metrics['final_bankroll']:,.0f}",
            f"- Max Drawdown: {metrics['max_drawdown']:.1%}",
            f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
            f"",
        ])
    
    report_lines.extend([
        f"---",
        f"",
        f"## Files",
        f"",
        f"- **Results:** `{results_file.name}`",
        f"",
    ])
    
    report_content = "\n".join(report_lines)
    report_file = PAPER_TRADING_DIR / f"{season}_{bet_type}_report.md"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"✓ Generated paper trading report for {season} ({bet_type})")
    print(f"  Report: {report_file}")
    print(f"  Results: {results_file}")
    
    # Print summary
    print(f"\n  Strategy Comparison:")
    for strategy_name in ["conservative", "moderate", "kelly", "aggressive"]:
        metrics = results[strategy_name]["metrics"]
        print(f"    {STRATEGIES[strategy_name].name:15s}: "
              f"{metrics['n_bets']:3d} bets, "
              f"ROI {metrics['roi']:+6.1%}, "
              f"Final ${metrics['final_bankroll']:,.0f}")
    
    return report_file


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Paper trading simulator"
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="NFL season year",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=list(STRATEGIES.keys()),
        default="moderate",
        help="Trading strategy",
    )
    parser.add_argument(
        "--bet-type",
        type=str,
        choices=["moneyline", "spread"],
        default="moneyline",
        help="Type of bets to simulate",
    )
    parser.add_argument(
        "--initial-bankroll",
        type=float,
        default=10000.0,
        help="Initial bankroll",
    )
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Compare all strategies",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate paper trading report",
    )
    
    args = parser.parse_args()
    
    if args.generate_report or args.compare_strategies:
        generate_report(args.season, args.bet_type)
    else:
        # Simulate single strategy
        strategy = STRATEGIES[args.strategy]
        sim = simulate_season(args.season, strategy, args.initial_bankroll, args.bet_type)
        metrics = sim.get_metrics()
        
        print(f"\n{strategy}")
        print(f"  Bets: {metrics['n_bets']}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  ROI: {metrics['roi']:.1%}")
        print(f"  Final Bankroll: ${metrics['final_bankroll']:,.0f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.1%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    main()

# Made with Bob
