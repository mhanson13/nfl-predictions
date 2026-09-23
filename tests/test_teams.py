"""Unit tests for src.utils.teams module."""

import pytest  # type: ignore
from src.utils.teams import normalize_team_abbr, get_team_abbr_from_name, TEAM_NAME_TO_ABBR


class TestNormalizeTeamAbbr:
    """Tests for normalize_team_abbr function."""
    
    def test_canonical_abbreviations_unchanged(self):
        """Test that canonical abbreviations pass through unchanged."""
        canonical = ["KC", "SF", "GB", "NE", "NO", "TB"]
        for abbr in canonical:
            assert normalize_team_abbr(abbr) == abbr
            assert normalize_team_abbr(abbr.lower()) == abbr
    
    def test_alternate_abbreviations_normalized(self):
        """Test that alternate abbreviations are normalized to canonical."""
        test_cases = {
            "JAC": "JAX",
            "WSH": "WAS",
            "ARZ": "ARI",
            "KAN": "KC",
            "NOR": "NO",
            "TAM": "TB",
            "GNB": "GB",
            "SFO": "SF",
            "NWE": "NE",
            "RAM": "LAR",
        }
        for alt, canonical in test_cases.items():
            assert normalize_team_abbr(alt) == canonical
            assert normalize_team_abbr(alt.lower()) == canonical
    
    def test_historical_relocations(self):
        """Test historical team relocations are handled."""
        assert normalize_team_abbr("SD") == "LAC"  # San Diego -> LA Chargers
        assert normalize_team_abbr("STL") == "LAR"  # St. Louis -> LA Rams
        assert normalize_team_abbr("OAK") == "LV"  # Oakland -> Las Vegas
    
    def test_la_with_team_name_rams(self):
        """Test LA abbreviation resolves to LAR with Rams team name."""
        assert normalize_team_abbr("LA", "Los Angeles Rams") == "LAR"
        assert normalize_team_abbr("LA", "rams") == "LAR"
    
    def test_la_with_team_name_chargers(self):
        """Test LA abbreviation resolves to LAC with Chargers team name."""
        assert normalize_team_abbr("LA", "Los Angeles Chargers") == "LAC"
        assert normalize_team_abbr("LA", "chargers") == "LAC"
    
    def test_la_without_team_name(self):
        """Test LA abbreviation without team name stays as LA."""
        assert normalize_team_abbr("LA") == "LA"
        assert normalize_team_abbr("LA", None) == "LA"
        assert normalize_team_abbr("LA", "") == "LA"
    
    def test_non_string_input(self):
        """Test non-string inputs are returned unchanged."""
        assert normalize_team_abbr(None) is None
        assert normalize_team_abbr(123) == 123
        assert normalize_team_abbr(12.5) == 12.5
    
    def test_whitespace_handling(self):
        """Test whitespace is stripped."""
        assert normalize_team_abbr("  KC  ") == "KC"
        assert normalize_team_abbr("\tSF\n") == "SF"
    
    def test_case_insensitive(self):
        """Test function is case-insensitive."""
        assert normalize_team_abbr("kc") == "KC"
        assert normalize_team_abbr("Kc") == "KC"
        assert normalize_team_abbr("jac") == "JAX"


class TestGetTeamAbbrFromName:
    """Tests for get_team_abbr_from_name function."""
    
    def test_current_team_names(self):
        """Test current NFL team names resolve correctly."""
        test_cases = {
            "Kansas City Chiefs": "KC",
            "San Francisco 49ers": "SF",
            "Green Bay Packers": "GB",
            "New England Patriots": "NE",
            "Las Vegas Raiders": "LV",
            "Los Angeles Chargers": "LAC",
            "Los Angeles Rams": "LAR",
        }
        for name, expected_abbr in test_cases.items():
            assert get_team_abbr_from_name(name) == expected_abbr
    
    def test_historical_team_names(self):
        """Test historical team names resolve to current abbreviations."""
        test_cases = {
            "Oakland Raiders": "LV",
            "San Diego Chargers": "LAC",
            "St. Louis Rams": "LAR",
            "Washington Redskins": "WAS",
            "Washington Football Team": "WAS",
        }
        for name, expected_abbr in test_cases.items():
            assert get_team_abbr_from_name(name) == expected_abbr
    
    def test_unknown_team_name(self):
        """Test unknown team names return None."""
        assert get_team_abbr_from_name("Unknown Team") is None
        assert get_team_abbr_from_name("") is None
    
    def test_case_sensitivity(self):
        """Test function is case-sensitive (exact match required)."""
        # Exact match works
        assert get_team_abbr_from_name("Kansas City Chiefs") == "KC"
        # Case mismatch returns None
        assert get_team_abbr_from_name("kansas city chiefs") is None
        assert get_team_abbr_from_name("KANSAS CITY CHIEFS") is None

    def test_sportsbook_abbreviated_team_names(self):
        """Test common sportsbook/vendor team labels resolve correctly."""
        test_cases = {
            "ATL Falcons": "ATL",
            "GB Packers": "GB",
            "BUF Bills": "BUF",
            "LA Chargers": "LAC",
            "LA Rams": "LAR",
            "NY Giants": "NYG",
            "NY Jets": "NYJ",
        }
        for name, expected_abbr in test_cases.items():
            assert get_team_abbr_from_name(name) == expected_abbr


class TestTeamNameToAbbrDict:
    """Tests for TEAM_NAME_TO_ABBR dictionary."""
    
    def test_all_32_current_teams_present(self):
        """Test all 32 current NFL teams are in the dictionary."""
        current_teams = [
            "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
            "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
            "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
            "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
            "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
            "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
            "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
            "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders",
        ]
        for team in current_teams:
            assert team in TEAM_NAME_TO_ABBR, f"{team} not found in TEAM_NAME_TO_ABBR"
    
    def test_no_duplicate_abbreviations(self):
        """Test no two current teams map to the same abbreviation."""
        current_teams = [
            "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
            "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
            "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
            "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
            "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
            "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
            "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
            "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders",
        ]
        abbrs = [TEAM_NAME_TO_ABBR[team] for team in current_teams]
        assert len(abbrs) == len(set(abbrs)), "Duplicate abbreviations found"
    
    def test_historical_teams_map_to_current(self):
        """Test historical team names map to current abbreviations."""
        historical_mappings = {
            "Oakland Raiders": "LV",
            "San Diego Chargers": "LAC",
            "St. Louis Rams": "LAR",
        }
        for historical, current_abbr in historical_mappings.items():
            assert TEAM_NAME_TO_ABBR[historical] == current_abbr


@pytest.mark.parametrize("abbr,expected", [
    ("KC", "KC"),
    ("JAC", "JAX"),
    ("SD", "LAC"),
    ("OAK", "LV"),
    ("WSH", "WAS"),
])
def test_normalize_team_abbr_parametrized(abbr, expected):
    """Parametrized test for common abbreviation normalizations."""
    assert normalize_team_abbr(abbr) == expected

# Made with Bob
