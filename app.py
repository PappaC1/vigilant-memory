import io
import re
from typing import Dict, List, Optional
import pandas as pd
from pypdf import PdfReader
import streamlit as st

# ==========================================
# 1. PREDICTION ENGINES
# ==========================================


class UHISEngine:

  def calculate(self, metrics: Dict[str, float]) -> float:
    if not metrics:
      return 0.0
    return round(sum(metrics.values()) / len(metrics), 2)


class ProbabilityEngine:

  def calculate(self, uhis: float, total_race_uhis: float) -> float:
    if total_race_uhis <= 0:
      return 0.0
    return round((uhis / total_race_uhis) * 100, 2)


class FairOddsEngine:

  def calculate(self, probability: float) -> float:
    if probability <= 0:
      return 0.0
    return round(100 / probability, 2)


class OverlayEngine:

  def calculate(self, fair_odds: float, market_odds: float) -> float:
    if fair_odds <= 0:
      return 0.0
    return round(((market_odds - fair_odds) / fair_odds) * 100, 2)


class EESEngine:

  def calculate(self, uhis: float, confidence: float, opportunity: float) -> float:
    score = (uhis * 0.50) + (opportunity * 0.30) + (confidence * 0.20)
    return round(score, 2)


class RecommendationEngine:

  def generate(self, ees: float, overlay: float) -> str:
    if ees >= 92:
      return "BEST_BET"
    if overlay >= 25:
      return "VALUE_BET"
    if ees >= 85:
      return "PLACE_BET"
    return "WATCH"


# ==========================================
# 2. GOLD CIRCLE PARSER & SCORING PIPELINE
# ==========================================


class GoldCircleParser:
  """
    Parses Gold Circle / South African racecard PDF text.
    Extracts race headers, horse names, draws, weights, jockeys, trainers, and odds.
    """

  # Matches headers like: "RACE 1 - 1200M" or "Race 2  1600m"
  RACE_HEADER_PATTERN = re.compile(
      r"(?i)\bRACE\s+(\d+)\b[^\n]*?(\d{3,4})\s*M\b"
  )

  # Matches runner lines such as:
  # "1  (3) THUNDER STRIKE   60.0  R Fourie       J Snaith       4/1"
  # or "1  THUNDER STRIKE (3)   60.0  R Fourie       J Snaith       5.00"
  RUNNER_PATTERN = re.compile(
      r"^\s*(?P<num>\d{1,2})\s+"
      r"(?:\((?P<draw1>\d{1,2})\)\s+)?"
      r"(?P<horse>[A-Z][A-Z\s\'\-\.]{2,28}?)\s+"
      r"(?:\((?P<draw2>\d{1,2})\)\s+)?"
      r"(?P<weight>\d{2}\.\d)\s+"
      r"(?P<jockey>[A-Za-z\.\s\'\-]{3,22}?)\s{2,}"
      r"(?P<trainer>[A-Za-z\.\s\'\-]{3,22}?)"
      r"(?:\s+(?P<odds>\d+(?:/\d+|\.\d+)?))?\s*$",
      re.MULTILINE,
  )

  # Fallback simpler pattern if columns are separated by single spaces
  SIMPLE_RUNNER_PATTERN = re.compile(
      r"^\s*(?P<num>\d{1,2})\s+\(?(?P<draw>\d{1,2})\)?\s+"
      r"(?P<horse>[A-Z][A-Z\s\'\-]{2,25})\s+"
      r"(?P<weight>\d{2}\.\d)",
      re.MULTILINE,
  )

  @staticmethod
  def parse_odds_to_decimal(odds_str: Optional[str]) -> float:
    """Converts SA fractional odds (e.g., '5/2', '4/1') or decimal strings to decimal odds."""
    if not odds_str:
      return 6.00  # Default neutral market odds if missing
    odds_str = odds_str.strip()
    try:
      if "/" in odds_str:
        num, den = odds_str.split("/")
        return round((float(num) / float(den)) + 1.0, 2)
      val = float(odds_str)
      return round(val if val > 1.0 else val + 1.0, 2)
    except ValueError:
      return 6.00

  def parse(self, text: str) -> Dict[str, List[dict]]:
    races: Dict[str, List[dict]] = {}
    headers = list(self.RACE_HEADER_PATTERN.finditer(text))

    if not headers:
      return {}

    for i, match in enumerate(headers):
      race_num = match.group(1)
      distance = match.group(2)
      race_title = f"Race {race_num} - {distance}m"

      start_idx = match.end()
      end_idx = headers[i + 1].start() if i + 1 < len(headers) else len(text)
      race_block = text[start_idx:end_idx]

      runners = []
      seen_horses = set()

      for r_match in self.RUNNER_PATTERN.finditer(race_block):
        horse = r_match.group("horse").strip().title()
        if horse in seen_horses or len(horse) < 3:
          continue
        seen_horses.add(horse)

        draw_str = r_match.group("draw1") or r_match.group("draw2") or "5"
        draw = int(draw_str)
        weight = float(r_match.group("weight"))
        jockey = r_match.group("jockey").strip()
        trainer = r_match.group("trainer").strip()
        market_odds = self.parse_odds_to_decimal(r_match.group("odds"))

        runners.append(
            self._build_runner_profile(
                horse=horse,
                draw=draw,
                weight=weight,
                jockey=jockey,
                trainer=trainer,
                market_odds=market_odds,
                distance=int(distance),
            )
        )

      # Try simpler pattern if primary regex matched nothing in this block
      if not runners:
        for s_match in self.SIMPLE_RUNNER_PATTERN.finditer(race_block):
          horse = s_match.group("horse").strip().title()
          if horse in seen_horses or len(horse) < 3:
            continue
          seen_horses.add(horse)

          runners.append(
              self._build_runner_profile(
                  horse=horse,
                  draw=int(s_match.group("draw")),
                  weight=float(s_match.group("weight")),
                  jockey="Extracted",
                  trainer="Extracted",
                  market_odds=5.00,
                  distance=int(distance),
              )
          )

      if runners:
        races[race_title] = runners

    return races

  def _build_runner_profile(
      self,
      horse: str,
      draw: int,
      weight: float,
      jockey: str,
      trainer: str,
      market_odds: float,
      distance: int,
  ) -> dict:
    """
        Calculates initial UHIS, Confidence, and Opportunity metrics
        from parsed racecard attributes.
        """
    # Market implied strength (lower odds -> higher market rating)
    market_score = max(60.0, min(96.0, 100.0 - (market_odds * 2.2)))

    # Draw advantage (inner draws score higher over sprints/turns)
    draw_penalty = (draw - 1) * (2.0 if distance <= 1400 else 1.2)
    draw_score = max(62.0, min(95.0, 92.0 - draw_penalty))

    # Weight rating (standard SA scale ~52kg to 62kg)
    weight_score = max(65.0, min(94.0, 85.0 + ((weight - 57.0) * 1.5)))

    metrics = {
        "form": round((market_score * 0.6) + (weight_score * 0.4), 1),
        "speed": round((market_score * 0.5) + (weight_score * 0.5), 1),
        "trainer": 82.0,
        "jockey": 83.0,
        "distance": 84.0,
        "track": 83.0,
        "going": 82.0,
        "draw": round(draw_score, 1),
        "market": round(market_score, 1),
        "pace": round((draw_score + market_score) / 2, 1),
    }

    uhis = UHISEngine().calculate(metrics)
    confidence = round((market_score + weight_score) / 2, 1)
    opportunity = round((draw_score + market_score) / 2, 1)

    return {
        "horse": horse,
        "jockey": jockey,
        "trainer": trainer,
        "draw": draw,
        "weight": weight,
        "market_odds": market_odds,
        "uhis": uhis,
        "conf": confidence,
        "opp": opportunity,
    }


def detect_provider(text: str) -> str:
  upper = text.upper()
  if (
      "GOLD CIRCLE" in upper
      or "GREYVILLE" in upper
      or "SCOTTSVILLE" in upper
      or "KENILWORTH" in upper
      or "TURFFONTEIN" in upper
      or "VAAL" in upper
  ):
    return "GOLDCIRCLE / SA RACING"
  if "HONG KONG JOCKEY CLUB" in upper:
    return "HKJC"
  if "RACING POST" in upper:
    return "RACINGPOST"
  return "GENERIC / UNKNOWN"


def get_fallback_sample_races() -> Dict[str, List[dict]]:
  return {
      "Sample Race 1 - 1200m (Demo Fallback)": [
          {
              "horse": "Cape Crusader",
              "jockey": "R. Fourie",
              "trainer": "J. Snaith",
              "draw": 2,
              "weight": 60.0,
              "market_odds": 2.80,
              "uhis": 91.20,
              "conf": 94,
              "opp": 92,
          },
          {
              "horse": "Highveld Star",
              "jockey": "G. Lerena",
              "trainer": "M. de Kock",
              "draw": 5,
              "weight": 58.5,
              "market_odds": 5.50,
              "uhis": 85.40,
              "conf": 85,
              "opp": 89,
          },
          {
              "horse": "Durban Dash",
              "jockey": "K. de Melo",
              "trainer": "D. Kannemeyer",
              "draw": 7,
              "weight": 57.0,
              "market_odds": 4.00,
              "uhis": 83.00,
              "conf": 84,
              "opp": 81,
          },
          {
              "horse": "Karoo Breeze",
              "jockey": "C. Zackey",
              "trainer": "A. Fortune",
              "draw": 10,
              "weight": 54.5,
              "market_odds": 15.00,
              "uhis": 74.50,
              "conf": 72,
              "opp": 76,
          },
      ]
  }


def analyze_meeting(extracted_text: str) -> tuple[Dict[str, pd.DataFrame], bool]:
  parser = GoldCircleParser()
  parsed_races = parser.parse(extracted_text)

  used_fallback = False
  if not parsed_races:
    parsed_races = get_fallback_sample_races()
    used_fallback = True

  prob_engine = ProbabilityEngine()
  odds_engine = FairOddsEngine()
  overlay_engine = OverlayEngine()
  ees_engine = EESEngine()
  rec_engine = RecommendationEngine()

  analyzed_races: Dict[str, pd.DataFrame] = {}

  for race_name, runners in parsed_races.items():
    total_race_uhis = sum(r["uhis"] for r in runners)
    rows = []

    for r in runners:
      prob = prob_engine.calculate(r["uhis"], total_race_uhis)
      fair_odds = odds_engine.calculate(prob)
      overlay = overlay_engine.calculate(fair_odds, r["market_odds"])
      ees = ees_engine.calculate(r["uhis"], r["conf"], r["opp"])
      rec = rec_engine.generate(ees, overlay)

      rows.append
          "Horse": r["horse"],
          "Draw": r["draw"],
          "Weight": r.get("weight", 58.0),
