from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


# ── ESPN CDN logo abbreviations ────────────────────────────────────────────────
_ESPN_ABBR: dict[str, str] = {
    "ARI": "ari", "ATL": "atl", "BAL": "bal", "BUF": "buf",
    "CAR": "car", "CHI": "chi", "CIN": "cin", "CLE": "cle",
    "DAL": "dal", "DEN": "den", "DET": "det", "GB": "gb",
    "HOU": "hou", "IND": "ind", "JAX": "jax", "KC": "kc",
    "LA": "lar", "LAC": "lac", "LV": "lv", "MIA": "mia",
    "MIN": "min", "NE": "ne", "NO": "no", "NYG": "nyg",
    "NYJ": "nyj", "PHI": "phi", "PIT": "pit", "SEA": "sea",
    "SF": "sf", "TB": "tb", "TEN": "ten", "WAS": "wsh",
}

_TEAM_COLOR: dict[str, str] = {
    "ARI": "#97233F", "ATL": "#A71930", "BAL": "#9B59D0", "BUF": "#00338D",
    "CAR": "#0085CA", "CHI": "#E87722", "CIN": "#FB4F14", "CLE": "#CC5500",
    "DAL": "#4189E0", "DEN": "#FB4F14", "DET": "#0076B6", "GB": "#4A8C3F",
    "HOU": "#03A4D0", "IND": "#4484D6", "JAX": "#00A8B5", "KC": "#E31837",
    "LA": "#4189E0", "LAC": "#0080C6", "LV": "#A8A8A8", "MIA": "#008E97",
    "MIN": "#7B52B9", "NE": "#4A7FC1", "NO": "#C9A84C", "NYG": "#4A6FBF",
    "NYJ": "#2EAD76", "PHI": "#4F9E8F", "PIT": "#FFB612", "SEA": "#4A7FC1",
    "SF": "#AA0000", "TB": "#D50A0A", "TEN": "#4B92DB", "WAS": "#8B3A3A",
}


def _logo(team: str) -> str:
    abbr = _ESPN_ABBR.get(team, team.lower())
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{abbr}.png"


def _color(team: str) -> str:
    return _TEAM_COLOR.get(team, "#2563eb")


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _conf(p: float) -> tuple[str, str]:
    if p >= 0.70:
        return "High Confidence", "#1a7f37"
    if p >= 0.58:
        return "Moderate", "#d97706"
    return "Lean", "#6b7280"


def _safe(row: pd.Series, col: str, default: float = float("nan")) -> float:
    v = row.get(col, default)
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return default


# ── Detail stat configuration ─────────────────────────────────────────────────
# Each entry: (label, home_col, away_col, higher_is_better)
_STAT_DEFS: list[tuple[str, str, str, bool]] = [
    ("Off. EPA / Play (Last 5)",   "home_off_epa_last_n",          "away_off_epa_last_n",          True),
    ("Def. EPA Allowed / Play",    "home_def_epa_allowed_last_n",  "away_def_epa_allowed_last_n",  False),
    ("Avg Point Diff (Last 5)",    "home_point_diff_last_n",       "away_point_diff_last_n",       True),
    ("ATS Cover Rate (Last 5)",    "home_ats_rate_last_n",         "away_ats_rate_last_n",         True),
    ("Avg Pts Scored (Last 5)",    "home_score_for_last_n",        "away_score_for_last_n",        True),
    ("Avg Pts Allowed (Last 5)",   "home_score_against_last_n",    "away_score_against_last_n",    False),
    ("Avg Offensive Plays/Game",   "home_pace_plays_last_n",       "away_pace_plays_last_n",       True),
    ("Days of Rest",               "home_rest",                    "away_rest",                    True),
]


def _stat_rows_js(row: pd.Series, home: str, away: str) -> str:
    stats = []
    for label, hcol, acol, higher_better in _STAT_DEFS:
        hv = _safe(row, hcol)
        av = _safe(row, acol)
        if hv != hv and av != av:
            continue
        home_better = (hv > av) if higher_better else (hv < av)
        stats.append({
            "label": label,
            "home": round(hv, 4) if hv == hv else None,
            "away": round(av, 4) if av == av else None,
            "homeColor": _color(home),
            "awayColor": _color(away),
            "homeBetter": home_better,
            "higherBetter": higher_better,
            "away": round(av, 4) if av == av else None,
            "awayTeam": away,
            "homeTeam": home,
        })
    return json.dumps(stats)


# ── Per-card HTML ─────────────────────────────────────────────────────────────

def _card_html(idx: int, row: pd.Series) -> str:
    away = str(row["away_team"])
    home = str(row["home_team"])
    away_spread = float(row["away_spread_line"])
    home_spread = float(row["home_spread_line"])
    prob_home = float(row["pred_prob_home_cover"])
    prob_away = 1.0 - prob_home
    pick = str(row["recommended_pick"])
    gameday = str(row.get("gameday", ""))[:10]
    total = _safe(row, "total_line")
    total_str = f"O/U {total:.1f}" if total == total else ""
    away_score = row.get("away_score")
    home_score = row.get("home_score")
    has_scores = pd.notna(away_score) and pd.notna(home_score)
    score_line = (
        f"Final: {away} {int(float(away_score))} - {home} {int(float(home_score))}"
        if has_scores
        else ""
    )
    has_result = "was_correct" in row.index and pd.notna(row.get("was_correct"))

    result_label = ""
    result_color = ""
    actual_result = row.get("actual_result", "") if has_result else ""
    if has_result:
      correct = bool(row.get("was_correct"))
      result_label = "Correct" if correct else "Incorrect"
      result_color = "#1a7f37" if correct else "#b42318"

    pick_is_home = prob_home >= 0.5
    pick_prob = prob_home if pick_is_home else prob_away
    pick_color = _color(home) if pick_is_home else _color(away)
    conf_label, conf_color = _conf(pick_prob)

    away_spread_str = f"{away_spread:+.1f}"
    home_spread_str = f"{home_spread:+.1f}"

    away_hl = "box-shadow:inset 4px 0 0 #f59e0b;" if not pick_is_home else ""
    home_hl = "box-shadow:inset -4px 0 0 #f59e0b;" if pick_is_home else ""

    stats_js = _stat_rows_js(row, home, away)
    
    # O/U data
    ou_pick = str(row.get("ou_pick", ""))
    prob_total_over = _safe(row, "pred_prob_total_over")
    prob_total_under = 1.0 - prob_total_over if prob_total_over == prob_total_over else float("nan")
    total_display = f"{total:.1f}" if total == total else "N/A"
    
    # O/U result for historical games
    has_ou_result = "was_correct_ou" in row.index and pd.notna(row.get("was_correct_ou"))
    ou_result_label = ""
    ou_result_color = ""
    ou_actual_result = ""
    ou_result_html = ""
    if has_ou_result:
      actual_over = bool(row.get("actual_total_over"))
      ou_correct = bool(row.get("was_correct_ou"))
      ou_result_label = "Correct" if ou_correct else "Incorrect"
      ou_result_color = "#1a7f37" if ou_correct else "#b42318"
      ou_actual_result = "Over Hit" if actual_over else "Under Hit"
      ou_result_html = f'<div style="margin-top:6px; padding-top:6px; border-top:1px solid rgba(255,255,255,0.1); font-size:.65rem; color:#10b981; text-align:center;"><span style="background:{ou_result_color}; padding:2px 8px; border-radius:4px; color:#fff; font-weight:700;">{ou_result_label}</span> &ndash; {ou_actual_result}</div>'

    # Store correctness and high confidence status for filtering
    is_high_conf = max(prob_home, prob_away) >= 0.70
    is_high_conf_ou = max(prob_total_over, prob_total_under) >= 0.70 if prob_total_over == prob_total_over else False
    was_correct = 1 if (has_result and bool(row.get("was_correct"))) else 0 if has_result else -1  # -1 = no result yet
    was_correct_ou = 1 if (has_ou_result and bool(row.get("was_correct_ou"))) else 0 if has_ou_result else -1
    
    return f"""
<div class="card" data-week="{int(float(row.get('week', 0)))}" data-correct="{was_correct}" data-correct-ou="{was_correct_ou}" data-high-conf="{int(is_high_conf)}" data-high-conf-ou="{int(is_high_conf_ou)}" data-ou-pick="{ou_pick}" data-ou-prob="{prob_total_over:.3f}" onclick="openModal({idx})" role="button" tabindex="0"
     onkeydown="if(event.key==='Enter')openModal({idx})">
  <div class="card-top">
    <span class="card-date">{gameday}</span>
    <span class="card-total">{total_str}</span>
  </div>
  <div class="matchup">
    <div class="team-col" style="{away_hl}">
      <img class="logo" src="{_logo(away)}" alt="{away}" onerror="this.style.opacity=0">
      <div class="team-abbr">{away}</div>
      <div class="spread" style="color:{_color(away)};">{away_spread_str}</div>
      <div class="prob-num">{_pct(prob_away)}</div>
    </div>
    <div class="vs-col"><span class="at-sign">@</span></div>
    <div class="team-col" style="{home_hl}">
      <img class="logo" src="{_logo(home)}" alt="{home}" onerror="this.style.opacity=0">
      <div class="team-abbr">{home}</div>
      <div class="spread" style="color:{_color(home)};">{home_spread_str}</div>
      <div class="prob-num">{_pct(prob_home)}</div>
    </div>
  </div>
  <div class="prob-bar-wrap">
    <div style="width:{prob_away*100:.1f}%;background:{_color(away)};height:100%;border-radius:99px 0 0 99px;"></div>
    <div style="width:{prob_home*100:.1f}%;background:{_color(home)};height:100%;border-radius:0 99px 99px 0;"></div>
  </div>
  <div class="prob-bar-labels">
    <span>{away} {_pct(prob_away)}</span><span>{home} {_pct(prob_home)}</span>
  </div>
  <div class="pick-banner" style="background:{pick_color};">
    <span class="pick-label">PICK</span>
    <span class="pick-text">{pick}</span>
    <span class="conf-badge" style="background:{conf_color};">{conf_label} &middot; {_pct(pick_prob)}</span>
  </div>
  <div class="ou-banner" style="background:#7c3aed; display:none; padding:12px;">
    <div style="font-size:.7rem; color:#e2e8f0; text-transform:uppercase; letter-spacing:.05em; margin-bottom:6px;">O/U {total_display}</div>
    <div style="display:flex; gap:8px; margin-bottom:6px;">
      <div style="flex:1; text-align:center; background:rgba(255,255,255,0.1); padding:6px 8px; border-radius:6px;">
        <div style="font-size:.65rem; color:#c4b5fd;">OVER</div>
        <div style="font-size:.85rem; font-weight:700; color:#fff;">{_pct(prob_total_over) if prob_total_over == prob_total_over else 'N/A'}</div>
      </div>
      <div style="flex:1; text-align:center; background:rgba(255,255,255,0.1); padding:6px 8px; border-radius:6px;">
        <div style="font-size:.65rem; color:#c4b5fd;">UNDER</div>
        <div style="font-size:.85rem; font-weight:700; color:#fff;">{_pct(prob_total_under) if prob_total_under == prob_total_under else 'N/A'}</div>
      </div>
    </div>
    <div style="font-size:.65rem; color:#d1d5db; text-align:center;">{ou_pick}</div>
    {ou_result_html}
  </div>
  {f'<div class="result-banner"><span class="result-chip" style="background:{result_color};">{result_label}</span><span class="result-text">{actual_result}</span></div>' if has_result else ''}
  {f'<div class="score-row">{score_line}</div>' if has_scores else ''}
  <div class="click-hint">Click for detailed stats &rsaquo;</div>
</div>
<script>window._matchupData = window._matchupData || {{}};
window._matchupData[{idx}] = {{
  away: "{away}", home: "{home}",
  awaySpread: "{away_spread_str}", homeSpread: "{home_spread_str}",
  awayProb: {prob_away:.4f}, homeProb: {prob_home:.4f},
  pick: {json.dumps(pick)}, conf: {json.dumps(conf_label)}, confColor: {json.dumps(conf_color)},
  pickColor: {json.dumps(pick_color)},
  awayLogo: "{_logo(away)}", homeLogo: "{_logo(home)}",
  awayColor: "{_color(away)}", homeColor: "{_color(home)}",
  gameday: "{gameday}", total: {json.dumps(total_str)},
  week: {int(float(row.get('week', 0)))},
  hasScores: {str(bool(has_scores)).lower()},
  scoreLine: {json.dumps(score_line)},
  hasResult: {str(bool(has_result)).lower()},
  resultLabel: {json.dumps(result_label)},
  resultColor: {json.dumps(result_color)},
  actualResult: {json.dumps(str(actual_result))},
  stats: {stats_js}
}};</script>
"""


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f0f1a;color:#e5e7eb;min-height:100vh}
header{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
  border-bottom:2px solid #2563eb;padding:24px 32px;display:flex;
  align-items:center;gap:16px}
header h1{font-size:1.8rem;font-weight:800;color:#f8fafc}
header .subtitle{font-size:.95rem;color:#94a3b8;margin-top:4px}
.badge{background:#2563eb;color:#fff;border-radius:999px;padding:4px 14px;
  font-size:.78rem;font-weight:700;letter-spacing:.05em;margin-left:auto}
main{max-width:1280px;margin:0 auto;padding:32px 16px}
.filter-row{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:0 0 18px;flex-wrap:wrap}
.filter-label{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em}
.filter-select{background:#1a1a2e;border:1px solid #334155;color:#e2e8f0;border-radius:10px;padding:8px 10px;font-weight:700}
.filter-note{font-size:.72rem;color:#64748b}
.stats-row{display:flex;gap:16px;margin-bottom:32px;flex-wrap:wrap}
.stat-box{background:#1a1a2e;border:1px solid #2d2d4e;border-radius:12px;
  padding:16px 24px;flex:1;min-width:140px}
.stat-box .label{font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em}
.stat-box .value{font-size:1.5rem;font-weight:800;color:#f8fafc;margin-top:4px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:20px}
.card{background:#1a1a2e;border:1px solid #2d2d4e;border-radius:16px;
  overflow:hidden;cursor:pointer;transition:transform .15s,box-shadow .15s}
.card:hover{transform:translateY(-3px);box-shadow:0 12px 40px rgba(0,0,0,.55)}
.card-top{display:flex;justify-content:space-between;padding:10px 14px 0;
  font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em}
.matchup{display:flex;align-items:stretch;padding:10px 8px 6px;gap:4px}
.team-col{flex:1;display:flex;flex-direction:column;align-items:center;
  gap:4px;padding:8px 4px;border-radius:10px}
.vs-col{display:flex;align-items:center;justify-content:center;padding:0 6px}
.at-sign{font-size:1.1rem;color:#475569;font-weight:700}
.logo{width:68px;height:68px;object-fit:contain}
.team-abbr{font-size:1.05rem;font-weight:800;color:#f1f5f9}
.spread{font-size:1.35rem;font-weight:900;letter-spacing:-.02em}
.prob-num{font-size:.75rem;color:#94a3b8}
.prob-bar-wrap{display:flex;height:6px;margin:6px 12px 0;border-radius:99px;overflow:hidden}
.prob-bar-labels{display:flex;justify-content:space-between;
  padding:4px 12px 8px;font-size:.65rem;color:#64748b}
.pick-banner{display:flex;align-items:center;gap:8px;padding:10px 14px;flex-wrap:wrap}
.pick-label{font-size:.6rem;font-weight:900;letter-spacing:.12em;
  color:rgba(255,255,255,.7);text-transform:uppercase}
.pick-text{flex:1;font-size:.85rem;font-weight:700;color:#fff}
.conf-badge{font-size:.63rem;font-weight:700;border-radius:999px;
  padding:2px 10px;color:#fff;white-space:nowrap}
.result-banner{display:flex;align-items:center;gap:8px;padding:8px 12px 4px;flex-wrap:wrap}
.result-chip{font-size:.63rem;font-weight:800;border-radius:999px;padding:2px 10px;color:#fff}
.result-text{font-size:.72rem;color:#cbd5e1;font-weight:600}
.score-row{font-size:.72rem;color:#e2e8f0;font-weight:700;padding:2px 12px 6px}
.click-hint{font-size:.65rem;color:#475569;text-align:center;
  padding:6px 0 10px;letter-spacing:.04em}
/* ── MODAL ── */
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.75);
  display:none;z-index:1000;align-items:center;justify-content:center;padding:16px}
.overlay.open{display:flex}
.modal{background:#1a1a2e;border:1px solid #2d2d4e;border-radius:20px;
  max-width:760px;width:100%;max-height:92vh;overflow-y:auto;
  box-shadow:0 24px 80px rgba(0,0,0,.7)}
.modal-header{display:flex;align-items:center;padding:20px 24px 14px;
  border-bottom:1px solid #2d2d4e;gap:12px}
.modal-teams{flex:1;display:flex;align-items:center;justify-content:center;gap:16px}
.modal-logo{width:56px;height:56px;object-fit:contain}
.modal-vs{font-size:1rem;color:#475569;font-weight:700;padding:0 4px}
.modal-team-block{display:flex;flex-direction:column;align-items:center;gap:2px}
.modal-team-name{font-size:1.1rem;font-weight:800;color:#f1f5f9}
.modal-spread{font-size:.85rem;font-weight:700}
.modal-close{background:none;border:none;color:#64748b;font-size:1.4rem;
  cursor:pointer;padding:4px 8px;border-radius:8px;transition:background .15s}
.modal-close:hover{background:#2d2d4e;color:#f1f5f9}
.modal-pick-row{padding:12px 24px;border-bottom:1px solid #2d2d4e;
  display:flex;align-items:center;gap:10px}
.modal-pick-label{font-size:.65rem;font-weight:900;letter-spacing:.1em;
  color:rgba(255,255,255,.6);text-transform:uppercase}
.modal-pick-text{font-size:.95rem;font-weight:700;color:#fff;flex:1}
.modal-result-row{padding:10px 24px;border-bottom:1px solid #2d2d4e;display:none;align-items:center;gap:10px}
.modal-body{padding:20px 24px}
.section-title{font-size:.7rem;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:#64748b;margin-bottom:16px}
.stat-compare{display:flex;flex-direction:column;gap:16px}
.stat-row{display:flex;flex-direction:column;gap:5px}
.stat-row-label{font-size:.78rem;color:#94a3b8;text-align:center;margin-bottom:3px}
.stat-values{display:flex;justify-content:space-between;align-items:center;gap:8px}
.stat-val{font-size:.95rem;font-weight:700;min-width:64px;text-align:center}
.stat-val.winner{color:#f1f5f9}
.stat-val.loser{color:#6b7280}
.stat-bar-wrap{flex:1;display:flex;height:8px;border-radius:99px;overflow:hidden;background:#2d2d4e}
.stat-row-footer{display:flex;justify-content:space-between;
  font-size:.65rem;color:#475569;margin-top:2px}
.no-stats{color:#475569;font-size:.85rem;text-align:center;padding:24px 0}
.view-toggle-btn{background:#2563eb;border:none;color:#fff;padding:8px 16px;border-radius:8px;font-weight:700;cursor:pointer;font-size:.85rem;transition:background .2s}
.view-toggle-btn:hover{background:#1e40af}
.ou-banner{display:flex;align-items:center;gap:6px}
body.ou-mode .card .pick-banner{display:none}
body.ou-mode .card .ou-banner{display:flex!important}
body.ou-mode #view-toggle-btn{background:#7c3aed}
body.ou-mode #view-toggle-btn:hover{background:#6d28d9}
footer{text-align:center;padding:32px;color:#374151;font-size:.78rem}
"""

# ── JavaScript ────────────────────────────────────────────────────────────────

_JS = """
function fmt(v, decimals) {
  if (v === null || v === undefined || isNaN(v)) return 'N/A';
  var d = decimals !== undefined ? decimals : 3;
  var s = v.toFixed(d);
  return v > 0 ? '+' + s : s;
}
function pct(v) {
  if (v === null || isNaN(v)) return 'N/A';
  return (v * 100).toFixed(1) + '%';
}
function openModal(idx) {
  var d = window._matchupData[idx];
  if (!d) return;
  document.getElementById('m-away-logo').src = d.awayLogo;
  document.getElementById('m-away-logo').alt = d.away;
  document.getElementById('m-home-logo').src = d.homeLogo;
  document.getElementById('m-home-logo').alt = d.home;
  document.getElementById('m-away-name').textContent = d.away;
  document.getElementById('m-home-name').textContent = d.home;
  document.getElementById('m-away-spread').textContent = d.awaySpread;
  document.getElementById('m-away-spread').style.color = d.awayColor;
  document.getElementById('m-home-spread').textContent = d.homeSpread;
  document.getElementById('m-home-spread').style.color = d.homeColor;
  document.getElementById('m-away-prob').textContent = pct(d.awayProb);
  document.getElementById('m-home-prob').textContent = pct(d.homeProb);
  var dateLine = d.gameday + (d.total ? '  ·  ' + d.total : '');
  if (d.hasScores && d.scoreLine) dateLine += '  ·  ' + d.scoreLine;
  document.getElementById('m-date').textContent = dateLine;
  document.getElementById('m-pick-row').style.background = d.pickColor;
  document.getElementById('m-pick-text').textContent = d.pick;
  document.getElementById('m-conf-badge').textContent = d.conf + ' · ' + pct(Math.max(d.homeProb, d.awayProb));
  document.getElementById('m-conf-badge').style.background = d.confColor;

  var resultRow = document.getElementById('m-result-row');
  if (d.hasResult) {
    resultRow.style.display = 'flex';
    document.getElementById('m-result-chip').textContent = d.resultLabel;
    document.getElementById('m-result-chip').style.background = d.resultColor;
    document.getElementById('m-result-text').textContent = d.actualResult;
  } else {
    resultRow.style.display = 'none';
    document.getElementById('m-result-chip').textContent = '';
    document.getElementById('m-result-text').textContent = '';
  }

  var container = document.getElementById('m-stats');
  container.innerHTML = '';
  if (!d.stats || d.stats.length === 0) {
    container.innerHTML = '<div class="no-stats">Detailed stats not available for this game.</div>';
  } else {
    d.stats.forEach(function(s) {
      var hv = s.home, av = s.away;
      var both = hv !== null && av !== null;
      var hBetter = s.homeBetter;
      var hAbs = hv !== null ? Math.abs(hv) : 0;
      var aAbs = av !== null ? Math.abs(av) : 0;
      var total = hAbs + aAbs || 1;
      var hPct = (hAbs / total * 100).toFixed(1);
      var aPct = (aAbs / total * 100).toFixed(1);
      var dec = (s.label.indexOf('EPA') > -1 || s.label.indexOf('Rate') > -1) ? 3 : 1;
      var hStr = hv !== null ? fmt(hv, dec) : 'N/A';
      var aStr = av !== null ? fmt(av, dec) : 'N/A';
      var hWin = both && hBetter;
      var aWin = both && !hBetter;

      var row = document.createElement('div');
      row.className = 'stat-row';
      row.innerHTML =
        '<div class="stat-row-label">' + s.label + '</div>' +
        '<div class="stat-values">' +
          '<div class="stat-val ' + (aWin ? 'winner' : 'loser') + '" style="' + (aWin ? 'color:' + s.awayColor : '') + '">' + aStr + '</div>' +
          '<div class="stat-bar-wrap">' +
            '<div style="width:' + aPct + '%;background:' + s.awayColor + ';height:100%;opacity:' + (both && !aWin ? 0.3 : 1) + ';border-radius:99px 0 0 99px;"></div>' +
            '<div style="width:' + hPct + '%;background:' + s.homeColor + ';height:100%;opacity:' + (both && !hWin ? 0.3 : 1) + ';border-radius:0 99px 99px 0;"></div>' +
          '</div>' +
          '<div class="stat-val ' + (hWin ? 'winner' : 'loser') + '" style="' + (hWin ? 'color:' + s.homeColor : '') + '">' + hStr + '</div>' +
        '</div>' +
        '<div class="stat-row-footer"><span>' + (s.awayTeam || d.away) + ' (away)</span><span>' + (s.homeTeam || d.home) + ' (home)</span></div>';
      container.appendChild(row);
    });
  }
  document.getElementById('overlay').classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeModal() {
  document.getElementById('overlay').classList.remove('open');
  document.body.style.overflow = '';
}
function filterWeek() {
  var sel = document.getElementById('week-filter');
  if (!sel) return;
  var val = sel.value;
  var isOuMode = document.body.classList.contains('ou-mode');
  var correctAttr = isOuMode ? 'data-correct-ou' : 'data-correct';
  var highConfAttr = isOuMode ? 'data-high-conf-ou' : 'data-high-conf';
  var cards = document.querySelectorAll('.card');
  var visible = 0;
  var visibleCorrect = 0;
  var visibleHighConf = 0;
  var visibleWithResult = 0;
  
  cards.forEach(function(card) {
    var wk = card.getAttribute('data-week');
    var show = (val === 'all' || wk === val);
    card.style.display = show ? '' : 'none';
    if (show) {
      visible += 1;
      var correct = parseInt(card.getAttribute(correctAttr) || '-1');
      var highConf = parseInt(card.getAttribute(highConfAttr) || '0');
      if (highConf) visibleHighConf += 1;
      if (correct >= 0) {
        visibleWithResult += 1;
        if (correct) visibleCorrect += 1;
      }
    }
  });
  
  // Update stats
  var gamesEl = document.getElementById('stat-games');
  var highConfEl = document.getElementById('stat-high-conf');
  var accuracyEl = document.getElementById('stat-accuracy');
  
  if (gamesEl) gamesEl.textContent = visible;
  if (highConfEl) highConfEl.textContent = visibleHighConf;
  if (accuracyEl) {
    var acc = visibleWithResult > 0
      ? (visibleCorrect / visibleWithResult * 100).toFixed(1) + '%'
      : 'N/A';
    accuracyEl.textContent = acc;
  }
  var rocAucEl = document.getElementById('stat-roc-auc');
  if (rocAucEl) {
    var modeRocAttr = isOuMode ? 'data-ou-roc-auc' : 'data-spread-roc-auc';
    var modeRoc = parseFloat(rocAucEl.getAttribute(modeRocAttr) || 'nan');
    rocAucEl.textContent = isNaN(modeRoc) ? 'N/A' : modeRoc.toFixed(3);
  }
  
  var note = document.getElementById('filter-note');
  if (note) note.textContent = val === 'all' ? ('Showing all weeks · ' + visible + ' matchups') : ('Showing week ' + val + ' · ' + visible + ' matchups');
}
function toggleViewMode() {
  var isSpread = !document.body.classList.contains('ou-mode');
  var btn = document.getElementById('view-toggle-btn');
  var label = document.getElementById('view-mode-label');
  if (isSpread) {
    document.body.classList.add('ou-mode');
    btn.textContent = 'Show Spread';
    label.textContent = 'Over/Under';
  } else {
    document.body.classList.remove('ou-mode');
    btn.textContent = 'Show O/U';
    label.textContent = 'Against the Spread';
  }
  filterWeek();
}
document.addEventListener('DOMContentLoaded', function() {
  filterWeek();
});
document.addEventListener('keydown', function(e) { if (e.key === 'Escape') closeModal(); });
"""

# ── Modal skeleton ────────────────────────────────────────────────────────────

_MODAL_HTML = """
<div class="overlay" id="overlay" onclick="if(event.target===this)closeModal()">
  <div class="modal" role="dialog" aria-modal="true">
    <div class="modal-header">
      <div class="modal-teams">
        <div class="modal-team-block">
          <img class="modal-logo" id="m-away-logo" src="" alt="">
          <div class="modal-team-name" id="m-away-name"></div>
          <div class="modal-spread" id="m-away-spread"></div>
          <div style="font-size:.72rem;color:#94a3b8" id="m-away-prob"></div>
        </div>
        <div class="modal-vs">@</div>
        <div class="modal-team-block">
          <img class="modal-logo" id="m-home-logo" src="" alt="">
          <div class="modal-team-name" id="m-home-name"></div>
          <div class="modal-spread" id="m-home-spread"></div>
          <div style="font-size:.72rem;color:#94a3b8" id="m-home-prob"></div>
        </div>
      </div>
      <button class="modal-close" onclick="closeModal()" aria-label="Close">&#x2715;</button>
    </div>
    <div style="padding:8px 24px 0;font-size:.72rem;color:#64748b" id="m-date"></div>
    <div class="modal-pick-row" id="m-pick-row">
      <span class="modal-pick-label">Pick</span>
      <span class="modal-pick-text" id="m-pick-text"></span>
      <span class="conf-badge" id="m-conf-badge"></span>
    </div>
    <div class="modal-result-row" id="m-result-row">
      <span class="modal-pick-label">Outcome</span>
      <span class="result-chip" id="m-result-chip"></span>
      <span class="result-text" id="m-result-text"></span>
    </div>
    <div class="modal-body">
      <div class="section-title">Team Comparison &mdash; Rolling Form</div>
      <div class="stat-compare" id="m-stats"></div>
    </div>
  </div>
</div>
"""

# ── Main export ───────────────────────────────────────────────────────────────


def generate_weekly_html(
    picks: pd.DataFrame,
    season: int,
  week: int | None,
    model_metrics: dict,
    model_metrics_ou: dict | None,
    output_path: Path,
) -> None:
    spread_accuracy = model_metrics.get("accuracy", float("nan"))
    ou_accuracy = model_metrics_ou.get("accuracy", float("nan")) if model_metrics_ou else float("nan")
    spread_roc_auc = model_metrics.get("roc_auc", float("nan"))
    ou_roc_auc = model_metrics_ou.get("roc_auc", float("nan")) if model_metrics_ou else float("nan")
    train_games = int(model_metrics.get("train_games", 0))
    n_games = len(picks)
    high_conf_spread = int(
        (picks["pred_prob_home_cover"].apply(lambda p: max(p, 1 - p)) >= 0.70).sum()
    )
    high_conf_ou = int(
        (picks["pred_prob_total_over"].apply(lambda p: max(p, 1 - p)) >= 0.70).sum()
    ) if "pred_prob_total_over" in picks.columns else 0

    def _fmt_pct(v: float) -> str:
        return f"{v:.1%}" if pd.notna(v) else "N/A"

    def _fmt_roc(v: float) -> str:
        return f"{v:.3f}" if pd.notna(v) else "N/A"

    stats_row = f"""
<div class="stats-row">
  <div class="stat-box"><div class="label">Games</div><div class="value" id="stat-games">{n_games}</div></div>
  <div class="stat-box"><div class="label">High Conf.</div><div class="value" id="stat-high-conf" data-spread-high-conf="{high_conf_spread}" data-ou-high-conf="{high_conf_ou}">{high_conf_spread}</div></div>
  <div class="stat-box"><div class="label">Accuracy</div><div class="value" id="stat-accuracy" data-spread-acc="{spread_accuracy}" data-ou-acc="{ou_accuracy}">{_fmt_pct(spread_accuracy)}</div></div>
  <div class="stat-box"><div class="label">ROC AUC</div><div class="value" id="stat-roc-auc" data-spread-roc-auc="{spread_roc_auc}" data-ou-roc-auc="{ou_roc_auc}">{_fmt_roc(spread_roc_auc)}</div></div>
  <div class="stat-box"><div class="label">Train Games</div><div class="value">{train_games:,}</div></div>
</div>"""

    week_values = sorted({int(float(w)) for w in picks["week"].dropna().tolist()}) if "week" in picks.columns else []
    week_options = "".join([f'<option value="{w}">Week {w}</option>' for w in week_values])
    filter_html = f"""
<div class="filter-row">
  <div>
    <div class="filter-label">Week Filter</div>
    <select id="week-filter" class="filter-select" onchange="filterWeek()">
      <option value="all" selected>All Weeks</option>
      {week_options}
    </select>
  </div>
  <div id="filter-note" class="filter-note">Showing all weeks · {n_games} matchups</div>
</div>"""

    cards = "".join(_card_html(i, row) for i, (_, row) in enumerate(picks.iterrows()))
    title_scope = f"Week {week}" if week is not None else "All Weeks"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>NFL Predictor &middot; {season} {title_scope}</title>
  <style>{_CSS}</style>
</head>
<body>
<header>
  <div>
    <h1>&#127944; NFL Predictor</h1>
    <div class="subtitle"><span id="view-mode-label">Against the Spread</span> &middot; {season} Season</div>
  </div>
  <button id="view-toggle-btn" class="view-toggle-btn" onclick="toggleViewMode()">Show O/U</button>
  <div class="badge">EPA &middot; Pace &middot; Rest &middot; ATS Form</div>
</header>
<main>
  {filter_html}
  {stats_row}
  <div class="grid">{cards}</div>
</main>
{_MODAL_HTML}
<footer>Generated by NFL Predictor &middot; Data via nflverse &middot; {train_games:,} training games</footer>
<script>{_JS}</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Saved HTML report to: {output_path}")

