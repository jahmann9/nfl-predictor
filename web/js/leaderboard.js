var supabaseClient;
var allRows = [];

function resultBadgeHtml(result) {
  if (result === "hit") {
    return "<span class='lb-result-badge lb-result-hit'>✅ Hit</span>";
  }
  if (result === "miss") {
    return "<span class='lb-result-badge lb-result-miss'>❌ Miss</span>";
  }
  if (result === "push") {
    return "<span class='lb-result-badge lb-result-push'>🟨 Push</span>";
  }
  return "<span class='lb-result-badge lb-result-pending'>Pending</span>";
}

function weekOptions() {
  var sel = document.getElementById("week-select");
  sel.innerHTML = "<option value='all'>All</option>";
  for (var i = 1; i <= 22; i += 1) {
    var opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = "Week " + i;
    sel.appendChild(opt);
  }
}

function filterRows() {
  var season = document.getElementById("season-select").value;
  var week = document.getElementById("week-select").value;
  return allRows.filter(function (r) {
    var seasonOk = season === "all" || String(r.season) === season;
    var weekOk = week === "all" || String(r.week) === week;
    return seasonOk && weekOk;
  });
}

function renderStats(rows) {
  var stats = {};
  window.PEOPLE.forEach(function (name) {
    stats[name] = { hits: 0, misses: 0, pushes: 0, pending: 0, total: 0 };
  });

  rows.forEach(function (r) {
    if (!stats[r.person]) return;
    stats[r.person].total += 1;
    if (r.result === "hit") stats[r.person].hits += 1;
    else if (r.result === "miss") stats[r.person].misses += 1;
    else if (r.result === "push") stats[r.person].pushes += 1;
    else stats[r.person].pending += 1;
  });

  var grid = document.getElementById("stats-grid");
  grid.innerHTML = "";

  function computeCurrentStreak(person, targetResult) {
    var personRows = rows
      .filter(function (r) { return r.person === person; })
      .slice()
      .sort(function (a, b) {
        if (a.season !== b.season) return a.season - b.season;
        if (a.week !== b.week) return a.week - b.week;
        return new Date(a.updated_at).getTime() - new Date(b.updated_at).getTime();
      })
      .filter(function (r) { return r.result === "hit" || r.result === "miss"; });

    var streak = 0;
    for (var i = personRows.length - 1; i >= 0; i -= 1) {
      if (personRows[i].result === targetResult) {
        streak += 1;
      } else {
        break;
      }
    }
    return streak;
  }

  var sortedPeople = window.PEOPLE.slice().sort(function (a, b) {
    var aStats = stats[a] || { hits: 0, misses: 0, pushes: 0, pending: 0, total: 0 };
    var bStats = stats[b] || { hits: 0, misses: 0, pushes: 0, pending: 0, total: 0 };

    if (bStats.hits !== aStats.hits) return bStats.hits - aStats.hits;
    if (aStats.misses !== bStats.misses) return aStats.misses - bStats.misses;
    if (bStats.total !== aStats.total) return bStats.total - aStats.total;
    return a.localeCompare(b);
  });

  sortedPeople.forEach(function (name) {
    var s = stats[name];
    var settled = s.hits + s.misses;
    var pctNumber = settled > 0 ? (s.hits / settled) * 100 : null;
    var pct = pctNumber !== null ? pctNumber.toFixed(1) + "%" : "N/A";
    var winStreak = computeCurrentStreak(name, "hit");
    var lossStreak = computeCurrentStreak(name, "miss");

    var barWidth = pctNumber !== null ? Math.max(0, Math.min(100, pctNumber)) : 0;
    var rateBarHtml =
      "<div class='lb-rate-wrap'>" +
        "<div class='lb-rate-header'>" +
          "<span>Hit Rate</span>" +
          "<strong>" + pct + "</strong>" +
        "</div>" +
        "<div class='lb-rate-track'>" +
          "<div class='lb-rate-fill' style='width:" + barWidth + "%'></div>" +
        "</div>" +
      "</div>";

    var chipsHtml =
      "<div class='lb-chip-row'>" +
        "<span class='lb-chip lb-chip-win'>W Streak " + winStreak + "</span>" +
        "<span class='lb-chip lb-chip-loss'>L Streak " + lossStreak + "</span>" +
        "<span class='lb-chip lb-chip-push'>Push " + s.pushes + "</span>" +
        "<span class='lb-chip lb-chip-pending'>Pending " + s.pending + "</span>" +
        "<span class='lb-chip lb-chip-picks'>Picks " + s.total + "</span>" +
      "</div>";

    var div = document.createElement("div");
    div.className = "stat";
    div.innerHTML =
      "<div class='name'>" + name + "</div>" +
      "<div class='value'>" + s.hits + "-" + s.misses + "</div>" +
      rateBarHtml +
      chipsHtml;
    grid.appendChild(div);
  });
}

function renderSummary(rows) {
  var tbody = document.getElementById("summary-rows");
  tbody.innerHTML = "";

  rows
    .slice()
    .sort(function (a, b) {
      if (a.season !== b.season) return b.season - a.season;
      if (a.week !== b.week) return b.week - a.week;
      return a.person.localeCompare(b.person);
    })
    .forEach(function (r) {
      var tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + r.season + "</td>" +
        "<td>" + r.week + "</td>" +
        "<td>" + r.person + "</td>" +
        "<td>" + (r.pick_type === "ou" ? "O/U" : "Spread") + "</td>" +
        "<td>" + (r.pick_text || "") + "</td>" +
        "<td>" + resultBadgeHtml(r.result) + "</td>" +
        "<td>" + new Date(r.updated_at).toLocaleString() + "</td>";
      tbody.appendChild(tr);
    });
}

function rerender() {
  var rows = filterRows();
  renderStats(rows);
  renderSummary(rows);
}

async function loadData() {
  var res = await supabaseClient.from("weekly_picks").select("*");
  if (res.error) {
    document.getElementById("config-warning").classList.remove("hidden");
    document.getElementById("config-warning").textContent = "Load error: " + res.error.message;
    return;
  }
  allRows = res.data || [];
  rerender();
}

async function boot() {
  try {
    supabaseClient = window.createSupabaseClient();
  } catch (e) {
    var warning = document.getElementById("config-warning");
    warning.classList.remove("hidden");
    warning.textContent = e.message;
    return;
  }

  weekOptions();
  document.getElementById("refresh-btn").addEventListener("click", loadData);
  document.getElementById("season-select").addEventListener("change", rerender);
  document.getElementById("week-select").addEventListener("change", rerender);

  await loadData();

  supabaseClient
    .channel("weekly_picks_changes")
    .on(
      "postgres_changes",
      { event: "*", schema: "public", table: "weekly_picks" },
      function () {
        loadData();
      }
    )
    .subscribe();
}

boot();
