/* Claude knowledge base — frontend for the FastAPI backend.
   Search/semantic ranking happen server-side (/api/*); this file renders results,
   draws the D3 graph, and listens to /api/stream (SSE) for live freshness. */
(function () {
  "use strict";

  const TYPE_COLOR = {
    discovery: "#60a5fa",
    feature: "#5eead4",
    bugfix: "#fb7185",
    change: "#fbbf24",
    decision: "#a78bfa",
    refactor: "#7c89a3",
    summary: "#a78bfa",
  };
  const SESSION_COLOR = "#5eead4",
    FILE_COLOR = "#7c89a3",
    CONCEPT_COLOR = "#fbbf24";

  const state = {
    query: "",
    project: "all",
    kind: "all",
    mode: "keyword",
    selectedRecId: null,
    selectedSessId: null,
  };
  let meta = null;
  const sessByContentId = new Map();
  let currentRecs = [];
  const recById = new Map();
  let queryToken = 0;

  const $ = (s) => document.querySelector(s);
  const el = (tag, cls, txt) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (txt != null) e.textContent = txt;
    return e;
  };
  const baseName = (p) => (p || "").split("/").pop();
  const truncate = (s, n) =>
    s && s.length > n ? s.slice(0, n - 1) + "…" : s || "";

  // Stopwords for the per-observation one-word label heuristic.
  const STOP = new Set([
    "the",
    "and",
    "that",
    "for",
    "with",
    "this",
    "from",
    "they",
    "have",
    "been",
    "about",
    "into",
    "your",
    "what",
    "when",
    "where",
    "which",
    "while",
    "will",
    "there",
    "only",
    "also",
    "some",
    "most",
    "just",
    "like",
    "then",
    "than",
    "because",
    "such",
    "these",
    "those",
    "much",
    "more",
    "were",
    "does",
    "session",
    "task",
    "tasks",
    "work",
    "want",
    "need",
    "make",
    "using",
    "into",
    "over",
    "under",
    "across",
  ]);

  // Pick the most distinctive single token from a record's title.
  // Prefers UPPER_SNAKE acronyms (AWAITING_CHECKIN), then InitialCap, then longest.
  function topToken(text) {
    if (!text) return null;
    const toks = (text.match(/[A-Za-z_][A-Za-z0-9_]{3,}/g) || []).filter(
      (t) => !STOP.has(t.toLowerCase()) && !/^\d+$/.test(t),
    );
    if (!toks.length) return null;
    const score = (t) =>
      t.length +
      (/^[A-Z][A-Z0-9_]+$/.test(t) ? 12 : 0) +
      (/^[A-Z]/.test(t) ? 3 : 0);
    toks.sort((a, b) => score(b) - score(a));
    return toks[0].slice(0, 20);
  }

  // Single-word label for an observation/summary record — defers to a server-provided
  // r.label if present, otherwise picks a distinctive token from the title.
  const recordLabel = (r) =>
    r.label || topToken(r.title) || truncate(r.title, 14);

  async function api(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${path} → ${r.status}`);
    return r.json();
  }

  // ===================== INIT =====================
  async function init() {
    try {
      meta = await api("/api/meta");
    } catch (e) {
      document.body.innerHTML =
        "<p style='padding:40px'>Backend not reachable. Start it with <code>python app.py</code>.</p>";
      return;
    }
    setCounts();

    const projSel = $("#proj-filter");
    ["all", ...meta.projects].forEach((p) => {
      const o = el("option", null, p === "all" ? "all projects" : p);
      o.value = p;
      projSel.appendChild(o);
    });
    projSel.value = state.project;
    projSel.addEventListener("change", () => {
      state.project = projSel.value;
      clearSelection();
      refresh();
    });

    const search = $("#search");
    let t;
    search.addEventListener("input", () => {
      clearTimeout(t);
      t = setTimeout(
        () => {
          state.query = search.value.trim();
          state.selectedSessId = null;
          refresh();
        },
        state.mode === "semantic" ? 300 : 120,
      );
    });

    document.querySelectorAll(".chip[data-kind]").forEach((c) =>
      c.addEventListener("click", () => {
        document
          .querySelectorAll(".chip[data-kind]")
          .forEach((x) => x.classList.remove("active"));
        c.classList.add("active");
        state.kind = c.dataset.kind;
        refresh();
      }),
    );

    document.querySelectorAll(".chip[data-mode]").forEach((c) =>
      c.addEventListener("click", () => {
        document
          .querySelectorAll(".chip[data-mode]")
          .forEach((x) => x.classList.remove("active"));
        c.classList.add("active");
        state.mode = c.dataset.mode;
        search.placeholder =
          state.mode === "semantic"
            ? "Describe what you're looking for — searches by meaning…"
            : "Search observations, summaries, facts, concepts, files…";
        refresh();
      }),
    );

    $("#reset-graph").addEventListener("click", () => {
      clearSelection();
      refresh();
    });

    try {
      const s = await api("/api/sessions");
      s.sessions.forEach((x) => sessByContentId.set(x.id, x));
    } catch (e) {
      /* non-fatal */
    }

    buildLegend();
    connectSSE();
    await refresh();
  }

  function setCounts() {
    $("#kb-counts").textContent =
      `${meta.counts.observations} observations · ${meta.counts.summaries} summaries · ${meta.counts.sessions} sessions · live`;
  }
  function setStatus(msg) {
    const s = $("#sem-status");
    if (s) s.textContent = msg;
  }

  // ===================== DATA =====================
  async function refresh() {
    const token = ++queryToken;
    if (state.mode === "semantic" && state.query) setStatus("searching…");
    const params = new URLSearchParams({
      q: state.query,
      project: state.project,
      kind: state.kind,
      mode: state.mode,
      limit: "250",
    });
    if (state.selectedSessId && !state.query)
      params.set("session", state.selectedSessId);

    let data;
    try {
      data = await api("/api/search?" + params.toString());
    } catch (e) {
      setStatus("backend error");
      return;
    }
    if (token !== queryToken) return; // superseded

    currentRecs = data.results;
    currentRecs.forEach((r) => recById.set(r.id, r));
    setStatus(
      data.mode === "semantic"
        ? `semantic · top ${data.total}`
        : data.mode === "indexing"
          ? "building semantic index… (keyword for now)"
          : "",
    );
    renderResults(currentRecs);
    await renderGraphView();
    renderDetail();
  }

  // ===================== RESULTS =====================
  function renderResults(recs) {
    const list = $("#results");
    list.innerHTML = "";
    $("#results-count").textContent = recs.length;
    if (!recs.length) {
      list.appendChild(
        el(
          "div",
          "empty",
          "No matches. Try a different term or widen the project filter.",
        ),
      );
      return;
    }
    const frag = document.createDocumentFragment();
    recs.forEach((r) => {
      const row = el("div", "result");
      row.dataset.id = r.id;
      if (r.id === state.selectedRecId) row.classList.add("sel");
      const head = el("div", "result-head");
      head.appendChild(el("span", "badge " + r.type, r.type));
      head.appendChild(el("span", "result-title", r.title));
      row.appendChild(head);
      if (r.subtitle) row.appendChild(el("div", "result-snip", r.subtitle));
      else if (r.text) row.appendChild(el("div", "result-snip", r.text));
      row.appendChild(
        el(
          "div",
          "result-meta",
          `${r.project} · ${(r.date || "").slice(0, 10)}` +
            (r.files.length ? ` · ${r.files.length} file(s)` : ""),
        ),
      );
      row.addEventListener("click", () => selectRecord(r.id));
      frag.appendChild(row);
    });
    list.appendChild(frag);
  }

  // ===================== DETAIL =====================
  function renderDetail() {
    const d = $("#detail");
    const r = state.selectedRecId && recById.get(state.selectedRecId);
    if (!r) {
      // A session is selected (drill-down) but no specific record — show the session card.
      const s =
        state.selectedSessId && sessByContentId.get(state.selectedSessId);
      if (s) return renderSessionDetail(d, s);
      d.className = "detail placeholder";
      d.textContent =
        "Select a result or a node to see its full content, files, concepts, and how to resume its session.";
      return;
    }
    d.className = "detail";
    d.innerHTML = "";
    d.appendChild(el("h3", null, r.title));
    const row = el("div", "row");
    row.appendChild(el("span", "badge " + r.type, r.type));
    row.appendChild(el("span", "cchip", r.project));
    if (r.date)
      row.appendChild(
        el("span", "cchip", r.date.slice(0, 16).replace("T", " ")),
      );
    d.appendChild(row);
    if (r.subtitle) {
      const s = el("p", "body", r.subtitle);
      s.style.fontWeight = "600";
      s.style.color = "var(--text)";
      d.appendChild(s);
    }
    if (r.text) d.appendChild(el("div", "body", r.text));
    if (r.facts && r.facts.length) {
      d.appendChild(el("div", "sub", "Facts"));
      const ul = el("ul");
      r.facts.forEach((f) => ul.appendChild(el("li", null, f)));
      d.appendChild(ul);
    }
    if (r.concepts && r.concepts.length) {
      d.appendChild(el("div", "sub", "Concepts"));
      const box = el("div");
      r.concepts.forEach((c) => {
        const chip = el("span", "cchip", c);
        chip.style.cursor = "pointer";
        chip.title = "filter by this concept";
        chip.addEventListener("click", () => {
          $("#search").value = c;
          state.query = c;
          clearSelection();
          refresh();
        });
        box.appendChild(chip);
      });
      d.appendChild(box);
    }
    if (r.files && r.files.length) {
      d.appendChild(el("div", "sub", "Files"));
      r.files.slice(0, 30).forEach((f) => d.appendChild(el("div", "fpath", f)));
    }

    const sess = sessByContentId.get(r.sessionId);
    const box = el("div", "session-box");
    box.appendChild(el("div", "sub", "Session"));
    if (sess) box.appendChild(el("div", "body", sess.title));
    if (r.sessionId) {
      box.appendChild(el("code", null, "claude --resume " + r.sessionId));
      const btn = el("button", "btn", "Copy resume command");
      btn.style.marginTop = "8px";
      btn.addEventListener("click", () => {
        copyText("claude --resume " + r.sessionId);
        btn.textContent = "Copied!";
        setTimeout(() => (btn.textContent = "Copy resume command"), 1200);
      });
      box.appendChild(btn);
      const focus = el("button", "btn", "Focus this session in graph");
      focus.style.marginTop = "8px";
      focus.style.marginLeft = "6px";
      focus.addEventListener("click", () => selectSession(r.sessionId));
      box.appendChild(focus);
    } else {
      box.appendChild(el("div", "body", "(no linked session)"));
    }
    d.appendChild(box);
  }

  // Rendered when a session anchor is selected but no record is picked yet.
  function renderSessionDetail(d, s) {
    d.className = "detail";
    d.innerHTML = "";
    d.appendChild(
      el("h3", null, s.label || truncate(s.title || "Session", 30)),
    );
    const row = el("div", "row");
    row.appendChild(el("span", "cchip", s.project || ""));
    if (s.obsCount != null)
      row.appendChild(el("span", "cchip", s.obsCount + " observations"));
    if (s.started)
      row.appendChild(
        el("span", "cchip", s.started.slice(0, 16).replace("T", " ")),
      );
    d.appendChild(row);
    if (s.summary) {
      const p = el("p", "body", s.summary);
      p.style.fontStyle = "italic";
      d.appendChild(p);
    }
    d.appendChild(el("div", "sub", "Session"));
    const box = el("div", "session-box");
    box.appendChild(el("div", "body", s.title || ""));
    if (s.id) {
      box.appendChild(el("code", null, "claude --resume " + s.id));
      const btn = el("button", "btn", "Copy resume command");
      btn.style.marginTop = "8px";
      btn.addEventListener("click", () => {
        copyText("claude --resume " + s.id);
        btn.textContent = "Copied!";
        setTimeout(() => (btn.textContent = "Copy resume command"), 1200);
      });
      box.appendChild(btn);
    }
    d.appendChild(box);
  }

  function selectRecord(id) {
    state.selectedRecId = id;
    const r = recById.get(id);
    document
      .querySelectorAll(".result")
      .forEach((x) => x.classList.toggle("sel", x.dataset.id === id));
    const sel = document.querySelector(".result.sel");
    if (sel) sel.scrollIntoView({ block: "nearest" });
    // highlight the matching node: the record itself when drilled into a session,
    // otherwise its session node on the overview map
    setActive(state.selectedSessId ? id : r ? "S:" + r.sessionId : null);
    renderDetail();
  }
  function selectSession(sessId) {
    state.selectedSessId = sessId;
    state.selectedRecId = null;
    state.query = "";
    $("#search").value = "";
    refresh();
  }
  function clearSelection() {
    state.selectedRecId = null;
    state.selectedSessId = null;
  }

  // ===================== GRAPH =====================
  let sim = null,
    gZoom = null,
    svgSel = null,
    rootG = null;

  // ---- overview: timeline with project swimlanes ----
  // Pick which sessions to render: project-filtered, narrowed to hit-sessions when a
  // query is active (sized by hit count instead of obsCount in that case).
  function timelineSessions() {
    let pool = [...sessByContentId.values()];
    if (state.project !== "all")
      pool = pool.filter((s) => s.project === state.project);
    if (state.query) {
      const hits = new Map();
      currentRecs.forEach((r) => {
        if (r.sessionId)
          hits.set(r.sessionId, (hits.get(r.sessionId) || 0) + 1);
      });
      pool = pool
        .filter((s) => hits.has(s.id))
        .map((s) => Object.assign({}, s, { hitCount: hits.get(s.id) }));
    }
    return pool;
  }

  let tZoom = null;
  function renderTimeline(sessions) {
    const wrap = $("#graph-wrap");
    const W = wrap.clientWidth || 1000,
      H = wrap.clientHeight || 500;

    if (!svgSel) {
      svgSel = d3.select("#graph");
      rootG = svgSel.append("g");
    }
    // Reset any prior zoom transform left by the force-graph view, and clear handlers.
    rootG.attr("transform", null);
    svgSel.on(".zoom", null);
    svgSel.attr("viewBox", `0 0 ${W} ${H}`);
    rootG.selectAll("*").remove();

    // Parse dates; drop sessions with no usable timestamp
    const data = sessions
      .map((s) =>
        Object.assign({}, s, { _date: s.started ? new Date(s.started) : null }),
      )
      .filter((s) => s._date && !isNaN(s._date));
    $("#graph-empty").style.display = data.length ? "none" : "flex";
    if (!data.length) return;

    const LEFT = 110,
      TOP = 28,
      RIGHT = 24,
      BOTTOM = 14;
    const innerW = Math.max(40, W - LEFT - RIGHT);
    const innerH = Math.max(40, H - TOP - BOTTOM);

    // Project lanes, sorted by total activity desc
    const activity = new Map();
    data.forEach((s) => {
      const p = s.project || "(none)";
      activity.set(p, (activity.get(p) || 0) + (s.obsCount || 1));
    });
    const projects = [...activity.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([p]) => p);

    const xExt = d3.extent(data, (s) => s._date);
    const xMin = d3.timeWeek.offset(d3.timeMonth.floor(xExt[0]), -1);
    const xMax = d3.timeDay.offset(xExt[1], 2);
    const xScale = d3
      .scaleTime()
      .domain([xMin, xMax])
      .range([LEFT, LEFT + innerW]);
    const yScale = d3
      .scaleBand()
      .domain(projects)
      .range([TOP, TOP + innerH])
      .padding(0.25);

    const sizeOf = (s) =>
      Math.min(
        14,
        4 + Math.sqrt((s.hitCount != null ? s.hitCount * 3 : s.obsCount) || 1),
      );
    // deterministic vertical jitter from session id so a node stays in place across renders
    const jitter = (sid) => {
      let h = 0;
      for (const c of sid || "") h = ((h << 5) - h + c.charCodeAt(0)) | 0;
      return (((h >>> 0) % 1000) / 1000 - 0.5) * yScale.bandwidth() * 0.55;
    };

    // lane backgrounds + separators
    rootG
      .append("g")
      .selectAll("rect")
      .data(projects)
      .enter()
      .append("rect")
      .attr("class", "t-lane")
      .attr("x", LEFT)
      .attr("y", (p) => yScale(p))
      .attr("width", innerW)
      .attr("height", yScale.bandwidth());

    // lane labels (project names) on the left
    rootG
      .append("g")
      .selectAll("text")
      .data(projects)
      .enter()
      .append("text")
      .attr("class", "t-lane-label")
      .attr("x", LEFT - 10)
      .attr("y", (p) => yScale(p) + yScale.bandwidth() / 2)
      .attr("text-anchor", "end")
      .attr("dominant-baseline", "central")
      .text((p) => truncate(p.replace(/^zatlas-/, ""), 16));

    // top time axis
    const xAxisFn = (scale) =>
      d3
        .axisTop(scale)
        .ticks(d3.timeMonth.every(1))
        .tickFormat(d3.timeFormat("%b %y"));
    const axisG = rootG
      .append("g")
      .attr("class", "t-axis")
      .attr("transform", `translate(0, ${TOP})`)
      .call(xAxisFn(xScale));
    axisG
      .selectAll(".tick line")
      .attr("y2", innerH)
      .attr("stroke", "var(--border2)")
      .attr("stroke-opacity", 0.18);
    axisG.select(".domain").remove();

    // sessions
    const node = rootG
      .append("g")
      .selectAll("g")
      .data(data, (d) => d.id)
      .enter()
      .append("g")
      .attr("class", "gnode")
      .attr("data-id", (d) => d.id);
    node
      .append("circle")
      .attr("class", "t-dot")
      .attr(
        "cy",
        (d) => yScale(d.project) + yScale.bandwidth() / 2 + jitter(d.id),
      )
      .attr("r", sizeOf)
      .attr("fill", SESSION_COLOR)
      .attr("stroke", SESSION_COLOR)
      .attr("stroke-opacity", 0.5)
      .style(
        "filter",
        (d) =>
          `drop-shadow(0 0 ${Math.max(2, sizeOf(d) * 0.5)}px ${SESSION_COLOR}55)`,
      );
    node
      .append("text")
      .attr("class", "t-label")
      .attr(
        "y",
        (d) => yScale(d.project) + yScale.bandwidth() / 2 + jitter(d.id),
      )
      .attr("dy", "0.32em")
      .text((d) => d.label || truncate(d.title || "session", 16));
    node.append("title").text((d) => d.title || "");
    // Native title attribute is the cheapest tooltip; renders on hover in every browser.

    const placeX = (scale) => {
      node.select("circle").attr("cx", (d) => scale(d._date));
      node.select("text").attr("x", (d) => scale(d._date) + sizeOf(d) + 4);
    };
    placeX(xScale);

    node
      .style("cursor", "pointer")
      .on("click", (e, d) => selectSession(d.id))
      .on("mouseenter", function (e, d) {
        d3.select(this).raise();
        rootG
          .selectAll(".gnode")
          .classed("dim", (n) => n.project !== d.project);
      })
      .on("mouseleave", () => rootG.selectAll(".gnode").classed("dim", false));

    // pan / zoom on the time axis
    tZoom = d3
      .zoom()
      .scaleExtent([0.5, 24])
      .translateExtent([
        [0, 0],
        [W, H],
      ])
      .extent([
        [LEFT, TOP],
        [LEFT + innerW, TOP + innerH],
      ])
      .on("zoom", (ev) => {
        const nx = ev.transform.rescaleX(xScale);
        axisG.call(xAxisFn(nx));
        axisG
          .selectAll(".tick line")
          .attr("y2", innerH)
          .attr("stroke", "var(--border2)")
          .attr("stroke-opacity", 0.18);
        axisG.select(".domain").remove();
        placeX(nx);
      });
    svgSel.call(tZoom);
    svgSel.transition().duration(0).call(tZoom.transform, d3.zoomIdentity);
  }

  // Drill-down: one session expanded into its observations + the files they share.
  function buildSubgraph(recs) {
    const focus = recs.slice(0, 40);
    const nodes = new Map(),
      links = [];
    const addNode = (id, o) => {
      if (!nodes.has(id)) nodes.set(id, Object.assign({ id }, o));
      return nodes.get(id);
    };
    new Set(focus.map((r) => r.sessionId).filter(Boolean)).forEach((sid) => {
      const s = sessByContentId.get(sid);
      addNode("S:" + sid, {
        kind: "session",
        label: (s && s.label) || (s ? truncate(s.title, 16) : "session"),
        color: SESSION_COLOR,
        r: 12,
        ref: sid,
      });
    });
    focus.forEach((r) => {
      addNode(r.id, {
        kind: r.kind,
        label: recordLabel(r),
        color: TYPE_COLOR[r.type] || "#6b7280",
        r: 5.5,
        ref: r.id,
      });
      if (r.sessionId && nodes.has("S:" + r.sessionId))
        links.push({ source: "S:" + r.sessionId, target: r.id, kind: "owns" });
    });
    addSharedConnectors(
      focus,
      "files",
      "F:",
      FILE_COLOR,
      baseName,
      nodes,
      links,
      addNode,
      18,
    );
    return { nodes: [...nodes.values()], links };
  }

  function addSharedConnectors(
    focus,
    field,
    prefix,
    color,
    labelFn,
    nodes,
    links,
    addNode,
    cap,
  ) {
    const usage = new Map();
    focus.forEach((r) =>
      (r[field] || []).forEach((v) => {
        if (!usage.has(v)) usage.set(v, []);
        usage.get(v).push(r.id);
      }),
    );
    [...usage.entries()]
      .filter(([, rs]) => rs.length >= 2)
      .sort((a, b) => b[1].length - a[1].length)
      .slice(0, cap)
      .forEach(([v, rs]) => {
        const id = prefix + v;
        addNode(id, {
          kind: field === "files" ? "file" : "concept",
          label: truncate(labelFn(v), 22),
          color,
          r: 4,
        });
        rs.forEach((rid) =>
          links.push({ source: id, target: rid, kind: field }),
        );
      });
  }

  // Decide which graph to show: drill-down (a session selected), search overview
  // (a query active), or the default session map.
  function renderGraphView() {
    const resetBtn = $("#reset-graph");
    if (state.selectedSessId) {
      if (resetBtn) resetBtn.textContent = "← all sessions";
      return renderForce(buildSubgraph(currentRecs), { labelAll: true });
    }
    if (resetBtn) resetBtn.textContent = "Reset view";
    renderTimeline(timelineSessions());
  }

  // Remembered between renderForce() and setActive() so the active-node highlight
  // doesn't accidentally hide all the non-session labels in drill-down.
  let currentLabelAll = false;

  function renderForce(g, opts) {
    opts = opts || {};
    currentLabelAll = !!opts.labelAll;
    const wrap = $("#graph-wrap");
    const W = wrap.clientWidth || 600,
      H = wrap.clientHeight || 500;
    $("#graph-empty").style.display = g.nodes.length ? "none" : "flex";

    if (!svgSel) {
      svgSel = d3.select("#graph");
      rootG = svgSel.append("g");
      gZoom = d3
        .zoom()
        .scaleExtent([0.2, 4])
        .on("zoom", (e) => rootG.attr("transform", e.transform));
      svgSel.call(gZoom);
    }
    svgSel.attr("viewBox", `0 0 ${W} ${H}`);
    rootG.selectAll("*").remove();
    if (!g.nodes.length) {
      if (sim) sim.stop();
      return;
    }

    const link = rootG
      .append("g")
      .selectAll("line")
      .data(g.links)
      .enter()
      .append("line")
      .attr("class", "glink")
      .attr("stroke-width", (d) =>
        d.kind === "shared"
          ? Math.min(4, 1 + (d.weight || 1) * 0.5)
          : d.kind === "owns"
            ? 1.3
            : 0.8,
      )
      .attr("stroke-dasharray", (d) =>
        d.kind === "files" || d.kind === "concepts" ? "3,3" : null,
      );

    const node = rootG
      .append("g")
      .selectAll("g")
      .data(g.nodes)
      .enter()
      .append("g")
      .attr("class", "gnode")
      .attr("data-id", (d) => d.ref || d.id);
    node
      .append("circle")
      .attr("r", (d) => d.r)
      .attr("fill", (d) => d.color)
      .attr("stroke", (d) => d.color)
      .attr("stroke-opacity", 0.45)
      .style(
        "filter",
        (d) => `drop-shadow(0 0 ${Math.max(2, d.r * 0.45)}px ${d.color}55)`,
      );
    node
      .append("text")
      .attr("x", (d) => d.r + 4)
      .attr("dy", "0.32em")
      // overview labels every (session) node; drill-down labels only the session
      // anchor by default and reveals the rest on hover/selection via setActive()
      .style("display", (d) =>
        opts.labelAll || d.kind === "session" ? null : "none",
      )
      .text((d) => d.label);
    node
      .style("cursor", "pointer")
      .on("click", (e, d) => onNodeClick(d))
      .on("mouseenter", (e, d) => setActive(d.id))
      .on("mouseleave", () => setActive(state.selectedRecId));
    node.call(
      d3
        .drag()
        .on("start", (e, d) => {
          if (!e.active) sim.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on("drag", (e, d) => {
          d.fx = e.x;
          d.fy = e.y;
        })
        .on("end", (e, d) => {
          if (!e.active) sim.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }),
    );

    sim = d3
      .forceSimulation(g.nodes)
      .force(
        "link",
        d3
          .forceLink(g.links)
          .id((d) => d.id)
          .distance((d) =>
            d.kind === "owns" ? 55 : d.kind === "shared" ? 130 : 90,
          )
          .strength(0.3),
      )
      .force("charge", d3.forceManyBody().strength(-340))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("x", d3.forceX(W / 2).strength(0.05))
      .force("y", d3.forceY(H / 2).strength(0.05))
      .force(
        "collide",
        d3.forceCollide().radius((d) => d.r + 18),
      )
      .on("tick", () => {
        link
          .attr("x1", (d) => d.source.x)
          .attr("y1", (d) => d.source.y)
          .attr("x2", (d) => d.target.x)
          .attr("y2", (d) => d.target.y);
        node.attr("transform", (d) => `translate(${d.x},${d.y})`);
      });
    svgSel.transition().duration(300).call(gZoom.transform, d3.zoomIdentity);
    setActive(state.selectedRecId);
  }

  function onNodeClick(d) {
    if (d.kind === "session") return selectSession(d.ref);
    if (d.kind === "file") {
      const v = baseName(d.id.slice(2));
      $("#search").value = v;
      state.query = v;
      clearSelection();
      return refresh();
    }
    if (d.kind === "concept") {
      const v = d.id.slice(2);
      $("#search").value = v;
      state.query = v;
      clearSelection();
      return refresh();
    }
    selectRecord(d.ref);
  }

  // Focus a node by its graph id (record id for obs/sum, "S:"+sid for sessions, etc.).
  // Dims everything outside its neighborhood and reveals labels for it + neighbors.
  // Pass null/undefined to clear (back to: only session labels shown).
  function setActive(nodeId) {
    if (!rootG) return;
    const linkId = (e) => (typeof e === "object" ? e.id : e);
    // if the target isn't in the current graph, clear rather than blanking everything
    if (
      nodeId &&
      !rootG
        .selectAll(".gnode")
        .data()
        .some((d) => d.id === nodeId)
    )
      nodeId = null;
    const neighbors = new Set();
    if (nodeId) {
      neighbors.add(nodeId);
      rootG.selectAll(".glink").each(function (d) {
        const s = linkId(d.source),
          t = linkId(d.target);
        if (s === nodeId || t === nodeId) {
          neighbors.add(s);
          neighbors.add(t);
        }
      });
    }
    rootG
      .selectAll(".gnode")
      .classed("dim", (d) => (nodeId ? !neighbors.has(d.id) : false))
      .classed("sel", (d) => d.id === nodeId);
    rootG
      .selectAll(".gnode text")
      .style("display", (d) =>
        currentLabelAll ||
        d.kind === "session" ||
        (nodeId && neighbors.has(d.id))
          ? null
          : "none",
      );
    rootG.selectAll(".glink").classed("dim", (d) => {
      if (!nodeId) return false;
      const s = linkId(d.source),
        t = linkId(d.target);
      return !(s === nodeId || t === nodeId);
    });
  }

  function buildLegend() {
    const items = [
      ["session", SESSION_COLOR],
      ["discovery", TYPE_COLOR.discovery],
      ["feature", TYPE_COLOR.feature],
      ["bugfix", TYPE_COLOR.bugfix],
      ["change", TYPE_COLOR.change],
      ["decision", TYPE_COLOR.decision],
      ["summary", TYPE_COLOR.summary],
      ["file", FILE_COLOR],
      ["concept", CONCEPT_COLOR],
    ];
    const leg = $("#legend");
    items.forEach(([label, color]) => {
      const it = el("span", "legend-item");
      const dot = el("span", "legend-dot");
      dot.style.background = color;
      dot.style.color = color; /* drives the box-shadow glow (currentColor) */
      it.appendChild(dot);
      it.appendChild(document.createTextNode(label));
      leg.appendChild(it);
    });
  }

  // ===================== SSE (live freshness) =====================
  function connectSSE() {
    let es;
    try {
      es = new EventSource("/api/stream");
    } catch (e) {
      return;
    }
    es.addEventListener("refresh", (e) => {
      const m = JSON.parse(e.data);
      meta.counts = m.counts;
      meta.projects = m.projects;
      setCounts();
      flashFresh();
      refreshSessions(); // pulls fresh session dicts; timeline reads sessByContentId
      refresh(); // backend already reloaded; re-run current view
    });
    let lastSeenEnriched = -1;
    es.addEventListener("ping", (e) => {
      const m = JSON.parse(e.data);
      // labels may have been freshly enriched since init(); pull updated session dicts
      if (typeof m.enriched === "number" && m.enriched !== lastSeenEnriched) {
        lastSeenEnriched = m.enriched;
        if (!m.enriching && m.enriched > 0) refreshSessions();
      }
      // only update status if it's currently empty (don't clobber search/result strings)
      if ($("#sem-status").textContent) return;
      if (m.enriching)
        setStatus(`clarifying labels… (${m.enriched}/${m.enrichTotal})`);
      else if (m.indexing) setStatus(`indexing semantic… (${m.indexed})`);
    });
    es.onerror = () => {
      /* EventSource auto-reconnects */
    };
  }
  // Re-fetch /api/sessions and update sessByContentId so label/summary changes that
  // happened server-side (LLM enrichment, claude-mem growth) reach the client.
  async function refreshSessions() {
    try {
      const s = await api("/api/sessions");
      s.sessions.forEach((x) => sessByContentId.set(x.id, x));
    } catch (e) {
      /* non-fatal */
    }
  }

  function flashFresh() {
    const s = $("#sem-status");
    if (!s) return;
    const prev = s.textContent;
    s.textContent = "↻ updated with new entries";
    setTimeout(() => {
      if (s.textContent === "↻ updated with new entries") s.textContent = prev;
    }, 2500);
  }

  // clipboard (works on localhost which is a secure context)
  function copyText(txt) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(txt).catch(() => fallbackCopy(txt));
    } else {
      fallbackCopy(txt);
    }
  }
  function fallbackCopy(txt) {
    const ta = document.createElement("textarea");
    ta.value = txt;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
    } catch (e) {
      /* no-op */
    }
    document.body.removeChild(ta);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
