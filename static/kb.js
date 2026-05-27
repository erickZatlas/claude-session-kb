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
    renderGraph(currentRecs);
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

  function selectRecord(id) {
    state.selectedRecId = id;
    const r = recById.get(id);
    state.selectedSessId = r ? r.sessionId : null;
    document
      .querySelectorAll(".result")
      .forEach((x) => x.classList.toggle("sel", x.dataset.id === id));
    const sel = document.querySelector(".result.sel");
    if (sel) sel.scrollIntoView({ block: "nearest" });
    setActive(id);
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
        label: s ? truncate(s.title, 28) : "session",
        color: SESSION_COLOR,
        r: 9,
        ref: sid,
      });
    });
    focus.forEach((r) => {
      addNode(r.id, {
        kind: r.kind,
        label: truncate(r.title, 26),
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
      25,
    );
    addSharedConnectors(
      focus,
      "concepts",
      "C:",
      CONCEPT_COLOR,
      (x) => x,
      nodes,
      links,
      addNode,
      25,
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

  function renderGraph(recs) {
    const g = buildSubgraph(recs);
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
      .attr("stroke-width", (d) => (d.kind === "owns" ? 1.3 : 0.8))
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
      // labels are noise at scale: show only session anchors by default,
      // reveal the rest on hover/selection via setActive()
      .style("display", (d) => (d.kind === "session" ? null : "none"))
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
          .distance((d) => (d.kind === "owns" ? 60 : 95))
          .strength(0.35),
      )
      .force("charge", d3.forceManyBody().strength(-280))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("x", d3.forceX(W / 2).strength(0.04))
      .force("y", d3.forceY(H / 2).strength(0.04))
      .force(
        "collide",
        d3.forceCollide().radius((d) => d.r + 16),
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
        d.kind === "session" || (nodeId && neighbors.has(d.id)) ? null : "none",
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
      refresh(); // backend already reloaded; re-run current view
    });
    es.addEventListener("ping", (e) => {
      const m = JSON.parse(e.data);
      if (m.indexing && !$("#sem-status").textContent)
        setStatus(`indexing semantic… (${m.indexed})`);
    });
    es.onerror = () => {
      /* EventSource auto-reconnects */
    };
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
