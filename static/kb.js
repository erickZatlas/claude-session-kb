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
    sortMode: "recent", // "recent" | "activity" (overview session cards only)
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
  // Prefers UPPER_SNAKE acronyms (RETRY_LIMIT), then InitialCap, then longest.
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

  // Convert any string to a kebab label, trimmed to ~22 chars at a word
  // boundary (so "verification-of-cache-config" doesn't get cut mid-word).
  function toKebab(text) {
    if (!text) return null;
    const s = String(text)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
    if (!s) return null;
    if (s.length <= 22) return s;
    // round down to the last hyphen that keeps us under 22 chars
    let out = s.slice(0, 22);
    const lastHy = out.lastIndexOf("-");
    if (lastHy > 8) out = out.slice(0, lastHy);
    return out;
  }

  // Multi-token kebab label for an observation node — what the drill-down
  // graph shows under each circle. Priority order:
  //   1. server-provided r.label (rare; reserved for future)
  //   2. the first 2 topical tags from r.concepts (our Phase C/D tags), joined
  //   3. 2 distinctive tokens from r.title, joined
  //   4. truncated title as a last resort
  // The 2-tag join is the new path that this user noticed was missing — single
  // English words like "Update" / "Handler" carried no information when
  // every node in a session is about the same subsystem.
  function topTokens(text, n) {
    if (!text) return [];
    const toks = (text.match(/[A-Za-z_][A-Za-z0-9_]{3,}/g) || []).filter(
      (t) => !STOP.has(t.toLowerCase()) && !/^\d+$/.test(t),
    );
    if (!toks.length) return [];
    const score = (t) =>
      t.length +
      (/^[A-Z][A-Z0-9_]+$/.test(t) ? 12 : 0) +
      (/^[A-Z]/.test(t) ? 3 : 0);
    // Sort by score desc, dedupe case-insensitively, return up to n
    toks.sort((a, b) => score(b) - score(a));
    const seen = new Set();
    const out = [];
    for (const t of toks) {
      const k = t.toLowerCase();
      if (seen.has(k)) continue;
      seen.add(k);
      out.push(t);
      if (out.length >= n) break;
    }
    return out;
  }

  const recordLabel = (r) => {
    if (r.label) return r.label;
    // Use our Phase C/D topical tags first — they're already domain-specific
    // (file names, identifiers, kebab concepts) and avoid the "every node says
    // Update" problem.
    if (r.concepts && r.concepts.length) {
      const picks = r.concepts.slice(0, 2).map(String).join(" ");
      const kebab = toKebab(picks);
      if (kebab && kebab.includes("-")) return kebab;
      // single-tag case (e.g. tag was already a multi-word kebab like
      // "oauth-token-refresh") — keep it as-is via toKebab normalization
      const single = toKebab(r.concepts[0]);
      if (single) return single;
    }
    // Fall back to 2 distinctive tokens from the title
    const tokens = topTokens(r.title, 2);
    if (tokens.length >= 2) return toKebab(tokens.join(" "));
    if (tokens.length === 1) return toKebab(tokens[0]);
    return truncate(r.title || "(untitled)", 22);
  };

  async function api(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${path} → ${r.status}`);
    return r.json();
  }

  // ===================== URL STATE ===================
  // URL ↔ in-memory state codec. Drives browser back/forward + makes views
  // shareable + refresh-survives. pushHistory creates a back-step (session
  // drill-ins, "← all sessions"); replaceHistory updates the URL silently
  // (keystrokes, filter chips, record selection). Defaults are omitted from
  // the URL so it stays readable: /kb.html means "default overview".
  function _encodeURL() {
    const p = new URLSearchParams();
    if (state.query) p.set("q", state.query);
    if (state.selectedSessId) p.set("session", state.selectedSessId);
    if (state.selectedRecId && state.selectedSessId)
      p.set("record", state.selectedRecId);
    if (state.project !== "all") p.set("project", state.project);
    if (state.kind !== "all") p.set("kind", state.kind);
    if (state.mode !== "keyword") p.set("mode", state.mode);
    if (state.sortMode !== "recent") p.set("sort", state.sortMode);
    const qs = p.toString();
    return qs ? location.pathname + "?" + qs : location.pathname;
  }
  function _decodeURL() {
    const p = new URLSearchParams(location.search);
    return {
      query: p.get("q") || "",
      selectedSessId: p.get("session") || null,
      selectedRecId: p.get("record") || null,
      project: p.get("project") || "all",
      kind: p.get("kind") || "all",
      mode: p.get("mode") || "keyword",
      sortMode: p.get("sort") || "recent",
    };
  }
  let _suppressHistory = false; // avoid re-pushing during popstate-driven restore
  function pushHistory() {
    if (_suppressHistory) return;
    history.pushState(null, "", _encodeURL());
  }
  function replaceHistory() {
    if (_suppressHistory) return;
    history.replaceState(null, "", _encodeURL());
  }
  // Mirror URL → state → visible toolbar → re-fetch.
  async function restoreFromURL() {
    const incoming = _decodeURL();
    _suppressHistory = true;
    try {
      Object.assign(state, incoming);
      const s = $("#search");
      if (s) s.value = state.query;
      const proj = $("#proj-filter");
      if (proj) proj.value = state.project;
      document
        .querySelectorAll(".chip[data-kind]")
        .forEach((c) =>
          c.classList.toggle("active", c.dataset.kind === state.kind),
        );
      document
        .querySelectorAll(".chip[data-mode]")
        .forEach((c) =>
          c.classList.toggle("active", c.dataset.mode === state.mode),
        );
      document
        .querySelectorAll(".chip[data-sort]")
        .forEach((c) =>
          c.classList.toggle("active", c.dataset.sort === state.sortMode),
        );
      if (s) {
        s.placeholder =
          state.mode === "semantic"
            ? "Describe what you're looking for — searches by meaning…"
            : "Search observations, summaries, facts, concepts, files…";
      }
      await refresh();
    } finally {
      _suppressHistory = false;
    }
  }
  window.addEventListener("popstate", () => {
    restoreFromURL();
  });

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
      replaceHistory();
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
          replaceHistory();
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
        replaceHistory();
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
        replaceHistory();
        refresh();
      }),
    );

    // sort toggle (session-card overview only)
    document.querySelectorAll(".chip[data-sort]").forEach((c) =>
      c.addEventListener("click", () => {
        document
          .querySelectorAll(".chip[data-sort]")
          .forEach((x) => x.classList.remove("active"));
        c.classList.add("active");
        state.sortMode = c.dataset.sort;
        replaceHistory();
        renderCards(); // pure re-render; no fetch needed
      }),
    );

    // "← all sessions" — push a back-step so browser back returns to the drill.
    $("#reset-graph").addEventListener("click", () => {
      clearSelection();
      pushHistory();
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
    // Boot from URL so ?session=… / ?q=… etc. land you exactly where the URL
    // points (refresh-survive + shareable links). Falls through to the
    // default overview when there's no query string.
    await restoreFromURL();
  }

  function setCounts() {
    const c = meta.counts;
    const lessons = c.lessons ? ` · ${c.lessons} lessons` : "";
    const mem = c.memoryFacts ? ` · ${c.memoryFacts} memory` : "";
    $("#kb-counts").textContent =
      `${c.observations} observations${lessons}${mem} · ${c.summaries} summaries · ${c.sessions} sessions · live`;
  }
  function setStatus(msg) {
    const s = $("#sem-status");
    if (s) s.textContent = msg;
  }

  // ===================== DATA =====================
  async function refresh() {
    const token = ++queryToken;
    if (state.mode === "semantic" && state.query) setStatus("searching…");
    // When drilled into a session, prefer the session filter and ignore the
    // search box for the fetch — keeps the input visually populated so going
    // back to overview restores the user's search context without re-typing.
    const params = new URLSearchParams({
      q: state.selectedSessId ? "" : state.query,
      project: state.project,
      kind: state.kind,
      mode: state.mode,
      limit: "250",
    });
    if (state.selectedSessId) params.set("session", state.selectedSessId);

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
    // In overview the cards own #results (rendered by renderGraphView -> renderCards).
    // In drill-down the existing renderResults rows fill the records pane.
    if (state.selectedSessId) renderResults(currentRecs);
    await renderGraphView();
    renderDetail();
  }

  // ===================== RESULTS =====================
  // Drill-down records pane (one session's observations). The cards/labels
  // are the same shape as the search-result card list, so the title needs to
  // say what these are — otherwise it stays at "Sessions" from the last
  // overview render and reads as if you're browsing sessions.
  function renderResults(recs) {
    const list = $("#results");
    list.innerHTML = "";
    $("#pane-title-label").textContent = "Observations";
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
          // Concept click pivots to a fresh search — push so back returns to
          // the record the user was reading.
          pushHistory();
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

    // Session footer block on a record-detail pane. When we're already drilled
    // into this session, drop redundant chrome (label heading + "Focus this
    // session in graph" button) — the user already knows which session this is.
    const sess = sessByContentId.get(r.sessionId);
    const alreadyDrilled = r.sessionId && state.selectedSessId === r.sessionId;
    const box = el("div", "session-box");
    box.appendChild(el("div", "sub", "Session"));
    if (!r.sessionId) {
      box.appendChild(el("div", "body", "(no linked session)"));
      d.appendChild(box);
      return;
    }
    if (sess && !alreadyDrilled) {
      // Heading: kebab label (preferred) or truncated first prompt
      const heading = el(
        "div",
        "session-heading",
        sess.label || truncate(sess.title || "session", 60),
      );
      heading.style.fontWeight = "600";
      heading.style.color = "var(--text)";
      heading.style.marginBottom = "4px";
      box.appendChild(heading);
      // Chips: project + worktree (if any)
      const chips = el("div", "row");
      chips.style.marginBottom = "6px";
      chips.appendChild(el("span", "cchip", sess.project || ""));
      if (sess.worktree) {
        const wt = el(
          "span",
          "cchip card-chip-worktree",
          "wt: " + sess.worktree,
        );
        wt.title = "git worktree directory";
        chips.appendChild(wt);
      }
      box.appendChild(chips);
      // Summary, when present
      if (sess.summary) {
        const sum = el("div", "body", sess.summary);
        sum.style.fontStyle = "italic";
        sum.style.opacity = "0.85";
        sum.style.marginBottom = "6px";
        box.appendChild(sum);
      }
    }
    // Resume command on its own line — readable, not running into the buttons
    const codeRow = el("div");
    codeRow.style.marginBottom = "8px";
    codeRow.appendChild(el("code", null, "claude --resume " + r.sessionId));
    box.appendChild(codeRow);
    // Buttons. "Focus this session in graph" hidden when already drilled in
    // (clicking it would be a no-op).
    const actions = el("div", "row");
    actions.style.gap = "6px";
    const btn = el("button", "btn", "Copy resume command");
    btn.addEventListener("click", () => {
      copyText("claude --resume " + r.sessionId);
      btn.textContent = "Copied!";
      setTimeout(() => (btn.textContent = "Copy resume command"), 1200);
    });
    actions.appendChild(btn);
    if (!alreadyDrilled) {
      const focus = el("button", "btn", "Focus this session in graph");
      focus.addEventListener("click", () => selectSession(r.sessionId));
      actions.appendChild(focus);
    }
    box.appendChild(actions);
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
    if (s.worktree) {
      const wt = el("span", "cchip card-chip-worktree", "wt: " + s.worktree);
      wt.title = "git worktree directory";
      row.appendChild(wt);
    }
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

    if (s.id) d.appendChild(renderDeleteSession(s));
  }

  // Danger zone for a session drill-in. Two-step arm→confirm (no blocking
  // native dialog): first click arms for 4s, second click within that window
  // fires DELETE /api/sessions/{id}. The transcript is trashed (recoverable),
  // not purged, matching the CLI default.
  function renderDeleteSession(s) {
    const wrap = el("div", "session-box");
    wrap.style.marginTop = "10px";
    wrap.appendChild(el("div", "sub", "Danger zone"));
    const hint = el(
      "div",
      "body",
      "Removes this session from the picker (transcript moved to ~/.claude-kb/trash, recoverable) and deletes its KB rows.",
    );
    hint.style.fontSize = "11px";
    wrap.appendChild(hint);

    const btn = el("button", "btn danger", "Delete session");
    btn.style.marginTop = "8px";
    let armed = false;
    let armTimer = null;
    const disarm = () => {
      armed = false;
      btn.classList.remove("arm");
      btn.textContent = "Delete session";
      if (armTimer) clearTimeout(armTimer);
    };
    btn.addEventListener("click", async () => {
      if (!armed) {
        armed = true;
        btn.classList.add("arm");
        btn.textContent = "Click again to confirm";
        armTimer = setTimeout(disarm, 4000);
        return;
      }
      if (armTimer) clearTimeout(armTimer);
      btn.disabled = true;
      btn.textContent = "Deleting…";
      try {
        const r = await fetch("/api/sessions/" + encodeURIComponent(s.id), {
          method: "DELETE",
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.detail || `${r.status}`);
        // Session is gone — drop the drill-in and reload the overview so the
        // card disappears. Refresh counts + the session map from the server,
        // then re-render.
        sessByContentId.delete(s.id);
        try {
          meta = await api("/api/meta");
          setCounts();
          sessByContentId.clear();
          const ss = await api("/api/sessions");
          ss.sessions.forEach((x) => sessByContentId.set(x.id, x));
        } catch (_) {
          /* non-fatal — the drill-in still clears below */
        }
        clearSelection();
        pushHistory();
        refresh();
      } catch (e) {
        btn.disabled = false;
        btn.classList.remove("arm");
        btn.textContent = "Delete failed — retry";
        const err = el("div", "body", String(e.message || e));
        err.style.color = "#d9776c";
        err.style.marginTop = "6px";
        wrap.appendChild(err);
      }
    });
    wrap.appendChild(btn);
    return wrap;
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
    // Record selection is too granular for the back-stack (it'd explode while
    // scrolling a results list). Update the URL silently so refresh-survival
    // still works; browser back skips over individual record picks.
    replaceHistory();
    renderDetail();
  }
  function selectSession(sessId) {
    state.selectedSessId = sessId;
    state.selectedRecId = null;
    // Preserve state.query + the visible search input so going "← all sessions"
    // restores the user's search context. refresh() ignores the query while
    // drilled (prefers the session filter for the fetch).
    pushHistory();
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

  // ---- overview: search-first card list ----
  // Two flavours, picked by state.query:
  //   • no query  -> session cards (sorted by recent | activity)
  //   • query     -> record cards (the API's ranked results, one card per hit)
  function relativeAgo(iso) {
    if (!iso) return "";
    const t = new Date(iso).getTime();
    if (isNaN(t)) return "";
    const s = Math.max(1, Math.floor((Date.now() - t) / 1000));
    if (s < 60) return s + "s ago";
    if (s < 3600) return Math.floor(s / 60) + "m ago";
    if (s < 86400) return Math.floor(s / 3600) + "h ago";
    if (s < 86400 * 14) return Math.floor(s / 86400) + "d ago";
    if (s < 86400 * 60) return Math.floor(s / (86400 * 7)) + "w ago";
    return Math.floor(s / (86400 * 30)) + "mo ago";
  }

  function sessionCardsData() {
    let pool = [...sessByContentId.values()];
    if (state.project !== "all")
      pool = pool.filter((s) => s.project === state.project);
    if (state.sortMode === "activity") {
      pool.sort((a, b) => (b.obsCount || 0) - (a.obsCount || 0));
    } else {
      pool.sort((a, b) => (b.started || "").localeCompare(a.started || ""));
    }
    return pool;
  }

  function renderCards() {
    const list = $("#results");
    list.innerHTML = "";
    const isSearching = !!state.query;
    if (isSearching) {
      // Record cards from currentRecs (the API's ranked /api/search hits)
      $("#results-count").textContent = currentRecs.length;
      $("#pane-title-label").textContent = "Results";
      if (!currentRecs.length) {
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
      currentRecs.forEach((r) => {
        const sess = sessByContentId.get(r.sessionId);
        const card = el("div", "card card-rec");
        card.dataset.id = r.id;
        if (r.id === state.selectedRecId) card.classList.add("sel");

        const head = el("div", "card-head");
        head.appendChild(el("span", "badge " + r.type, r.type));
        if (sess && sess.label)
          head.appendChild(el("span", "cchip", sess.label));
        card.appendChild(head);

        card.appendChild(el("h3", "card-title", r.title || "(untitled)"));
        const snip = r.subtitle || (r.text || "").slice(0, 200);
        if (snip) card.appendChild(el("div", "card-summary", snip));

        const meta = el("div", "card-meta");
        meta.appendChild(el("span", "card-chip", r.project));
        if (r.date)
          meta.appendChild(el("span", "card-chip", r.date.slice(0, 10)));
        if (r.files && r.files.length)
          meta.appendChild(
            el(
              "span",
              "card-chip",
              r.files.length + " file" + (r.files.length === 1 ? "" : "s"),
            ),
          );
        card.appendChild(meta);

        if (sess) {
          const from = el("div", "card-from");
          from.appendChild(document.createTextNode("from session: "));
          const link = el(
            "a",
            null,
            sess.label || truncate(sess.title || "(untitled)", 40),
          );
          link.href = "#";
          link.addEventListener("click", (ev) => {
            ev.preventDefault();
            ev.stopPropagation();
            selectSession(sess.id);
          });
          from.appendChild(link);
          card.appendChild(from);
        }

        card.addEventListener("click", () => selectRecord(r.id));
        frag.appendChild(card);
      });
      list.appendChild(frag);
    } else {
      // Session cards (default overview)
      const pool = sessionCardsData();
      $("#results-count").textContent = pool.length;
      $("#pane-title-label").textContent = "Sessions";
      if (!pool.length) {
        list.appendChild(
          el("div", "empty", "No sessions in this project yet."),
        );
        return;
      }
      const frag = document.createDocumentFragment();
      pool.forEach((s) => {
        const card = el("div", "card card-sess");
        card.dataset.id = s.id;

        card.appendChild(
          el(
            "h3",
            "card-title card-title-teal",
            s.label || truncate(s.title || "session", 30),
          ),
        );

        const meta = el("div", "card-meta");
        meta.appendChild(
          el("span", "card-chip card-chip-strong", s.project || ""),
        );
        if (s.worktree) {
          const wt = el(
            "span",
            "card-chip card-chip-worktree",
            "wt: " + s.worktree,
          );
          wt.title = "git worktree directory";
          meta.appendChild(wt);
        }
        if (s.obsCount != null)
          meta.appendChild(el("span", "card-chip", s.obsCount + " obs"));
        if (s.filesCount) {
          const fc = el("span", "card-chip", "📁 " + s.filesCount);
          fc.title = "files touched";
          meta.appendChild(fc);
        }
        if (s.started)
          meta.appendChild(el("span", "card-chip", relativeAgo(s.started)));
        card.appendChild(meta);

        if (s.summary) {
          card.appendChild(
            el("div", "card-summary card-summary-italic", s.summary),
          );
        } else if (s.title) {
          card.appendChild(el("div", "card-summary", truncate(s.title, 200)));
        }

        const actions = el("div", "card-actions");
        const resume = el("button", "btn", "Copy resume");
        resume.addEventListener("click", (ev) => {
          ev.stopPropagation();
          copyText("claude --resume " + s.id);
          resume.textContent = "Copied!";
          setTimeout(() => (resume.textContent = "Copy resume"), 1200);
        });
        actions.appendChild(resume);
        card.appendChild(actions);

        card.addEventListener("click", () => selectSession(s.id));
        frag.appendChild(card);
      });
      list.appendChild(frag);
    }
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

  // Full-width "you are here" banner that sits between the toolbar and the
  // three panes when drilled into a session. Always answers the question
  // "which session am I in?" without forcing the user to scan the graph or
  // wait for the right detail pane to fill. Hidden in overview mode.
  function renderSessionBanner() {
    let banner = document.getElementById("session-banner");
    if (!state.selectedSessId) {
      if (banner) banner.style.display = "none";
      return;
    }
    if (!banner) {
      banner = el("div");
      banner.id = "session-banner";
      banner.className = "session-banner";
      banner.style.cssText =
        "display:flex;align-items:center;gap:10px;flex-wrap:wrap;" +
        "padding:10px 16px;margin:0;background:rgba(94,234,212,0.06);" +
        "border-top:1px solid rgba(94,234,212,0.25);" +
        "border-bottom:1px solid rgba(94,234,212,0.25);" +
        "font-size:13px;color:var(--text,#e2e8f0);";
      const main = document.querySelector(".kb-main");
      if (main && main.parentNode) main.parentNode.insertBefore(banner, main);
      else document.body.appendChild(banner);
    }
    banner.style.display = "flex";
    banner.innerHTML = "";
    const sess = sessByContentId.get(state.selectedSessId);
    // Indicator + kebab label as the prominent identity marker
    const dot = el("span", null, "▸");
    dot.style.cssText = "color:#5eead4;font-weight:700;font-size:14px;";
    banner.appendChild(dot);
    const label = el(
      "strong",
      null,
      sess
        ? sess.label || truncate(sess.title || "session", 40)
        : state.selectedSessId.slice(0, 8) + "…",
    );
    label.style.cssText = "color:#5eead4;font-size:14px;letter-spacing:0.2px;";
    banner.appendChild(label);
    if (sess) {
      banner.appendChild(el("span", "cchip", sess.project || ""));
      if (sess.worktree) {
        const wt = el(
          "span",
          "cchip card-chip-worktree",
          "wt: " + sess.worktree,
        );
        wt.title = "git worktree directory";
        banner.appendChild(wt);
      }
      if (sess.obsCount != null) {
        banner.appendChild(el("span", "cchip", sess.obsCount + " obs"));
      }
      if (sess.summary) {
        const sum = el("span", null, sess.summary);
        sum.style.cssText =
          "color:#94a3b8;font-style:italic;font-size:12px;" +
          "flex:1;min-width:200px;overflow:hidden;text-overflow:ellipsis;" +
          "white-space:nowrap;";
        sum.title = sess.summary;
        banner.appendChild(sum);
      }
    }
  }

  // Decide which graph to show: drill-down (a session selected), search overview
  // (a query active), or the default session map.
  function renderGraphView() {
    const resetBtn = $("#reset-graph");
    const body = document.body;
    if (state.selectedSessId) {
      // drill-down: force-graph + records list (the existing three-pane layout)
      body.classList.remove("view-cards");
      body.classList.toggle("searching", !!state.query);
      if (resetBtn) resetBtn.textContent = "← all sessions";
      renderSessionBanner();
      return renderForce(buildSubgraph(currentRecs), { labelAll: true });
    }
    // overview: cards (graph pane hidden via .view-cards on body)
    body.classList.add("view-cards");
    body.classList.toggle("searching", !!state.query);
    if (resetBtn) resetBtn.textContent = "Reset view";
    renderSessionBanner();
    renderCards();
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
