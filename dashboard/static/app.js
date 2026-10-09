/**
 * NeuralFabric dashboard: D3 topology, Chart.js metrics, WebSocket live updates.
 *
 * Link IDs are "u--v" with u < v (one per physical link), as sent by the server.
 * Topology link states:
 *   congested now  served_or_queued label from the server (solid red)
 *   predicted      calibrated GNN probability > live alert threshold (pulsing dashed outline)
 *   reroute        old path dashed red, new path cyan, trigger link dark halo + "!" badge
 *                  (shown for TRANSIENT_MS; clicking the timeline entry pins it)
 *
 * Event timeline: the server sends transition events (congestion / cleared / predicted,
 * coalesced per tick), reroutes, spikes and controller/workload switches, each with a unique
 * id; a viewer that connects mid-run receives the run's backlog with the topology message.
 */

// ── State ──────────────────────────────────────────────────────
let ws = null;
let topologyData = null;
let linkElements = {};     // linkId -> base line
let linkGeom = {};         // linkId -> {x1, y1, x2, y2}
let predOutlines = {};     // linkId -> predicted-congestion outline (fabric links only)
let transientLayer = null; // reroute highlights that fade after TRANSIENT_MS
let selectionLayer = null; // highlight pinned by clicking a timeline entry
let isRunning = false;

let liveThreshold = null;   // GNN alert threshold (slider), synced from the server
let alertThreshold = null;  // calibrated threshold stored in the checkpoint
let lastLocalThresholdAt = -Infinity;
let modelLoaded = false;
let activeController = null;
let activeWorkload = 'stride';

const TRANSIENT_MS = 2000;   // how long a new reroute stays drawn on the topology
const TRANSIENT_MAX = 3;     // at most this many fresh reroutes drawn at once (the guard can make several per tick)
const RELATED_TICKS = 10;    // a congestion event lists reroutes triggered within this many ticks
const TIMELINE_MAX = 300;    // entries kept in the timeline
const STORE_MAX = 5000;      // events kept for click-to-highlight
const CONTROLLER_NAMES = { ecmp: 'ECMP', guarded_gnn: 'Guarded GNN', guarded_score: 'Guarded score' };

// Fonts come from the CSS system stacks (no web fonts, works offline).
const rootStyle = getComputedStyle(document.documentElement);
const FONT_MAIN = rootStyle.getPropertyValue('--font-main').trim() || 'sans-serif';
const FONT_MONO = rootStyle.getPropertyValue('--font-mono').trim() || 'monospace';

// Sparkline data
const SPARK_KEYS = ['flows', 'packets', 'fabric', 'host', 'droprate', 'reroutes', 'predictions'];
const SPARK_MAX = 30;
const sparkData = Object.fromEntries(SPARK_KEYS.map(k => [k, []]));

// Time-series data buffers
const MAX_HISTORY = 120;
const metricsHistory = {
    ticks: [],
    avgUtilization: [],
    maxUtilization: [],
    flows: [],
    fabricDropsPerTick: [],
    hostDropsPerTick: [],
};

// Fabric health: 100 − 10 × fabric-drop % of offered packets over the last HEALTH_WINDOW ticks.
const HEALTH_WINDOW = 20;
let healthWindow = [];   // per-tick {fabric, offered}
let previous = null;     // cumulative metrics of the previous update

// Latest link data for tooltips
let latestLinkData = {};
let latestPredictions = {};

// Timeline / highlight state
const eventStore = new Map();  // event id -> event (with .run)
let rerouteEvents = [];        // reroute events, for linking congestion -> reroutes
let runCounter = 0;            // bumped on every restart: ticks restart at 0
let selectedId = null;
let unseenEvents = 0;

// Charts
let utilizationChart = null;
let dropsChart = null;

// ── Initialization ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initSparklines();
    setupControls();
    setupTimeline();
    trackControlBarHeight();
    addEntry('info', 'Dashboard ready. Press Start to run the live network (default workload: stride).');
});

function setupControls() {
    document.getElementById('btn-start').addEventListener('click', startSimulation);
    document.getElementById('btn-stop').addEventListener('click', stopSimulation);
    document.getElementById('btn-spike').addEventListener('click', injectSpike);
    const workload = document.getElementById('workload-select');
    workload.addEventListener('change', () => switchWorkload(workload.value));
    const controller = document.getElementById('controller-select');
    controller.addEventListener('change', () => switchController(controller.value));
    setupSliders();
}

// The right column sticks below the control bar; keep its offset equal to the bar's height
// (the bar wraps to two rows on narrow screens).
function trackControlBarHeight() {
    const bar = document.getElementById('control-bar');
    const apply = () => document.documentElement.style.setProperty('--controlbar-h', `${bar.offsetHeight}px`);
    apply();
    if (window.ResizeObserver) new ResizeObserver(apply).observe(bar);
}

// ── Link and node names ────────────────────────────────────────
function linkId(a, b) {
    return a < b ? `${a}--${b}` : `${b}--${a}`;
}

function shortName(nodeId) {
    return nodeId
        .replace('core_', 'C')
        .replace(/agg_(\d+)_(\d+)/, 'A$1.$2')
        .replace(/edge_(\d+)_(\d+)/, 'E$1.$2')
        .replace(/host_(\d+)_(\d+)_(\d+)/, 'H$1.$2.$3');
}

const LAYER_RANK = { core: 0, agg: 1, edge: 2, host: 3 };

function nodeRank(nodeId) {
    return LAYER_RANK[nodeId.split('_')[0]] ?? 4;
}

// Upper layer first, e.g. "C0–A0.0", "A1.0–E1.1".
function linkEnds(id) {
    return id.split('--').sort((a, b) => nodeRank(a) - nodeRank(b) || (a < b ? -1 : 1));
}

function linkLabel(id) {
    return linkEnds(id).map(shortName).join('–');
}

function pathLinks(path) {
    const ids = [];
    for (let i = 0; i + 1 < path.length; i++) ids.push(linkId(path[i], path[i + 1]));
    return ids;
}

// Switch hops only (src and dst hosts are named separately), e.g. "E0.0 › A0.1 › C2 › A1.1 › E1.0".
function switchPath(path) {
    return path.filter(n => !n.startsWith('host_')).map(shortName).join(' › ');
}

function pct(p) {
    return `${(p * 100).toFixed(1)}%`;
}

// ── WebSocket ──────────────────────────────────────────────────
function connectWebSocket() {
    if (ws && ws.readyState <= WebSocket.OPEN) return;  // already connecting or open
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        document.getElementById('status-badge').classList.add('connected');
        document.getElementById('status-text').textContent = 'Live';
        addEntry('info', 'Connected to the simulation engine.');
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'topology') {
                topologyData = msg.data;
                renderTopology(msg.data);
                (msg.recent_events || []).forEach(evt => handleServerEvent(evt, { live: false }));
            } else if (msg.type === 'update') {
                handleUpdate(msg);
            } else if (msg.type === 'error') {
                addEntry('error', msg.message);
            }
        } catch (e) {
            console.error('Message handling error:', e);
        }
    };

    ws.onclose = () => {
        document.getElementById('status-badge').classList.remove('connected');
        document.getElementById('status-text').textContent = 'Offline';
        addEntry('error', 'Connection to the simulation engine lost; retrying while the simulation runs.');
        setTimeout(() => { if (isRunning) connectWebSocket(); }, 3000);
    };
}

// ── Simulation Controls ────────────────────────────────────────
async function startSimulation() {
    try {
        const runAtRequest = runCounter;
        const resp = await fetch('/api/start', { method: 'POST' });
        const data = await resp.json();

        if (data.status === 'started' || data.status === 'already_running') {
            setRunningUI(true);
            // Reset unless the new run's first updates already triggered it (handleUpdate).
            if (data.status === 'started' && runCounter === runAtRequest) resetSeries();
            if (data.topology) {
                topologyData = data.topology;
                renderTopology(data.topology);
            }
            // A new run resets the live threshold to the checkpoint's calibrated value.
            if (data.status === 'started') syncControllerState(data, { forceThreshold: true });
            connectWebSocket();
        }
    } catch (e) {
        addEntry('error', `Failed to start: ${e.message}`);
    }
}

function setRunningUI(running) {
    isRunning = running;
    document.getElementById('btn-start').disabled = running;
    document.getElementById('btn-stop').disabled = !running;
    document.getElementById('btn-spike').disabled = !running;
    if (running) {
        document.getElementById('controller-select').disabled = false;
        document.getElementById('topology-overlay').classList.add('hidden');
    }
}

async function stopSimulation() {
    try {
        await fetch('/api/stop', { method: 'POST' });
        setRunningUI(false);
        addEntry('info', 'Simulation stopped. Press Start for a fresh run.');
    } catch (e) {
        addEntry('error', `Failed to stop: ${e.message}`);
    }
}

async function injectSpike() {
    try {
        await fetch('/api/spike', { method: 'POST' });  // the server logs a "spike" event
    } catch (e) {
        addEntry('error', `Spike injection failed: ${e.message}`);
    }
}

// Before the first start this only selects the workload; afterwards the server restarts the
// simulation with it, keeping the controller and the live threshold.
async function switchWorkload(name) {
    const select = document.getElementById('workload-select');
    try {
        const runAtRequest = runCounter;
        const resp = await fetch('/api/workload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ workload: name }),
        });
        const data = await resp.json();
        if (!resp.ok) {
            addEntry('error', `Workload switch refused: ${typeof data.detail === 'string' ? data.detail : resp.statusText}`);
            select.value = activeWorkload;
            return;
        }
        if (data.status === 'restarted') {
            if (runCounter === runAtRequest) resetSeries();  // unless an update already did
            setRunningUI(true);
            syncControllerState(data);
            connectWebSocket();
        } else {
            activeWorkload = name;
            document.getElementById('workload-active').textContent = name;
            addEntry('info', `Workload set to ${name}; it runs when the engine starts.`);
        }
    } catch (e) {
        addEntry('error', `Workload switch failed: ${e.message}`);
        select.value = activeWorkload;
    }
}

async function switchController(name) {
    const select = document.getElementById('controller-select');
    try {
        const resp = await fetch('/api/controller', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ controller: name }),
        });
        const data = await resp.json();
        if (!resp.ok) {
            addEntry('error', `Controller switch refused: ${typeof data.detail === 'string' ? data.detail : resp.statusText}`);
            if (activeController) select.value = activeController;
            return;
        }
        clearTransient();
        syncControllerState(data);
    } catch (e) {
        addEntry('error', `Controller switch failed: ${e.message}`);
        if (activeController) select.value = activeController;
    }
}

// Applies the run fields the server sends with every update (and the start/switch responses).
function syncControllerState(msg, { forceThreshold = false } = {}) {
    modelLoaded = msg.model_loaded;
    alertThreshold = msg.alert_threshold;
    activeController = msg.controller;

    const select = document.getElementById('controller-select');
    select.querySelector('option[value="guarded_gnn"]').disabled = !modelLoaded;
    if (document.activeElement !== select) select.value = msg.controller;
    document.getElementById('controller-active').textContent = msg.controller_label;

    activeWorkload = msg.workload;
    const workloadSelect = document.getElementById('workload-select');
    if (document.activeElement !== workloadSelect) workloadSelect.value = msg.workload;
    document.getElementById('workload-active').textContent = msg.workload;

    const reset = document.getElementById('threshold-reset');
    reset.disabled = alertThreshold == null;
    reset.textContent = alertThreshold == null ? 'calibrated: n/a' : `calibrated: ${alertThreshold.toFixed(3)}`;

    // Follow the server's threshold (e.g. reset on a new run), except right after a local
    // slider change, while messages sent before the server applied it are still arriving.
    const recentlyMoved = performance.now() - lastLocalThresholdAt < 1000;
    if (msg.threshold != null && msg.threshold !== liveThreshold && (forceThreshold || !recentlyMoved)) {
        setThreshold(msg.threshold, { send: false });
    }

    const s = msg.since_switch;
    if (s) {
        document.getElementById('since-ticks').textContent = s.ticks;
        document.getElementById('since-drops').textContent = formatNumber(s.total_drops);
        document.getElementById('since-fabric').textContent = formatNumber(s.fabric_drops);
        document.getElementById('since-host').textContent = formatNumber(s.host_drops);
        document.getElementById('since-rate').textContent = s.ticks > 0 ? formatNumber(Math.round(s.total_drops / s.ticks)) : '0';
        document.getElementById('since-reroutes').textContent = formatNumber(s.reroutes);
    }
}

// ── Handle Real-Time Updates ───────────────────────────────────
// Clears per-run history; called when the simulation restarts (e.g. a workload switch).
function resetSeries() {
    Object.values(metricsHistory).forEach(series => { series.length = 0; });
    Object.values(sparkData).forEach(series => { series.length = 0; });
    Object.values(sparkCanvases).forEach(ctx => ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height));
    healthWindow = [];
    previous = null;
    latestPredictions = {};
    runCounter++;
    clearTransient();
    clearSelection();
    updateCharts();
}

function handleUpdate(msg) {
    const { links, predictions, metrics } = msg;

    // A tick counter going backwards means the server restarted the run (possibly from another viewer).
    if (previous && metrics.tick < previous.tick) resetSeries();

    syncControllerState(msg);

    document.getElementById('tick-value').textContent = metrics.tick;
    document.getElementById('val-flows').textContent = metrics.active_flows;
    document.getElementById('val-packets').textContent = formatNumber(metrics.total_packets);
    document.getElementById('val-fabric-drops').textContent = formatNumber(metrics.fabric_drops);
    document.getElementById('val-host-drops').textContent = formatNumber(metrics.host_drops);
    document.getElementById('val-droprate').textContent = `${metrics.drop_rate}%`;
    document.getElementById('val-reroutes').textContent = formatNumber(metrics.total_reroutes);

    latestPredictions = predictions || {};
    const alertCount = countAlerts(latestPredictions);
    document.getElementById('val-predictions').textContent = alertCount;

    pushSparkData('flows', metrics.active_flows);
    pushSparkData('packets', metrics.total_packets);
    pushSparkData('fabric', metrics.fabric_drops);
    pushSparkData('host', metrics.host_drops);
    pushSparkData('droprate', metrics.drop_rate);
    pushSparkData('reroutes', metrics.total_reroutes);
    pushSparkData('predictions', alertCount);
    drawAllSparklines();

    updateTopologyLinks(links);
    refreshPredictionViews();

    // Per-tick deltas of the cumulative counters
    const delta = key => previous ? Math.max(0, metrics[key] - previous[key]) : 0;
    const fabricThisTick = delta('fabric_drops');
    const hostThisTick = delta('host_drops');
    const offeredThisTick = delta('total_packets');
    previous = metrics;
    updateTimeSeries(links, metrics, fabricThisTick, hostThisTick);
    if (metrics.tick % 3 === 0) updateCharts();
    updateHealthScore(fabricThisTick, offeredThisTick);

    // Timeline: this tick's transitions first, then its reroutes (newest ends up on top).
    (msg.events || []).forEach(evt => handleServerEvent(evt));
    (msg.reroute_events || []).forEach(evt => handleServerEvent({ ...evt, type: 'reroute' }));
}

function countAlerts(predictions) {
    if (liveThreshold === null) return 0;
    return Object.values(predictions).filter(p => p > liveThreshold).length;
}

// Re-applies the live threshold to the topology outlines, heatmap and alert count.
function refreshPredictionViews() {
    updatePredictionOutlines(latestPredictions);
    updatePredictionPanel(latestPredictions);
    document.getElementById('val-predictions').textContent = countAlerts(latestPredictions);
}

// ── Event Timeline ─────────────────────────────────────────────
const TYPE_META = {
    congestion: { label: 'Congestion', icon: '▲' },
    congestion_cleared: { label: 'Cleared', icon: '▽' },
    predicted: { label: 'Predicted', icon: '◆' },
    reroute: { label: 'Reroute', icon: '↗' },
    spike: { label: 'Spike', icon: '⚡︎' },
    controller: { label: 'Controller', icon: '⇄' },
    workload: { label: 'Workload', icon: '⟳' },
    start: { label: 'Start', icon: '▶︎' },
    info: { label: 'Info', icon: 'ℹ︎' },
    error: { label: 'Error', icon: '✕' },
};

function setupTimeline() {
    const list = document.getElementById('event-log');
    const latest = document.getElementById('timeline-latest');
    list.addEventListener('scroll', () => {
        if (list.scrollTop <= 4) {  // back at the newest entry: resume following
            unseenEvents = 0;
            latest.hidden = true;
        }
    });
    latest.addEventListener('click', () => {
        list.scrollTo({ top: 0 });
        unseenEvents = 0;
        latest.hidden = true;
    });
    document.getElementById('timeline-clear').addEventListener('click', clearLog);
}

function describeEvent(evt) {
    const who = CONTROLLER_NAMES[evt.controller] || evt.controller;
    switch (evt.type) {
        case 'congestion':
            return `Congestion on ${evt.links.length === 1 ? '' : `${evt.links.length} links: `}` + evt.links.map(d =>
                `${linkLabel(d.link)} (offered ${pct(d.utilization)}, served ${pct(d.served)}, ` +
                `queue ${d.queue_depth} pkts = ${pct(d.queue_fraction)} of buffer)`).join('; ');
        case 'congestion_cleared':
            return `Congestion cleared on ${evt.links.map(linkLabel).join(', ')}`;
        case 'predicted':
            return `Predicted congestion within 5 ticks on ` + evt.links.map(d =>
                `${linkLabel(d.link)} (p = ${d.probability.toFixed(2)})`).join(', ') +
                ` (above the alert threshold ${evt.threshold.toFixed(3)})`;
        case 'reroute': {
            const trigger = evt.trigger_link
                ? ` off ${linkLabel(evt.trigger_link)} (${evt.trigger_signal} ${evt.trigger_value.toFixed(2)})`
                : '';
            return `${who} moved flow ${evt.flow_id} (${shortName(evt.src)} → ${shortName(evt.dst)})${trigger}: ` +
                `old ${switchPath(evt.old_path)} → new ${switchPath(evt.new_path)}`;
        }
        case 'spike':
            return `Traffic spike: ${evt.flows} hotspot/incast flows injected`;
        case 'controller':
            return `Controller → ${evt.controller_label}. Every flow is back on its ECMP path; the since-switch counters restart.`;
        case 'workload':
            return `Workload → ${evt.workload}: simulation restarted (controller ${evt.controller_label} kept).`;
        case 'start':
            return `Simulation started: workload ${evt.workload}, controller ${evt.controller_label}.`;
        default:
            return evt.message || evt.type;
    }
}

// Logs one server event (once per id) and, for a live reroute, flashes it on the topology.
function handleServerEvent(evt, { live = true } = {}) {
    if (evt.id != null && eventStore.has(evt.id)) return;
    evt.run = runCounter;
    if (evt.id != null) {
        eventStore.set(evt.id, evt);
        if (eventStore.size > STORE_MAX) eventStore.delete(eventStore.keys().next().value);
    }
    if (evt.type === 'reroute') {
        rerouteEvents.push(evt);
        if (rerouteEvents.length > STORE_MAX) rerouteEvents.shift();
        if (live) flashReroute(evt);
    }
    addEntry(evt.type, describeEvent(evt), evt);
}

// Adds a timeline entry at the top. If the reader has scrolled away from the newest entry,
// their view is kept in place and a "new events" button offers to jump back.
function addEntry(type, message, evt = null) {
    const list = document.getElementById('event-log');
    const meta = TYPE_META[type] || TYPE_META.info;
    const li = document.createElement('li');
    li.className = `tl-entry tl-${type}`;
    li.innerHTML = '<div class="tl-row"><span class="tl-badge"><span class="tl-icon" aria-hidden="true"></span>' +
        '<span class="tl-type"></span></span><span class="tl-time"></span></div><div class="tl-msg"></div>';
    li.querySelector('.tl-icon').textContent = meta.icon;
    li.querySelector('.tl-type').textContent = meta.label;
    const clock = new Date().toLocaleTimeString('en-GB', { hour12: false });
    li.querySelector('.tl-time').textContent = evt && evt.tick != null ? `${clock} · tick ${evt.tick}` : clock;
    li.querySelector('.tl-msg').textContent = message;

    if (evt && evt.id != null && (type === 'reroute' || type === 'congestion')) {
        li.dataset.eventId = evt.id;
        li.classList.add('clickable');
        li.tabIndex = 0;
        li.setAttribute('role', 'button');
        li.setAttribute('aria-pressed', 'false');
        li.title = type === 'reroute' ? 'Show this flow\'s old path, new path and trigger link'
                                      : 'Highlight these links and list the reroutes they triggered';
        li.addEventListener('click', () => toggleSelection(evt.id));
        li.addEventListener('keydown', e => {
            if (e.target === li && (e.key === 'Enter' || e.key === ' ')) {
                e.preventDefault();
                toggleSelection(evt.id);
            }
        });
    }

    const following = list.scrollTop <= 4;
    const before = list.scrollHeight;
    list.insertBefore(li, list.firstChild);
    if (!following) {
        list.scrollTop += list.scrollHeight - before;  // keep the entries being read in place
        unseenEvents++;
        const latest = document.getElementById('timeline-latest');
        latest.textContent = `↑ ${unseenEvents} new`;
        latest.hidden = false;
    }
    while (list.children.length > TIMELINE_MAX) {
        const last = list.lastElementChild;
        if (last.dataset.eventId && Number(last.dataset.eventId) === selectedId) clearSelection();
        last.remove();
    }
}

function clearLog() {
    clearSelection();
    document.getElementById('event-log').innerHTML = '';
    unseenEvents = 0;
    document.getElementById('timeline-latest').hidden = true;
}

function entryFor(id) {
    return document.querySelector(`#event-log [data-event-id="${id}"]`);
}

function toggleSelection(id) {
    if (selectedId === id) clearSelection();
    else selectEvent(id);
}

function selectEvent(id) {
    clearSelection();
    const evt = eventStore.get(id);
    if (!evt) return;
    selectedId = id;
    clearTransient();  // the pinned highlight is shown on its own
    const li = entryFor(id);
    if (li) {
        li.classList.add('selected');
        li.setAttribute('aria-pressed', 'true');
        if (evt.type === 'congestion') showRelatedReroutes(evt, li);
    }
    drawSelection();
}

function clearSelection() {
    if (selectedId != null) {
        const li = entryFor(selectedId);
        if (li) {
            li.classList.remove('selected');
            li.setAttribute('aria-pressed', 'false');
            const details = li.querySelector('.tl-details');
            if (details) details.remove();
        }
    }
    selectedId = null;
    if (selectionLayer) selectionLayer.selectAll('*').remove();
}

// Reroutes whose trigger link is one of this event's links, within RELATED_TICKS ticks after it.
function relatedReroutes(evt) {
    const links = new Set(evt.links.map(d => d.link));
    return rerouteEvents.filter(r => r.run === evt.run && links.has(r.trigger_link) &&
        r.tick >= evt.tick && r.tick <= evt.tick + RELATED_TICKS);
}

function showRelatedReroutes(evt, li) {
    const related = relatedReroutes(evt);
    const details = document.createElement('div');
    details.className = 'tl-details';
    const stillOpen = (previous ? previous.tick : evt.tick) < evt.tick + RELATED_TICKS;
    if (!related.length) {
        details.textContent = `No reroute was triggered from ${evt.links.length === 1 ? 'this link' : 'these links'} ` +
            `within ${RELATED_TICKS} ticks${stillOpen ? ' (so far)' : ''}` +
            (activeController === 'ecmp' ? '; ECMP never reroutes.' : '.');
    } else {
        details.textContent = `Reroutes triggered from ${evt.links.length === 1 ? 'this link' : 'these links'} ` +
            `within ${RELATED_TICKS} ticks (click to show the paths):`;
        const ul = document.createElement('ul');
        related.forEach(r => {
            const item = document.createElement('li');
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'tl-related';
            button.textContent = `tick ${r.tick}: flow ${r.flow_id} off ${linkLabel(r.trigger_link)} → ${switchPath(r.new_path)}`;
            button.addEventListener('click', e => { e.stopPropagation(); selectEvent(r.id); });
            button.addEventListener('keydown', e => e.stopPropagation());
            item.appendChild(button);
            ul.appendChild(item);
        });
        details.appendChild(ul);
    }
    li.appendChild(details);
}

// ── Topology highlights ────────────────────────────────────────
function drawSegment(g, id, cls) {
    const p = linkGeom[id];
    if (!p) return;
    g.append('line').attr('class', cls).attr('x1', p.x1).attr('y1', p.y1).attr('x2', p.x2).attr('y2', p.y2);
}

function drawBadge(g, id, text) {
    const p = linkGeom[id];
    if (!p) return;
    const badge = g.append('g').attr('class', 'hl-badge')
        .attr('transform', `translate(${(p.x1 + p.x2) / 2}, ${(p.y1 + p.y2) / 2})`);
    badge.append('circle').attr('r', 7.5);
    badge.append('text').text(text);
}

// Old path dashed red, new path cyan, trigger link dark halo + "!" badge.
function drawReroute(layer, evt) {
    const g = layer.append('g').attr('class', 'reroute-highlight');
    if (evt.trigger_link) drawSegment(g, evt.trigger_link, 'hl-trigger-halo');
    pathLinks(evt.old_path || []).forEach(id => drawSegment(g, id, 'hl-old-path'));
    pathLinks(evt.new_path || []).forEach(id => drawSegment(g, id, 'hl-new-path'));
    if (evt.trigger_link) drawBadge(g, evt.trigger_link, '!');
    return g;
}

// New reroutes flash briefly, unless the user has pinned one (the timeline still logs them).
function flashReroute(evt) {
    if (!transientLayer || selectedId != null) return;
    const shown = transientLayer.selectAll('.reroute-highlight').nodes();
    shown.slice(0, Math.max(0, shown.length - TRANSIENT_MAX + 1)).forEach(node => node.remove());
    const g = drawReroute(transientLayer, evt).classed('hl-transient', true);
    setTimeout(() => g.classed('fading', true), TRANSIENT_MS - 400);
    setTimeout(() => g.remove(), TRANSIENT_MS);
}

function clearTransient() {
    if (transientLayer) transientLayer.selectAll('*').remove();
}

function drawSelection() {
    if (!selectionLayer) return;
    selectionLayer.selectAll('*').remove();
    const evt = selectedId != null ? eventStore.get(selectedId) : null;
    if (!evt) return;
    if (evt.type === 'reroute') {
        drawReroute(selectionLayer, evt);
    } else if (evt.type === 'congestion') {
        const g = selectionLayer.append('g');
        evt.links.forEach(d => drawSegment(g, d.link, 'hl-selected-link'));
        evt.links.forEach(d => drawBadge(g, d.link, '!'));
    }
}

// ── Health Score ───────────────────────────────────────────────
// Fabric health: 100 − 10 × fabric-link drop % of offered packets over the last HEALTH_WINDOW
// ticks (10% fabric loss = 0). Host-link drops are left out: no path choice avoids them.
function fabricHealth(window) {
    const fabric = window.reduce((s, t) => s + t.fabric, 0);
    const offered = window.reduce((s, t) => s + t.offered, 0);
    const dropPct = offered > 0 ? 100 * fabric / offered : 0;
    return Math.max(0, Math.min(100, Math.round(100 - 10 * dropPct)));
}

function updateHealthScore(fabricThisTick, offeredThisTick) {
    healthWindow.push({ fabric: fabricThisTick, offered: offeredThisTick });
    if (healthWindow.length > HEALTH_WINDOW) healthWindow.shift();
    const health = fabricHealth(healthWindow);

    const scoreEl = document.getElementById('health-score');
    const ringEl = document.getElementById('health-ring-fill');
    scoreEl.textContent = health;
    ringEl.style.strokeDashoffset = 113.1 * (1 - health / 100);  // circumference 2π·18
    const color = health > 70 ? '#38a169' : health > 40 ? '#d69e2e' : '#e53e3e';
    ringEl.style.stroke = color;
    scoreEl.style.color = color;
}

// ── Prediction Panel ───────────────────────────────────────────
let heatmapCells = {};

function updatePredictionPanel(predictions) {
    const container = document.getElementById('prediction-heatmap');
    const statusEl = document.getElementById('gnn-status');
    const ids = Object.keys(predictions);

    if (ids.length === 0) {
        statusEl.textContent = modelLoaded ? 'Collecting data...' : 'No model loaded';
        return;
    }

    statusEl.textContent = 'Active';
    statusEl.style.color = '#2f855a';

    // Build labelled cells once, grouped by layer: core–agg, then agg–edge (by pod).
    if (Object.keys(heatmapCells).length !== ids.length || ids.some(id => !heatmapCells[id])) {
        container.innerHTML = '';
        heatmapCells = {};
        const sorted = [...ids].sort((a, b) => {
            const [a0, a1] = linkEnds(a), [b0, b1] = linkEnds(b);
            return nodeRank(a0) - nodeRank(b0) || a1.localeCompare(b1, undefined, { numeric: true })
                || a0.localeCompare(b0, undefined, { numeric: true });
        });
        let group = null;
        sorted.forEach(id => {
            const g = nodeRank(linkEnds(id)[0]) === 0 ? 'Core – Aggregation' : 'Aggregation – Edge';
            if (g !== group) {
                group = g;
                const label = document.createElement('div');
                label.className = 'heatmap-group-label';
                label.textContent = g;
                container.appendChild(label);
            }
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';
            cell.textContent = linkLabel(id);
            cell.setAttribute('data-link-id', id);
            container.appendChild(cell);
            heatmapCells[id] = cell;
        });
    }

    let maxProb = -1, maxId = null, avgProb = 0, riskCount = 0;
    ids.forEach(id => {
        const prob = predictions[id];
        const cell = heatmapCells[id];
        const predicted = liveThreshold !== null && prob > liveThreshold;
        cell.style.background = predictionColor(prob);
        cell.style.color = prob > 0.55 ? '#ffffff' : '#2d3748';
        cell.classList.toggle('predicted', predicted);
        cell.title = `${linkLabel(id)} (${id}): ${pct(prob)} chance of congestion in 5 ticks`;
        if (prob > maxProb) { maxProb = prob; maxId = id; }
        avgProb += prob;
        if (predicted) riskCount++;
    });
    avgProb /= ids.length;

    document.getElementById('pred-avg').textContent = pct(avgProb);
    document.getElementById('pred-max').textContent = pct(maxProb);
    document.getElementById('pred-max-link').textContent = maxId ? linkLabel(maxId) : '';
    document.getElementById('pred-risk-count').textContent = riskCount;
    document.getElementById('pred-risk-threshold').textContent =
        liveThreshold === null ? '' : `p > ${liveThreshold.toFixed(3)}`;
}

function predictionColor(prob) {
    // Sequential purple, matching the predicted-congestion outline on the topology.
    return d3.interpolatePurples(0.08 + 0.87 * Math.min(1, Math.max(0, prob)));
}

// ── Sparklines ─────────────────────────────────────────────────
const sparkCanvases = {};

function initSparklines() {
    SPARK_KEYS.forEach(key => {
        const canvas = document.getElementById(`spark-${key}`);
        if (canvas) sparkCanvases[key] = canvas.getContext('2d');
    });
}

function pushSparkData(key, value) {
    sparkData[key].push(value);
    if (sparkData[key].length > SPARK_MAX) sparkData[key].shift();
}

function drawAllSparklines() {
    const colors = {
        flows: '#00b5d8', packets: '#667eea', fabric: '#e53e3e', host: '#718096',
        droprate: '#dd6b20', reroutes: '#086f83', predictions: '#805ad5',
    };
    Object.keys(sparkCanvases).forEach(key => drawSparkline(sparkCanvases[key], sparkData[key], colors[key]));
}

function drawSparkline(ctx, data, color) {
    if (!ctx || data.length < 2) return;
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;
    ctx.clearRect(0, 0, w, h);
    const max = Math.max(...data) || 1;
    const min = Math.min(...data);
    const range = max - min || 1;

    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    data.forEach((val, i) => {
        const x = (i / (data.length - 1)) * w;
        const y = h - ((val - min) / range) * (h - 4) - 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.globalAlpha = 0.08;
    ctx.fillStyle = color;
    ctx.fill();
    ctx.globalAlpha = 1;
}

// ── D3.js Topology Rendering ───────────────────────────────────
function renderTopology(data) {
    const container = document.getElementById('topology-container');
    const svg = d3.select('#topology-svg');
    svg.selectAll('*').remove();
    linkElements = {};
    linkGeom = {};
    predOutlines = {};

    const width = container.clientWidth;
    const height = container.clientHeight;
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    let tooltip = document.querySelector('.tooltip');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        document.body.appendChild(tooltip);
    }

    const layerPositions = computeLayerLayout(data, width, height);

    // Layers, bottom to top: predicted outlines, links, transient reroutes, pinned selection, nodes.
    const predGroup = svg.append('g').attr('class', 'pred-outlines');
    const linkGroup = svg.append('g').attr('class', 'links');
    transientLayer = svg.append('g').attr('class', 'highlight-layer transient');
    selectionLayer = svg.append('g').attr('class', 'highlight-layer selection');

    data.links.forEach(link => {
        const sourcePos = layerPositions[link.source];
        const targetPos = layerPositions[link.target];
        if (!sourcePos || !targetPos) return;

        const id = linkId(link.source, link.target);
        linkGeom[id] = { x1: sourcePos.x, y1: sourcePos.y, x2: targetPos.x, y2: targetPos.y };
        const coords = line => line
            .attr('x1', sourcePos.x).attr('y1', sourcePos.y)
            .attr('x2', targetPos.x).attr('y2', targetPos.y);

        if (!link.is_host_link) {
            predOutlines[id] = coords(predGroup.append('line'))
                .attr('class', 'pred-outline')
                .attr('stroke-width', 11);
        }

        const line = coords(linkGroup.append('line'))
            .attr('class', 'link-line')
            .attr('stroke', '#1a2638')
            .attr('stroke-width', baseWidth(link))
            .attr('stroke-opacity', link.is_host_link ? 0.25 : 0.5)
            .attr('data-link-id', id);
        linkElements[id] = line;

        line.on('mouseover', function(event) {
            d3.select(this).attr('stroke-width', parseFloat(d3.select(this).attr('stroke-width')) + 3);
            showLinkTooltip(event, link, id);
        }).on('mousemove', function(event) {
            moveTooltip(event);
        }).on('mouseout', function() {
            styleLink(id, link);
            hideTooltip();
        });
        line.datum(link);
    });

    const nodeGroup = svg.append('g').attr('class', 'nodes');
    data.nodes.forEach(node => {
        const pos = layerPositions[node.id];
        if (!pos) return;

        const g = nodeGroup.append('g')
            .attr('class', 'node-group')
            .attr('transform', `translate(${pos.x}, ${pos.y})`);
        const radius = getNodeRadius(node.type);
        const color = getNodeColor(node.type);

        if (node.type !== 'host') {
            g.append('circle').attr('class', 'node-glow').attr('r', radius + 6).attr('fill', color).attr('opacity', 0.08);
        }
        g.append('circle')
            .attr('class', 'node-circle')
            .attr('r', radius)
            .attr('fill', color)
            .attr('stroke', 'rgba(255,255,255,0.15)')
            .attr('stroke-width', node.type === 'host' ? 0.5 : 1.5)
            .attr('data-node-id', node.id);
        if (node.type !== 'host') {
            g.append('text').attr('class', 'node-label').attr('dy', radius + 13).text(shortName(node.id));
        }

        g.on('mouseover', function(event) {
            d3.select(this).select('.node-circle').attr('stroke-width', 3).attr('stroke', color);
            showNodeTooltip(event, node);
        }).on('mousemove', function(event) {
            moveTooltip(event);
        }).on('mouseout', function() {
            d3.select(this).select('.node-circle')
                .attr('stroke-width', node.type === 'host' ? 0.5 : 1.5)
                .attr('stroke', 'rgba(255,255,255,0.15)');
            hideTooltip();
        });
    });

    // Re-apply the latest state (e.g. after a resize).
    if (Object.keys(latestLinkData).length) updateTopologyLinks(latestLinkData);
    updatePredictionOutlines(latestPredictions);
    drawSelection();
}

function computeLayerLayout(data, width, height) {
    const positions = {};
    const pad = { x: 50, y: 36 };
    const usableW = width - 2 * pad.x;
    const usableH = height - 2 * pad.y;

    const layers = { 0: [], 1: [], 2: [], 3: [] };
    data.nodes.forEach(n => {
        const layer = n.layer !== undefined ? n.layer : 3;
        if (!layers[layer]) layers[layer] = [];
        layers[layer].push(n);
    });

    const layerY = {
        0: pad.y + 16,
        1: pad.y + usableH * 0.3,
        2: pad.y + usableH * 0.6,
        3: pad.y + usableH * 0.92,
    };

    const k = 4;
    Object.entries(layers).forEach(([layer, nodes]) => {
        const y = layerY[layer];
        if (nodes.length === 0) return;
        nodes.sort((a, b) => (a.pod !== b.pod ? a.pod - b.pod : (a.index || 0) - (b.index || 0)));

        if (parseInt(layer) >= 1) {
            // Aggregation, edge and hosts: grouped by pod.
            const perPod = nodes.length / k;
            const podWidth = usableW / k;
            nodes.forEach((n, i) => {
                const podIdx = Math.floor(i / perPod);
                const inPodIdx = i % perPod;
                const spacing = podWidth / (perPod + 1);
                positions[n.id] = { x: pad.x + podIdx * podWidth + spacing * (inPodIdx + 1), y };
            });
        } else {
            // Core: spread evenly
            const spacing = usableW / (nodes.length + 1);
            nodes.forEach((n, i) => { positions[n.id] = { x: pad.x + spacing * (i + 1), y }; });
        }
    });
    return positions;
}

function getNodeRadius(type) {
    return { core: 14, aggregation: 11, edge: 10, host: 5 }[type] || 6;
}

function getNodeColor(type) {
    return { core: '#00d2ff', aggregation: '#7b68ee', edge: '#4fd1c5', host: '#5a6578' }[type] || '#94a3b8';
}

// ── Update Topology Links ──────────────────────────────────────
function baseWidth(link) {
    return link.is_host_link ? 1.5 : 2.5;
}

// Current state of one link: solid red when congested now (served_or_queued label),
// otherwise coloured by offered load.
function styleLink(id, link) {
    const line = linkElements[id];
    const data = latestLinkData[id];
    if (!line || !data) return;
    const util = Math.min(1, data.utilization);
    line.classed('congested-now', data.congested)
        .attr('stroke', utilizationColor(util))
        .attr('stroke-opacity', data.congested ? 1 : 0.45 + util * 0.55)
        .attr('stroke-width', baseWidth(link) + (data.congested ? 3.5 : util * 3));
}

function updateTopologyLinks(links) {
    if (!links) return;
    latestLinkData = links;
    Object.keys(links).forEach(id => {
        const line = linkElements[id];
        if (line) styleLink(id, line.datum());
    });
}

// Predicted congestion: calibrated probability above the live alert threshold.
function updatePredictionOutlines(predictions) {
    Object.entries(predOutlines).forEach(([id, outline]) => {
        const p = predictions[id];
        outline.classed('active', p !== undefined && liveThreshold !== null && p > liveThreshold);
    });
}

// Offered load, green → amber → orange. Red is reserved for current congestion.
function utilizationColor(util) {
    if (util < 1 / 3) return d3.interpolateRgb('#c6e8d2', '#48bb78')(util * 3);
    if (util < 2 / 3) return d3.interpolateRgb('#48bb78', '#ecc94b')((util - 1 / 3) * 3);
    return d3.interpolateRgb('#ecc94b', '#dd6b20')(Math.min(1, (util - 2 / 3) * 3));
}

// ── Tooltips ───────────────────────────────────────────────────
function tooltipRow(label, value, color) {
    const style = color ? ` style="color:${color}"` : '';
    return `<div class="tooltip-row"><span class="tooltip-label">${label}</span><span class="tooltip-value"${style}>${value}</span></div>`;
}

function showLinkTooltip(event, link, id) {
    const tooltip = document.querySelector('.tooltip');
    if (!tooltip) return;
    const data = latestLinkData[id];
    let html = `<div class="tooltip-title">${linkLabel(id)} <span class="tooltip-label">${id}</span></div>`;
    if (data) {
        const util = Math.min(1, data.utilization);
        const utilColor = utilizationColor(util);
        html += tooltipRow('Offered load', pct(data.utilization), utilColor);
        html += `<div class="tooltip-bar"><div class="tooltip-bar-fill" style="width:${util * 100}%;background:${utilColor}"></div></div>`;
        html += tooltipRow('Congestion score', data.score.toFixed(3));
        html += tooltipRow('Congested now', data.congested ? 'yes' : 'no', data.congested ? 'var(--link-congested)' : null);
        const p = latestPredictions[id];
        if (p !== undefined) {
            const predicted = liveThreshold !== null && p > liveThreshold;
            html += tooltipRow('P(congested at t+5)', pct(p), predicted ? 'var(--link-predicted)' : null);
        }
        html += tooltipRow('Queue depth', data.queue_depth);
        html += tooltipRow('Drops (this tick)', data.drops, data.drops > 0 ? '#c53030' : null);
        html += tooltipRow('Latency', `${data.latency.toFixed(1)} μs`);
    } else {
        html += tooltipRow('Status', 'Idle');
    }
    tooltip.innerHTML = html;
    positionTooltip(event);
    tooltip.classList.add('visible');
}

function showNodeTooltip(event, node) {
    const tooltip = document.querySelector('.tooltip');
    if (!tooltip) return;
    const typeNames = { core: 'Core switch', aggregation: 'Aggregation switch', edge: 'Edge (ToR) switch', host: 'Server host' };
    let html = `<div class="tooltip-title">${typeNames[node.type] || node.type}</div>`;
    html += tooltipRow('ID', node.id);
    if (node.pod >= 0) html += tooltipRow('Pod', node.pod);
    html += tooltipRow('Layer', node.layer);
    tooltip.innerHTML = html;
    positionTooltip(event);
    tooltip.classList.add('visible');
}

function positionTooltip(event) {
    const tooltip = document.querySelector('.tooltip');
    tooltip.style.left = `${Math.min(event.clientX + 15, window.innerWidth - 300)}px`;
    tooltip.style.top = `${Math.min(event.clientY - 10, window.innerHeight - 230)}px`;
}

function moveTooltip(event) {
    positionTooltip(event);
}

function hideTooltip() {
    const tooltip = document.querySelector('.tooltip');
    if (tooltip) tooltip.classList.remove('visible');
}

// ── Time-Series Data ───────────────────────────────────────────
function updateTimeSeries(links, metrics, fabricThisTick, hostThisTick) {
    let totalUtil = 0, maxUtil = 0, count = 0;
    Object.values(links).forEach(data => {
        totalUtil += data.utilization;
        maxUtil = Math.max(maxUtil, data.utilization);
        count++;
    });

    metricsHistory.ticks.push(metrics.tick);
    metricsHistory.avgUtilization.push(count > 0 ? totalUtil / count : 0);
    metricsHistory.maxUtilization.push(maxUtil);
    metricsHistory.flows.push(metrics.active_flows);
    metricsHistory.fabricDropsPerTick.push(fabricThisTick);
    metricsHistory.hostDropsPerTick.push(hostThisTick);

    // Trim in place: the charts hold references to these arrays.
    Object.values(metricsHistory).forEach(series => {
        if (series.length > MAX_HISTORY) series.splice(0, series.length - MAX_HISTORY);
    });
}

// ── Charts ─────────────────────────────────────────────────────
function initCharts() {
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: {
                labels: {
                    color: '#4a5568',
                    font: { family: FONT_MAIN, size: 11, weight: '500' },
                    boxWidth: 10,
                    boxHeight: 3,
                    padding: 10,
                    usePointStyle: true,
                    pointStyle: 'rectRounded',
                },
                position: 'top',
            },
            tooltip: {
                backgroundColor: 'rgba(26, 32, 44, 0.92)',
                titleFont: { family: FONT_MAIN, size: 11 },
                bodyFont: { family: FONT_MONO, size: 10 },
                padding: 9,
                cornerRadius: 6,
                boxPadding: 4,
            },
        },
        scales: {
            x: { display: false, grid: { display: false } },
            y: {
                grid: { color: 'rgba(0,0,0,0.05)' },
                ticks: { color: '#4a5568', font: { size: 10, family: FONT_MONO }, maxTicksLimit: 5 },
                border: { display: false },
            },
        },
        elements: {
            point: { radius: 0, hoverRadius: 3 },
            line: { tension: 0.4, borderWidth: 1.5 },
        },
    };

    utilizationChart = new Chart(document.getElementById('chart-utilization'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Avg offered load', data: [], borderColor: '#00b5d8', backgroundColor: 'rgba(0, 181, 216, 0.06)', fill: true },
                { label: 'Peak offered load', data: [], borderColor: '#e53e3e', backgroundColor: 'rgba(229, 62, 62, 0.04)', fill: true, borderDash: [4, 3] },
            ],
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: {
                    // Offered load can exceed 100%; auto-scale above a 100% floor.
                    ...commonOptions.scales.y,
                    min: 0,
                    suggestedMax: 1,
                    ticks: { ...commonOptions.scales.y.ticks, callback: v => `${(v * 100).toFixed(0)}%` },
                },
            },
        },
    });

    dropsChart = new Chart(document.getElementById('chart-drops'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                { label: 'Fabric drops / tick', data: [], borderColor: '#e53e3e', backgroundColor: 'rgba(229, 62, 62, 0.06)', fill: true, yAxisID: 'y' },
                { label: 'Host-link drops / tick', data: [], borderColor: '#718096', borderDash: [4, 3], fill: false, yAxisID: 'y' },
                { label: 'Active flows', data: [], borderColor: '#319795', backgroundColor: 'rgba(49, 151, 149, 0.04)', fill: true, yAxisID: 'y1' },
            ],
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: { ...commonOptions.scales.y, position: 'left' },
                y1: {
                    position: 'right',
                    grid: { display: false },
                    ticks: { color: '#2c7a7b', font: { size: 10, family: FONT_MONO }, maxTicksLimit: 5 },
                    border: { display: false },
                },
            },
        },
    });
}

function updateCharts() {
    if (!utilizationChart || !dropsChart) return;
    utilizationChart.data.labels = metricsHistory.ticks;
    utilizationChart.data.datasets[0].data = metricsHistory.avgUtilization;
    utilizationChart.data.datasets[1].data = metricsHistory.maxUtilization;
    utilizationChart.update('none');

    dropsChart.data.labels = metricsHistory.ticks;
    dropsChart.data.datasets[0].data = metricsHistory.fabricDropsPerTick;
    dropsChart.data.datasets[1].data = metricsHistory.hostDropsPerTick;
    dropsChart.data.datasets[2].data = metricsHistory.flows;
    dropsChart.update('none');
}

// ── Sliders ────────────────────────────────────────────────────
// Sets the live GNN alert threshold; `send` pushes it to the server (guarded_gnn uses it).
function setThreshold(value, { send = true } = {}) {
    liveThreshold = Number(value);
    const slider = document.getElementById('threshold-slider');
    if (Math.abs(Number(slider.value) - liveThreshold) > 1e-9) slider.value = liveThreshold;
    document.getElementById('threshold-value').textContent = liveThreshold.toFixed(3);
    if (send) {
        lastLocalThresholdAt = performance.now();
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: 'threshold', value: liveThreshold }));
        }
    }
    refreshPredictionViews();
}

function formatSpeed(v) {
    return `${v.toFixed(v < 1 ? 2 : 1)}x`;
}

function setupSliders() {
    const thresholdSlider = document.getElementById('threshold-slider');
    thresholdSlider.addEventListener('input', () => setThreshold(thresholdSlider.value));
    document.getElementById('threshold-reset').addEventListener('click', () => {
        if (alertThreshold != null) setThreshold(alertThreshold);
    });

    // Simulation speed: left = slower, right = faster, applied immediately.
    const speedSlider = document.getElementById('speed-slider');
    const speedValue = document.getElementById('speed-value');
    const applySpeed = () => {
        const speed = parseFloat(speedSlider.value);
        speedValue.textContent = formatSpeed(speed);
        speedSlider.setAttribute('aria-valuetext', `${formatSpeed(speed)} speed`);
        fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ speed }),
        }).catch(() => {});
    };
    speedSlider.addEventListener('input', applySpeed);
    speedValue.textContent = formatSpeed(parseFloat(speedSlider.value));
}

// ── Utilities ──────────────────────────────────────────────────
function formatNumber(n) {
    if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
    return n.toString();
}

// ── Window Resize ──────────────────────────────────────────────
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        if (topologyData) renderTopology(topologyData);
    }, 300);
});
