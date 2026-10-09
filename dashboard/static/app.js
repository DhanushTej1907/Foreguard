/**
 * NeuralFabric — GNN Congestion Pre-emption Dashboard
 * =====================================================
 * Real-time D3.js topology + Chart.js metrics + WebSocket live updates.
 *
 * Link IDs are "u--v" with u < v (one per physical link), as sent by the server.
 * Topology link states:
 *   congested now  — served_or_queued label from the server (solid red)
 *   predicted      — calibrated GNN probability > live alert threshold (pulsing dashed outline)
 *   rerouted path  — a flow's new path, highlighted for REROUTE_HIGHLIGHT_MS
 */

// ── State ──────────────────────────────────────────────────────
let ws = null;
let topologyData = null;
let linkElements = {};     // linkId -> base line
let predOutlines = {};     // linkId -> predicted-congestion outline (fabric links only)
let rerouteLines = {};     // linkId -> reroute highlight line
let rerouteTimers = {};
let nodeElements = {};
let isRunning = false;

let liveThreshold = null;   // GNN alert threshold (slider), synced from the server
let alertThreshold = null;  // calibrated threshold stored in the checkpoint
let lastLocalThresholdAt = -Infinity;
let modelLoaded = false;
let activeController = null;

const REROUTE_HIGHLIGHT_MS = 1000;
const CONTROLLER_NAMES = {
    ecmp: 'ECMP',
    guarded_gnn: 'Guarded GNN',
    guarded_score: 'Guarded score',
};

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

// Charts
let utilizationChart = null;
let dropsChart = null;

// ── Initialization ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initSparklines();
    setupSliders();
    setupControllerSelect();
    setupWorkloadSelect();
    addLog('Dashboard ready. Click "Start Engine" to begin.', 'info');
});

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
        addLog('Connected to simulation engine', 'success');
    };

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            if (msg.type === 'topology') {
                topologyData = msg.data;
                renderTopology(msg.data);
                addLog(`Topology loaded: ${msg.data.nodes.length} nodes, ${msg.data.links.length} links`, 'info');
            } else if (msg.type === 'update') {
                handleUpdate(msg);
            } else if (msg.type === 'error') {
                addLog(msg.message, 'danger');
            }
        } catch (e) {
            console.error('Message parse error:', e);
        }
    };

    ws.onclose = () => {
        document.getElementById('status-badge').classList.remove('connected');
        document.getElementById('status-text').textContent = 'Offline';
        addLog('Connection lost', 'warning');
        setTimeout(() => { if (isRunning) connectWebSocket(); }, 3000);
    };

    ws.onerror = () => addLog('Connection error', 'danger');
}

// ── Simulation Controls ────────────────────────────────────────
async function startSimulation() {
    try {
        const resp = await fetch('/api/start', { method: 'POST' });
        const data = await resp.json();

        if (data.status === 'started' || data.status === 'already_running') {
            setRunningUI(true);
            if (data.status === 'started') resetSeries();

            if (data.topology) {
                topologyData = data.topology;
                renderTopology(data.topology);
            }
            if (data.status === 'started') {
                // A new run resets the live threshold to the checkpoint's calibrated value.
                syncControllerState(data, { forceThreshold: true });
                addLog(data.model_loaded
                    ? `Simulation engine started · ${data.workload} · ${data.controller_label} · calibrated alert threshold ${data.alert_threshold.toFixed(3)}`
                    : `Simulation engine started · ${data.workload} · no schema-3 checkpoint, so ECMP only and no forecast`, 'success');
            }

            connectWebSocket();
        }
    } catch (e) {
        addLog(`Failed to start: ${e.message}`, 'danger');
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
        addLog('Simulation stopped', 'warning');
    } catch (e) {
        addLog(`Failed to stop: ${e.message}`, 'danger');
    }
}

async function injectSpike() {
    try {
        const resp = await fetch('/api/spike', { method: 'POST' });
        const data = await resp.json();
        addLog(`Traffic spike injected — ${data.new_flows} hotspot/incast flows`, 'danger');

        // Visual feedback
        const btn = document.getElementById('btn-spike');
        btn.style.transform = 'scale(0.95)';
        setTimeout(() => { btn.style.transform = ''; }, 200);
    } catch (e) {
        addLog(`Spike injection failed: ${e.message}`, 'danger');
    }
}

// ── Workload Selector ──────────────────────────────────────────
let activeWorkload = 'stride';

function setupWorkloadSelect() {
    const select = document.getElementById('workload-select');
    select.addEventListener('change', () => switchWorkload(select.value));
}

// Before the first start this only selects the workload; afterwards the server restarts the
// simulation with it, keeping the controller and the live threshold.
async function switchWorkload(name) {
    const select = document.getElementById('workload-select');
    try {
        const resp = await fetch('/api/workload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ workload: name }),
        });
        const data = await resp.json();
        if (!resp.ok) {
            const detail = typeof data.detail === 'string' ? data.detail : resp.statusText;
            addLog(`Workload switch refused: ${detail}`, 'danger');
            select.value = activeWorkload;
            return;
        }
        if (data.status === 'restarted') {
            resetSeries();
            setRunningUI(true);
            syncControllerState(data);
            connectWebSocket();
            addLog(`Workload → ${name}: simulation restarted (controller ${data.controller_label} kept)`, 'info');
        } else {
            activeWorkload = name;
            document.getElementById('workload-active').textContent = name;
            addLog(`Workload set to ${name}; it runs when the engine starts`, 'info');
        }
    } catch (e) {
        addLog(`Workload switch failed: ${e.message}`, 'danger');
        select.value = activeWorkload;
    }
}

// ── Controller Selector ────────────────────────────────────────
function setupControllerSelect() {
    const select = document.getElementById('controller-select');
    select.addEventListener('change', () => switchController(select.value));
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
            const detail = typeof data.detail === 'string' ? data.detail : resp.statusText;
            addLog(`Controller switch refused: ${detail}`, 'danger');
            if (activeController) select.value = activeController;
            return;
        }
        clearRerouteHighlights();
        syncControllerState(data);
        addLog(`Controller → ${data.controller_label}. All flows restored to ECMP paths; counters reset.`, 'info');
    } catch (e) {
        addLog(`Controller switch failed: ${e.message}`, 'danger');
        if (activeController) select.value = activeController;
    }
}

// Applies the controller/threshold fields the server sends with every update
// (and with the start and switch responses).
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
    clearRerouteHighlights();
    updateCharts();
}

function handleUpdate(msg) {
    const { links, predictions, metrics } = msg;

    // A tick counter going backwards means the server restarted the run (possibly from another viewer).
    if (previous && metrics.tick < previous.tick) resetSeries();

    syncControllerState(msg);

    // Update tick counter
    document.getElementById('tick-value').textContent = metrics.tick;

    // Update metric cards
    document.getElementById('val-flows').textContent = metrics.active_flows;
    document.getElementById('val-packets').textContent = formatNumber(metrics.total_packets);
    document.getElementById('val-fabric-drops').textContent = formatNumber(metrics.fabric_drops);
    document.getElementById('val-host-drops').textContent = formatNumber(metrics.host_drops);
    document.getElementById('val-droprate').textContent = `${metrics.drop_rate}%`;
    document.getElementById('val-reroutes').textContent = formatNumber(metrics.total_reroutes);

    latestPredictions = predictions || {};
    const alertCount = countAlerts(latestPredictions);
    document.getElementById('val-predictions').textContent = alertCount;

    // Update sparklines
    pushSparkData('flows', metrics.active_flows);
    pushSparkData('packets', metrics.total_packets);
    pushSparkData('fabric', metrics.fabric_drops);
    pushSparkData('host', metrics.host_drops);
    pushSparkData('droprate', metrics.drop_rate);
    pushSparkData('reroutes', metrics.total_reroutes);
    pushSparkData('predictions', alertCount);
    drawAllSparklines();

    // Update topology link colors and the forecast views
    updateTopologyLinks(links);
    refreshPredictionViews();

    // Per-tick deltas of the cumulative counters
    const delta = key => previous ? Math.max(0, metrics[key] - previous[key]) : 0;
    const fabricThisTick = delta('fabric_drops');
    const hostThisTick = delta('host_drops');
    const offeredThisTick = delta('total_packets');
    previous = metrics;
    updateTimeSeries(links, metrics, fabricThisTick, hostThisTick);

    // Update charts (throttled)
    if (metrics.tick % 3 === 0) updateCharts();

    // Update health score
    updateHealthScore(fabricThisTick, offeredThisTick);

    // Reroute events from server
    (msg.reroute_events || []).forEach(evt => {
        highlightPath(evt.new_path);
        const who = CONTROLLER_NAMES[evt.controller] || evt.controller;
        const path = evt.new_path.map(shortName).join(' › ');
        addLog(`${who} rerouted flow ${evt.flow_id} (${shortName(evt.src)} → ${shortName(evt.dst)}) via ${path}`, 'reroute');
    });
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

    // Ring: circumference = 2*π*18 ≈ 113.1
    const offset = 113.1 * (1 - health / 100);
    ringEl.style.strokeDashoffset = offset;

    // Color
    if (health > 70) {
        ringEl.style.stroke = '#48bb78';
        scoreEl.style.color = '#48bb78';
    } else if (health > 40) {
        ringEl.style.stroke = '#f6ad55';
        scoreEl.style.color = '#f6ad55';
    } else {
        ringEl.style.stroke = '#e94560';
        scoreEl.style.color = '#e94560';
    }
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
    statusEl.style.color = '#48bb78';
    statusEl.style.borderColor = 'rgba(72, 187, 120, 0.3)';

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
        if (canvas) {
            sparkCanvases[key] = canvas.getContext('2d');
        }
    });
}

function pushSparkData(key, value) {
    sparkData[key].push(value);
    if (sparkData[key].length > SPARK_MAX) sparkData[key].shift();
}

function drawAllSparklines() {
    const colors = {
        flows: '#00d2ff',
        packets: '#667eea',
        fabric: '#e94560',
        host: '#718096',
        droprate: '#f6ad55',
        reroutes: '#7b68ee',
        predictions: '#805ad5',
    };

    Object.keys(sparkCanvases).forEach(key => {
        drawSparkline(sparkCanvases[key], sparkData[key], colors[key]);
    });
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

    // Fill under curve
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
    predOutlines = {};
    rerouteLines = {};
    Object.values(rerouteTimers).forEach(clearTimeout);
    rerouteTimers = {};
    nodeElements = {};

    const width = container.clientWidth;
    const height = container.clientHeight;

    svg.attr('viewBox', `0 0 ${width} ${height}`);

    // Create tooltip if needed
    let tooltip = document.querySelector('.tooltip');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.className = 'tooltip';
        document.body.appendChild(tooltip);
    }

    // Layout
    const layerPositions = computeLayerLayout(data, width, height);

    // Layers, bottom to top: predicted outlines, links, reroute highlights, nodes.
    const predGroup = svg.append('g').attr('class', 'pred-outlines');
    const linkGroup = svg.append('g').attr('class', 'links');
    const rerouteGroup = svg.append('g').attr('class', 'reroute-highlights');

    data.links.forEach(link => {
        const sourcePos = layerPositions[link.source];
        const targetPos = layerPositions[link.target];
        if (!sourcePos || !targetPos) return;

        const id = linkId(link.source, link.target);
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

        rerouteLines[id] = coords(rerouteGroup.append('line'))
            .attr('class', 'reroute-path')
            .attr('stroke-width', link.is_host_link ? 4 : 5);

        // Hover
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

    // Draw nodes
    const nodeGroup = svg.append('g').attr('class', 'nodes');

    data.nodes.forEach(node => {
        const pos = layerPositions[node.id];
        if (!pos) return;

        const g = nodeGroup.append('g')
            .attr('class', 'node-group')
            .attr('transform', `translate(${pos.x}, ${pos.y})`);

        const radius = getNodeRadius(node.type);
        const color = getNodeColor(node.type);

        // Outer glow for switches
        if (node.type !== 'host') {
            g.append('circle')
                .attr('class', 'node-glow')
                .attr('r', radius + 6)
                .attr('fill', color)
                .attr('opacity', 0.08);
        }

        // Main circle
        g.append('circle')
            .attr('class', 'node-circle')
            .attr('r', radius)
            .attr('fill', color)
            .attr('stroke', 'rgba(255,255,255,0.15)')
            .attr('stroke-width', node.type === 'host' ? 0.5 : 1.5)
            .attr('data-node-id', node.id);

        // Label for switches
        if (node.type !== 'host') {
            g.append('text')
                .attr('class', 'node-label')
                .attr('dy', radius + 13)
                .text(shortName(node.id));
        }

        // Hover
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

        nodeElements[node.id] = g;
    });

    // Re-apply the latest state (e.g. after a resize).
    if (Object.keys(latestLinkData).length) updateTopologyLinks(latestLinkData);
    updatePredictionOutlines(latestPredictions);
}

function computeLayerLayout(data, width, height) {
    const positions = {};
    const pad = { x: 50, y: 40 };
    const usableW = width - 2 * pad.x;
    const usableH = height - 2 * pad.y;

    const layers = { 0: [], 1: [], 2: [], 3: [] };
    data.nodes.forEach(n => {
        const layer = n.layer !== undefined ? n.layer : 3;
        if (!layers[layer]) layers[layer] = [];
        layers[layer].push(n);
    });

    const layerY = {
        0: pad.y + 20,
        1: pad.y + usableH * 0.3,
        2: pad.y + usableH * 0.58,
        3: pad.y + usableH * 0.88,
    };

    Object.entries(layers).forEach(([layer, nodes]) => {
        const y = layerY[layer];
        if (nodes.length === 0) return;

        // Sort by pod for visual grouping
        nodes.sort((a, b) => {
            if (a.pod !== b.pod) return a.pod - b.pod;
            return (a.index || 0) - (b.index || 0);
        });

        // Add pod spacing
        const k = 4;
        if (parseInt(layer) === 3) {
            // Hosts: group by pod more tightly
            const perPod = nodes.length / k;
            const podWidth = usableW / k;
            nodes.forEach((n, i) => {
                const podIdx = Math.floor(i / perPod);
                const inPodIdx = i % perPod;
                const podStart = pad.x + podIdx * podWidth;
                const spacing = podWidth / (perPod + 1);
                positions[n.id] = { x: podStart + spacing * (inPodIdx + 1), y };
            });
        } else if (parseInt(layer) >= 1) {
            // Agg/Edge: group by pod
            const perPod = nodes.length / k;
            const podWidth = usableW / k;
            nodes.forEach((n, i) => {
                const podIdx = Math.floor(i / perPod);
                const inPodIdx = i % perPod;
                const podStart = pad.x + podIdx * podWidth;
                const spacing = podWidth / (perPod + 1);
                positions[n.id] = { x: podStart + spacing * (inPodIdx + 1), y };
            });
        } else {
            // Core: spread evenly
            const spacing = usableW / (nodes.length + 1);
            nodes.forEach((n, i) => {
                positions[n.id] = { x: pad.x + spacing * (i + 1), y };
            });
        }
    });

    return positions;
}

function getNodeRadius(type) {
    switch (type) {
        case 'core': return 14;
        case 'aggregation': return 11;
        case 'edge': return 10;
        case 'host': return 5;
        default: return 6;
    }
}

function getNodeColor(type) {
    switch (type) {
        case 'core': return '#00d2ff';
        case 'aggregation': return '#7b68ee';
        case 'edge': return '#4fd1c5';
        case 'host': return '#5a6578';
        default: return '#94a3b8';
    }
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

function highlightPath(path) {
    for (let i = 0; i + 1 < path.length; i++) {
        const id = linkId(path[i], path[i + 1]);
        const line = rerouteLines[id];
        if (!line) continue;
        line.classed('active', true);
        clearTimeout(rerouteTimers[id]);
        rerouteTimers[id] = setTimeout(() => line.classed('active', false), REROUTE_HIGHLIGHT_MS);
    }
}

function clearRerouteHighlights() {
    Object.values(rerouteTimers).forEach(clearTimeout);
    rerouteTimers = {};
    Object.values(rerouteLines).forEach(line => line.classed('active', false));
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
        html += tooltipRow('Drops (this tick)', data.drops, data.drops > 0 ? '#e94560' : null);
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

    const typeNames = { core: 'Core Switch', aggregation: 'Aggregation Switch', edge: 'Edge (ToR) Switch', host: 'Server Host' };
    const typeName = typeNames[node.type] || node.type;

    let html = `<div class="tooltip-title">${typeName}</div>`;
    html += tooltipRow('ID', node.id);
    if (node.pod >= 0) {
        html += tooltipRow('Pod', node.pod);
    }
    html += tooltipRow('Layer', node.layer);

    tooltip.innerHTML = html;
    positionTooltip(event);
    tooltip.classList.add('visible');
}

function positionTooltip(event) {
    const tooltip = document.querySelector('.tooltip');
    const x = event.clientX + 15;
    const y = event.clientY - 10;

    // Keep tooltip in viewport
    const maxX = window.innerWidth - 290;
    const maxY = window.innerHeight - 220;

    tooltip.style.left = `${Math.min(x, maxX)}px`;
    tooltip.style.top = `${Math.min(y, maxY)}px`;
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

    const avgUtil = count > 0 ? totalUtil / count : 0;

    metricsHistory.ticks.push(metrics.tick);
    metricsHistory.avgUtilization.push(avgUtil);
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
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                labels: {
                    color: '#7a8598',
                    font: { family: FONT_MAIN, size: 10, weight: '500' },
                    boxWidth: 10,
                    boxHeight: 3,
                    padding: 10,
                    usePointStyle: true,
                    pointStyle: 'rectRounded',
                },
                position: 'top',
            },
            tooltip: {
                backgroundColor: 'rgba(6, 10, 19, 0.9)',
                borderColor: 'rgba(255,255,255,0.08)',
                borderWidth: 1,
                titleFont: { family: FONT_MAIN, size: 11 },
                bodyFont: { family: FONT_MONO, size: 10 },
                padding: 10,
                cornerRadius: 8,
                displayColors: true,
                boxPadding: 4,
            },
        },
        scales: {
            x: {
                display: false,
                grid: { display: false },
            },
            y: {
                grid: {
                    color: 'rgba(0,0,0,0.05)',
                    drawBorder: false,
                },
                ticks: {
                    color: '#4a5568',
                    font: { size: 9, family: FONT_MONO },
                    maxTicksLimit: 5,
                },
                border: { display: false },
            },
        },
        elements: {
            point: { radius: 0, hoverRadius: 3 },
            line: { tension: 0.4, borderWidth: 1.5 },
        },
    };

    // Utilization chart
    utilizationChart = new Chart(
        document.getElementById('chart-utilization'),
        {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Avg Utilization',
                        data: [],
                        borderColor: '#00d2ff',
                        backgroundColor: 'rgba(0, 210, 255, 0.06)',
                        fill: true,
                    },
                    {
                        label: 'Peak Utilization',
                        data: [],
                        borderColor: '#e94560',
                        backgroundColor: 'rgba(233, 69, 96, 0.04)',
                        fill: true,
                        borderDash: [4, 3],
                    },
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
                        ticks: {
                            ...commonOptions.scales.y.ticks,
                            callback: v => `${(v * 100).toFixed(0)}%`,
                        },
                    },
                },
            },
        }
    );

    // Drops + Flows chart
    dropsChart = new Chart(
        document.getElementById('chart-drops'),
        {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Fabric drops / tick',
                        data: [],
                        borderColor: '#e94560',
                        backgroundColor: 'rgba(233, 69, 96, 0.06)',
                        fill: true,
                        yAxisID: 'y',
                    },
                    {
                        label: 'Host-link drops / tick',
                        data: [],
                        borderColor: '#718096',
                        backgroundColor: 'rgba(113, 128, 150, 0.04)',
                        borderDash: [4, 3],
                        fill: false,
                        yAxisID: 'y',
                    },
                    {
                        label: 'Active Flows',
                        data: [],
                        borderColor: '#4fd1c5',
                        backgroundColor: 'rgba(79, 209, 197, 0.04)',
                        fill: true,
                        yAxisID: 'y1',
                    },
                ],
            },
            options: {
                ...commonOptions,
                scales: {
                    ...commonOptions.scales,
                    y: {
                        ...commonOptions.scales.y,
                        position: 'left',
                    },
                    y1: {
                        position: 'right',
                        grid: { display: false },
                        ticks: {
                            color: '#3d8a80',
                            font: { size: 9, family: FONT_MONO },
                            maxTicksLimit: 5,
                        },
                        border: { display: false },
                    },
                },
            },
        }
    );
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

function setupSliders() {
    const thresholdSlider = document.getElementById('threshold-slider');
    thresholdSlider.addEventListener('input', () => setThreshold(thresholdSlider.value));
    document.getElementById('threshold-reset').addEventListener('click', () => {
        if (alertThreshold != null) setThreshold(alertThreshold);
    });

    const speedSlider = document.getElementById('speed-slider');
    const speedValue = document.getElementById('speed-value');
    speedSlider.addEventListener('input', () => {
        speedValue.textContent = `${parseFloat(speedSlider.value).toFixed(1)}x`;
        fetch('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ speed: parseFloat(speedSlider.value) }),
        }).catch(() => {});
    });
}

// ── Event Log ──────────────────────────────────────────────────
function addLog(message, type = 'info') {
    const log = document.getElementById('event-log');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;

    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });

    const icons = { info: '●', success: '✓', warning: '⚠', danger: '✕', reroute: '↗' };

    entry.innerHTML = `
        <span class="log-time">${time}</span>
        <span class="log-icon">${icons[type] || '●'}</span>
        <span class="log-msg"></span>
    `;
    entry.querySelector('.log-msg').textContent = message;

    log.insertBefore(entry, log.firstChild);

    while (log.children.length > 100) {
        log.removeChild(log.lastChild);
    }
}

function clearLog() {
    document.getElementById('event-log').innerHTML = '';
    addLog('Log cleared', 'info');
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
