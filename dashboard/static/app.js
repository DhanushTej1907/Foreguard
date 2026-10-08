/**
 * NeuralFabric — GNN Congestion Pre-emption Dashboard
 * =====================================================
 * Real-time D3.js topology + Chart.js metrics + WebSocket live updates
 * with congestion prediction heatmap, sparkline trends, and animated packet flows
 */

// ── State ──────────────────────────────────────────────────────
let ws = null;
let topologyData = null;
let linkElements = {};
let nodeElements = {};
let isRunning = false;
let totalReroutes = 0;
let totalAlerts = 0;

// Sparkline data
const SPARK_MAX = 30;
const sparkData = {
    flows: [], packets: [], drops: [], droprate: [], reroutes: [], predictions: []
};

// Time-series data buffers
const MAX_HISTORY = 120;
const metricsHistory = {
    ticks: [],
    avgUtilization: [],
    maxUtilization: [],
    drops: [],
    flows: [],
    dropsPerTick: [],
};

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
    addLog('Dashboard ready. Click "Start Engine" to begin.', 'info');
});

// ── WebSocket ──────────────────────────────────────────────────
function connectWebSocket() {
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
            isRunning = true;
            document.getElementById('btn-start').disabled = true;
            document.getElementById('btn-stop').disabled = false;
            document.getElementById('btn-spike').disabled = false;
            document.getElementById('topology-overlay').classList.add('hidden');

            if (data.topology) {
                topologyData = data.topology;
                renderTopology(data.topology);
            }

            connectWebSocket();
            addLog('Simulation engine started', 'success');
        }
    } catch (e) {
        addLog(`Failed to start: ${e.message}`, 'danger');
    }
}

async function stopSimulation() {
    try {
        await fetch('/api/stop', { method: 'POST' });
        isRunning = false;
        document.getElementById('btn-start').disabled = false;
        document.getElementById('btn-stop').disabled = true;
        document.getElementById('btn-spike').disabled = true;
        addLog('Simulation stopped', 'warning');
    } catch (e) {
        addLog(`Failed to stop: ${e.message}`, 'danger');
    }
}

async function injectSpike() {
    try {
        const resp = await fetch('/api/spike', { method: 'POST' });
        const data = await resp.json();
        addLog(`Traffic spike injected — ${data.new_flows} incast flows targeting random hosts`, 'danger');

        // Visual feedback
        const btn = document.getElementById('btn-spike');
        btn.style.transform = 'scale(0.95)';
        setTimeout(() => { btn.style.transform = ''; }, 200);
    } catch (e) {
        addLog(`Spike injection failed: ${e.message}`, 'danger');
    }
}

// ── Handle Real-Time Updates ───────────────────────────────────
let lastDrops = 0;

function handleUpdate(msg) {
    const { links, predictions, metrics } = msg;

    // Update tick counter
    document.getElementById('tick-value').textContent = metrics.tick;

    // Update metric cards
    document.getElementById('val-flows').textContent = metrics.active_flows;
    document.getElementById('val-packets').textContent = formatNumber(metrics.total_packets);
    document.getElementById('val-drops').textContent = formatNumber(metrics.total_drops);
    document.getElementById('val-droprate').textContent = `${metrics.drop_rate}%`;
    document.getElementById('val-reroutes').textContent = formatNumber(totalReroutes);

    // Count predictions
    let alertCount = 0;
    if (predictions && Object.keys(predictions).length > 0) {
        Object.values(predictions).forEach(prob => {
            if (prob > 0.7) alertCount++;
        });
    }
    totalAlerts = alertCount;
    document.getElementById('val-predictions').textContent = alertCount;

    // Update sparklines
    pushSparkData('flows', metrics.active_flows);
    pushSparkData('packets', metrics.total_packets);
    pushSparkData('drops', metrics.total_drops);
    pushSparkData('droprate', metrics.drop_rate);
    pushSparkData('reroutes', totalReroutes);
    pushSparkData('predictions', alertCount);
    drawAllSparklines();

    // Update topology link colors
    updateTopologyLinks(links, predictions);

    // Update predictions panel
    updatePredictionPanel(predictions);

    // Update time-series data
    const dropsThisTick = metrics.total_drops - lastDrops;
    lastDrops = metrics.total_drops;
    updateTimeSeries(links, metrics, dropsThisTick);

    // Update charts (throttled)
    if (metrics.tick % 3 === 0) updateCharts();

    // Update health score
    updateHealthScore(links, metrics);

    // Reroute events from server
    if (msg.reroute_events && msg.reroute_events.length > 0) {
        msg.reroute_events.forEach(evt => {
            totalReroutes++;
            addLog(`GNN rerouted flow ${evt.flow_id} (${evt.src} → ${evt.dst})`, 'reroute');
        });
    }
}

// ── Health Score ───────────────────────────────────────────────
function updateHealthScore(links, metrics) {
    if (!links) return;

    let totalUtil = 0, maxUtil = 0, count = 0;
    Object.values(links).forEach(d => {
        totalUtil += d.utilization;
        maxUtil = Math.max(maxUtil, d.utilization);
        count++;
    });

    const avgUtil = count > 0 ? totalUtil / count : 0;

    // Health: 100 = perfect, penalized by utilization and drops
    let health = 100;
    health -= avgUtil * 30;          // Avg util penalty
    health -= maxUtil * 20;          // Peak util penalty
    health -= Math.min(30, metrics.drop_rate * 5); // Drop rate penalty
    health = Math.max(0, Math.min(100, Math.round(health)));

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
function updatePredictionPanel(predictions) {
    const container = document.getElementById('prediction-heatmap');
    const statusEl = document.getElementById('gnn-status');

    if (!predictions || Object.keys(predictions).length === 0) {
        statusEl.textContent = 'Collecting data...';
        return;
    }

    statusEl.textContent = 'Active';
    statusEl.style.color = '#48bb78';
    statusEl.style.borderColor = 'rgba(72, 187, 120, 0.3)';

    const probs = Object.values(predictions).map(Number);

    // Build heatmap cells
    if (container.children.length !== probs.length || container.querySelector('.heatmap-placeholder')) {
        container.innerHTML = '';
        probs.forEach((_, i) => {
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';
            cell.title = `Link ${i}`;
            cell.setAttribute('data-edge-idx', i);
            container.appendChild(cell);
        });
    }

    // Update colors
    const cells = container.querySelectorAll('.heatmap-cell');
    let maxProb = 0, avgProb = 0, riskCount = 0;

    probs.forEach((prob, i) => {
        if (i < cells.length) {
            cells[i].style.background = predictionColor(prob);
            cells[i].title = `Link ${i}: ${(prob * 100).toFixed(1)}% congestion risk`;
        }
        maxProb = Math.max(maxProb, prob);
        avgProb += prob;
        if (prob > 0.7) riskCount++;
    });

    avgProb = probs.length > 0 ? avgProb / probs.length : 0;

    document.getElementById('pred-avg').textContent = `${(avgProb * 100).toFixed(1)}%`;
    document.getElementById('pred-max').textContent = `${(maxProb * 100).toFixed(1)}%`;
    document.getElementById('pred-risk-count').textContent = riskCount;

    latestPredictions = predictions;
}

function predictionColor(prob) {
    if (prob < 0.3) return `rgba(72, 187, 120, ${0.2 + prob * 0.5})`;
    if (prob < 0.6) return `rgba(246, 173, 85, ${0.3 + prob * 0.5})`;
    if (prob < 0.8) return `rgba(237, 137, 54, ${0.4 + prob * 0.5})`;
    return `rgba(233, 69, 96, ${0.5 + prob * 0.5})`;
}

// ── Sparklines ─────────────────────────────────────────────────
const sparkCanvases = {};

function initSparklines() {
    ['flows', 'packets', 'drops', 'droprate', 'reroutes', 'predictions'].forEach(key => {
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
        drops: '#e94560',
        droprate: '#f6ad55',
        reroutes: '#7b68ee',
        predictions: '#f6ad55',
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
    ctx.fillStyle = color.replace(')', ', 0.08)').replace('rgb', 'rgba');
    ctx.fill();
}

// ── D3.js Topology Rendering ───────────────────────────────────
function renderTopology(data) {
    const container = document.getElementById('topology-container');
    const svg = d3.select('#topology-svg');
    svg.selectAll('*').remove();
    linkElements = {};
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

    // Defs
    const defs = svg.append('defs');

    // Link glow filter
    const linkGlow = defs.append('filter').attr('id', 'linkGlow').attr('x', '-50%').attr('y', '-50%').attr('width', '200%').attr('height', '200%');
    linkGlow.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'blur');
    const linkMerge = linkGlow.append('feMerge');
    linkMerge.append('feMergeNode').attr('in', 'blur');
    linkMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Node glow filter
    const nodeGlow = defs.append('filter').attr('id', 'nodeGlow').attr('x', '-100%').attr('y', '-100%').attr('width', '300%').attr('height', '300%');
    nodeGlow.append('feGaussianBlur').attr('stdDeviation', '6').attr('result', 'blur');
    const nodeMerge = nodeGlow.append('feMerge');
    nodeMerge.append('feMergeNode').attr('in', 'blur');
    nodeMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Draw links
    const linkGroup = svg.append('g').attr('class', 'links');

    data.links.forEach(link => {
        const sourcePos = layerPositions[link.source];
        const targetPos = layerPositions[link.target];
        if (!sourcePos || !targetPos) return;

        const linkId = `${link.source}--${link.target}`;
        const altLinkId = `${link.target}--${link.source}`;

        const line = linkGroup.append('line')
            .attr('class', 'link-line')
            .attr('x1', sourcePos.x).attr('y1', sourcePos.y)
            .attr('x2', targetPos.x).attr('y2', targetPos.y)
            .attr('stroke', '#1a2638')
            .attr('stroke-width', link.is_host_link ? 1 : 2.5)
            .attr('stroke-opacity', link.is_host_link ? 0.25 : 0.5)
            .attr('data-link-id', linkId);

        linkElements[linkId] = line;
        linkElements[altLinkId] = line;

        // Hover
        line.on('mouseover', function(event) {
            d3.select(this).attr('stroke-width', parseFloat(d3.select(this).attr('stroke-width')) + 3);
            showLinkTooltip(event, link, linkId);
        }).on('mousemove', function(event) {
            moveTooltip(event);
        }).on('mouseout', function() {
            const isHost = link.is_host_link;
            const baseWidth = isHost ? 1 : 2.5;
            const linkData = latestLinkData[linkId];
            const util = linkData ? linkData.utilization : 0;
            d3.select(this).attr('stroke-width', baseWidth + util * 3);
            hideTooltip();
        });
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
            const shortLabel = node.id
                .replace('core_', 'C')
                .replace(/agg_(\d+)_(\d+)/, 'A$1.$2')
                .replace(/edge_(\d+)_(\d+)/, 'E$1.$2');

            g.append('text')
                .attr('class', 'node-label')
                .attr('dy', radius + 13)
                .text(shortLabel);
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
function updateTopologyLinks(links, predictions) {
    if (!links) return;

    latestLinkData = links;

    Object.entries(links).forEach(([linkId, data]) => {
        const line = linkElements[linkId];
        if (!line) return;

        const util = data.utilization;
        const color = utilizationColor(util);
        const baseWidth = 2.5;

        line.attr('stroke', color)
            .attr('stroke-opacity', 0.4 + util * 0.6)
            .attr('stroke-width', baseWidth + util * 3);

        // Add congestion class for high utilization
        line.classed('congestion-predicted', util > 0.85);
        if (util > 0.85) {
            line.attr('filter', 'url(#linkGlow)');
        } else {
            line.attr('filter', null);
        }
    });
}

function utilizationColor(util) {
    if (util < 0.25) return d3.interpolateRgb('#1a3a2a', '#48bb78')(util / 0.25);
    if (util < 0.5)  return d3.interpolateRgb('#48bb78', '#f6ad55')((util - 0.25) / 0.25);
    if (util < 0.75) return d3.interpolateRgb('#f6ad55', '#ed8936')((util - 0.5) / 0.25);
    if (util < 0.9)  return d3.interpolateRgb('#ed8936', '#e94560')((util - 0.75) / 0.15);
    return d3.interpolateRgb('#e94560', '#ff2d55')((util - 0.9) / 0.1);
}

// ── Tooltips ───────────────────────────────────────────────────
function showLinkTooltip(event, link, linkId) {
    const tooltip = document.querySelector('.tooltip');
    if (!tooltip) return;

    const data = latestLinkData[linkId];
    let html = `<div class="tooltip-title">${link.source} ↔ ${link.target}</div>`;

    if (data) {
        const utilPct = (data.utilization * 100).toFixed(1);
        const utilColor = utilizationColor(data.utilization);
        html += `<div class="tooltip-row"><span class="tooltip-label">Utilization</span><span class="tooltip-value" style="color:${utilColor}">${utilPct}%</span></div>`;
        html += `<div class="tooltip-bar"><div class="tooltip-bar-fill" style="width:${utilPct}%;background:${utilColor}"></div></div>`;
        html += `<div class="tooltip-row"><span class="tooltip-label">Queue Depth</span><span class="tooltip-value">${data.queue_depth}</span></div>`;
        html += `<div class="tooltip-row"><span class="tooltip-label">Drops (this tick)</span><span class="tooltip-value" style="color:${data.drops > 0 ? '#e94560' : '#5a6578'}">${data.drops}</span></div>`;
        html += `<div class="tooltip-row"><span class="tooltip-label">Latency</span><span class="tooltip-value">${data.latency.toFixed(1)} μs</span></div>`;
    } else {
        html += `<div class="tooltip-row"><span class="tooltip-label">Status</span><span class="tooltip-value">Idle</span></div>`;
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
    html += `<div class="tooltip-row"><span class="tooltip-label">ID</span><span class="tooltip-value">${node.id}</span></div>`;
    if (node.pod >= 0) {
        html += `<div class="tooltip-row"><span class="tooltip-label">Pod</span><span class="tooltip-value">${node.pod}</span></div>`;
    }
    html += `<div class="tooltip-row"><span class="tooltip-label">Layer</span><span class="tooltip-value">${node.layer}</span></div>`;

    tooltip.innerHTML = html;
    positionTooltip(event);
    tooltip.classList.add('visible');
}

function positionTooltip(event) {
    const tooltip = document.querySelector('.tooltip');
    const x = event.clientX + 15;
    const y = event.clientY - 10;

    // Keep tooltip in viewport
    const rect = tooltip.getBoundingClientRect();
    const maxX = window.innerWidth - 290;
    const maxY = window.innerHeight - 200;

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
function updateTimeSeries(links, metrics, dropsThisTick) {
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
    metricsHistory.drops.push(metrics.total_drops);
    metricsHistory.flows.push(metrics.active_flows);
    metricsHistory.dropsPerTick.push(dropsThisTick);

    if (metricsHistory.ticks.length > MAX_HISTORY) {
        Object.keys(metricsHistory).forEach(key => {
            metricsHistory[key] = metricsHistory[key].slice(-MAX_HISTORY);
        });
    }
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
                    font: { family: "'Outfit', sans-serif", size: 10, weight: '500' },
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
                titleFont: { family: "'Outfit', sans-serif", size: 11 },
                bodyFont: { family: "'JetBrains Mono', monospace", size: 10 },
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
                    color: 'rgba(255,255,255,0.03)',
                    drawBorder: false,
                },
                ticks: {
                    color: '#4a5568',
                    font: { size: 9, family: "'JetBrains Mono', monospace" },
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
                        ...commonOptions.scales.y,
                        min: 0,
                        max: 1.2,
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
                        label: 'Drops / Tick',
                        data: [],
                        borderColor: '#e94560',
                        backgroundColor: 'rgba(233, 69, 96, 0.06)',
                        fill: true,
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
                            font: { size: 9, family: "'JetBrains Mono', monospace" },
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
    dropsChart.data.datasets[0].data = metricsHistory.dropsPerTick;
    dropsChart.data.datasets[1].data = metricsHistory.flows;
    dropsChart.update('none');
}

// ── Sliders ────────────────────────────────────────────────────
function setupSliders() {
    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdValue = document.getElementById('threshold-value');
    thresholdSlider.addEventListener('input', () => {
        thresholdValue.textContent = parseFloat(thresholdSlider.value).toFixed(2);
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: 'threshold', value: thresholdSlider.value }));
        }
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
        <span class="log-msg">${message}</span>
    `;

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
