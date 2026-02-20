/**
 * AI Behavior Detection — Dashboard Frontend Logic
 * WebSocket connections, DOM updates, alert sounds, zone drawing
 */

// ============================================
// STATE
// ============================================
const state = {
    videoWs: null,
    alertWs: null,
    isConnected: false,
    alerts: [],
    persons: {},
    behaviorScores: {},
    showZones: true,
    showSkeleton: true,
};

const WS_BASE = `ws://${window.location.host}`;
const API_BASE = `http://${window.location.host}/api`;

// ============================================
// INITIALIZATION
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initVideoStream();
    initAlertStream();
    initControls();
    fetchInitialData();

    // Reconnect on disconnect
    setInterval(checkConnections, 5000);
});

// ============================================
// VIDEO WEBSOCKET
// ============================================
function initVideoStream() {
    const canvas = document.getElementById('videoCanvas');
    const ctx = canvas.getContext('2d');
    const overlay = document.getElementById('videoOverlay');

    try {
        state.videoWs = new WebSocket(`${WS_BASE}/ws/video`);

        state.videoWs.onopen = () => {
            state.isConnected = true;
            overlay.classList.add('hidden');
            updateSystemStatus(true);
        };

        state.videoWs.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'frame') {
                // Decode and draw frame
                const img = new Image();
                img.onload = () => {
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                };
                img.src = 'data:image/jpeg;base64,' + data.data;

                // Update stats
                document.getElementById('fpsValue').textContent = data.fps;
                document.getElementById('personCount').textContent = data.persons;
            }
        };

        state.videoWs.onclose = () => {
            state.isConnected = false;
            overlay.classList.remove('hidden');
            overlay.querySelector('span').textContent = 'Connection lost. Reconnecting...';
            updateSystemStatus(false);
            setTimeout(initVideoStream, 3000);
        };

        state.videoWs.onerror = () => {
            overlay.querySelector('span').textContent = 'Connection error. Check if system is running.';
        };
    } catch (e) {
        console.error('Video WebSocket error:', e);
    }
}

// ============================================
// ALERT WEBSOCKET
// ============================================
function initAlertStream() {
    try {
        state.alertWs = new WebSocket(`${WS_BASE}/ws/alerts`);

        state.alertWs.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'scores') {
                updateBehaviorScores(data.data);
            }
        };

        state.alertWs.onclose = () => {
            setTimeout(initAlertStream, 3000);
        };
    } catch (e) {
        console.error('Alert WebSocket error:', e);
    }

    // Poll alerts via REST (more reliable for history)
    setInterval(fetchAlerts, 2000);
}

// ============================================
// DATA FETCHING
// ============================================
async function fetchInitialData() {
    await Promise.all([
        fetchAlerts(),
        fetchPersons(),
        fetchEvidence()
    ]);
}

async function fetchAlerts() {
    try {
        const res = await fetch(`${API_BASE}/alerts?count=30`);
        const data = await res.json();

        const newAlerts = data.alerts || [];
        if (newAlerts.length > state.alerts.length) {
            // New alert arrived
            const latestNew = newAlerts[newAlerts.length - 1];
            if (state.alerts.length > 0) {
                playAlertSound(latestNew.alert_level);
            }
        }

        state.alerts = newAlerts;
        renderAlerts();
        document.getElementById('alertCount').textContent = state.alerts.length;
    } catch (e) {
        // API not available yet
    }
}

async function fetchPersons() {
    try {
        const res = await fetch(`${API_BASE}/persons`);
        const data = await res.json();
        renderPersons(data.persons || []);
    } catch (e) { }
}

async function fetchEvidence() {
    try {
        const res = await fetch(`${API_BASE}/evidence?count=10`);
        const data = await res.json();
        renderEvidence(data.clips || []);
    } catch (e) { }
}

// ============================================
// RENDERERS
// ============================================
function renderAlerts() {
    const container = document.getElementById('alertsList');
    if (state.alerts.length === 0) {
        container.innerHTML = '<div class="empty-state">No alerts yet</div>';
        return;
    }

    // Show newest first
    const sorted = [...state.alerts].reverse();
    container.innerHTML = sorted.map(alert => {
        const time = new Date(alert.timestamp * 1000).toLocaleTimeString();
        const reasons = alert.reasons.slice(0, 2).join(', ');
        return `
            <div class="alert-card ${alert.alert_level}" data-id="${alert.alert_id}">
                <div class="alert-header">
                    <span class="alert-level">${alert.alert_level}</span>
                    <span class="alert-time">${time}</span>
                </div>
                <div class="alert-person">${alert.person_name} (ID: ${alert.track_id})</div>
                <div class="alert-reason">${reasons}</div>
                <span class="alert-score">Score: ${alert.behavior_score.toFixed(1)}</span>
            </div>
        `;
    }).join('');
}

function renderPersons(persons) {
    const container = document.getElementById('personsList');
    if (persons.length === 0) {
        container.innerHTML = '<div class="empty-state">No persons detected</div>';
        return;
    }

    container.innerHTML = persons.map(p => {
        const initials = p.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);
        const flagged = p.is_flagged ? 'flagged' : '';
        const badge = p.is_flagged
            ? '<span class="person-badge flagged-badge">⚠ Flagged</span>'
            : '<span class="person-badge active-badge">Active</span>';

        return `
            <div class="person-card ${flagged}">
                <div class="person-avatar">${initials}</div>
                <div class="person-info">
                    <div class="person-name">${p.name}</div>
                    <div class="person-meta">Recognized ${p.times_recognized}x • ${p.images} photos</div>
                </div>
                ${badge}
            </div>
        `;
    }).join('');
}

function updateBehaviorScores(scores) {
    state.behaviorScores = scores;
    const container = document.getElementById('behaviorGrid');

    const entries = Object.values(scores);
    if (entries.length === 0) {
        container.innerHTML = '<div class="empty-state">Waiting for data...</div>';
        return;
    }

    container.innerHTML = entries.map(s => {
        const cardClass = s.total_score >= 65 ? 'danger' :
            s.total_score >= 40 ? 'suspicious' : '';
        const scoreColor = s.total_score >= 65 ? 'red' :
            s.total_score >= 40 ? 'orange' :
                s.total_score >= 20 ? 'yellow' : 'green';

        return `
            <div class="behavior-card ${cardClass}">
                <div class="person-id">${s.person_name} — Track #${s.track_id}</div>
                ${renderScoreBar('Action', s.action_score)}
                ${renderScoreBar('Trajectory', s.trajectory_score)}
                ${renderScoreBar('Pose', s.pose_score)}
                ${renderScoreBar('Zone', s.zone_score)}
                <div class="total-score">
                    <div class="score-number" style="color: var(--alert-${scoreColor === 'green' ? 'low' : scoreColor}, var(--accent-green))">${s.total_score.toFixed(0)}</div>
                    <div class="score-label">Suspicion Score</div>
                </div>
            </div>
        `;
    }).join('');
}

function renderScoreBar(label, value) {
    const pct = Math.min(100, Math.max(0, value));
    const color = pct >= 65 ? 'red' : pct >= 40 ? 'orange' : pct >= 20 ? 'yellow' : 'green';
    return `
        <div class="score-bar-container">
            <div class="score-bar-label">
                <span>${label}</span>
                <span>${pct.toFixed(0)}</span>
            </div>
            <div class="score-bar">
                <div class="score-bar-fill ${color}" style="width: ${pct}%"></div>
            </div>
        </div>
    `;
}

function renderEvidence(clips) {
    const container = document.getElementById('evidenceList');
    if (clips.length === 0) {
        container.innerHTML = '<div class="empty-state">No evidence recorded</div>';
        return;
    }

    container.innerHTML = clips.reverse().map(clip => {
        const time = new Date(clip.start_time * 1000).toLocaleString();
        return `
            <div class="evidence-card" onclick="window.open('/api/evidence/${clip.clip_id}/video')">
                <img class="evidence-thumb" src="/api/evidence/${clip.clip_id}/thumbnail" 
                     onerror="this.style.display='none'" alt="Evidence thumbnail">
                <div class="evidence-info">
                    <div class="evidence-title">${clip.person_name} — Score ${clip.behavior_score.toFixed(0)}</div>
                    <div class="evidence-meta">${time} • ${clip.duration.toFixed(0)}s</div>
                </div>
            </div>
        `;
    }).join('');
}

// ============================================
// CONTROLS
// ============================================
function initControls() {
    // Toggle zones
    document.getElementById('toggleZones').addEventListener('click', (e) => {
        state.showZones = !state.showZones;
        e.target.classList.toggle('active');
    });

    // Toggle skeleton
    document.getElementById('toggleSkeleton').addEventListener('click', (e) => {
        state.showSkeleton = !state.showSkeleton;
        e.target.classList.toggle('active');
    });

    // Clear alerts
    document.getElementById('clearAlerts').addEventListener('click', () => {
        state.alerts = [];
        renderAlerts();
    });

    // Refresh evidence
    document.getElementById('refreshEvidence').addEventListener('click', fetchEvidence);

    // Register modal
    document.getElementById('closeModal').addEventListener('click', () => {
        document.getElementById('registerModal').classList.remove('active');
    });

    // Register form
    document.getElementById('registerForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('personName').value;
        const photo = document.getElementById('personPhoto').files[0];

        if (!name || !photo) return;

        const formData = new FormData();
        formData.append('name', name);
        formData.append('file', photo);

        try {
            const res = await fetch(`${API_BASE}/persons/register`, {
                method: 'POST',
                body: formData
            });
            if (res.ok) {
                document.getElementById('registerModal').classList.remove('active');
                fetchPersons();
            }
        } catch (e) {
            console.error('Registration failed:', e);
        }
    });
}

// ============================================
// UTILITIES
// ============================================
function updateSystemStatus(online) {
    const statusEl = document.getElementById('systemStatus');
    const dot = statusEl.querySelector('.status-dot');
    const text = statusEl.querySelector('.status-text');

    if (online) {
        dot.className = 'status-dot online';
        text.textContent = 'Live';
    } else {
        dot.className = 'status-dot offline';
        text.textContent = 'Offline';
    }
}

function playAlertSound(level) {
    if (level === 'high' || level === 'medium') {
        try {
            // Create a beep using Web Audio API
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioCtx.createOscillator();
            const gainNode = audioCtx.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(audioCtx.destination);

            oscillator.frequency.value = level === 'high' ? 880 : 660;
            oscillator.type = 'sine';
            gainNode.gain.value = 0.3;

            oscillator.start();
            setTimeout(() => {
                gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.3);
                setTimeout(() => oscillator.stop(), 300);
            }, 200);
        } catch (e) { }
    }
}

function checkConnections() {
    if (!state.videoWs || state.videoWs.readyState === WebSocket.CLOSED) {
        initVideoStream();
    }
    if (!state.alertWs || state.alertWs.readyState === WebSocket.CLOSED) {
        initAlertStream();
    }

    // Refresh persons periodically
    fetchPersons();
}
