/*  
VirtualMoonLander interactive portfolio demo.
Human mode is a browser-only simulation. AI mode is connected to the
trained TensorFlow DQN through the FastAPI evaluation service and uses
real Gymnasium environment states and Q-values.
*/

(() => {
    const canvas = document.getElementById('game');
    const ctx = canvas.getContext('2d');
    const status = document.getElementById('status');
    const modeLabel = document.getElementById('modeLabel');
    const episodeState = document.getElementById('episodeState');
    const message = document.getElementById('gameMessage');
    const restartBtn = document.getElementById('restart');
    const startAIBtn = document.getElementById('startAI');
    const qDescription = document.getElementById('qDescription');
    const decision = document.getElementById('decision');
    const qRows = [...document.querySelectorAll('.q-row')];

    const W = canvas.width, H = canvas.height;
    const HUMAN_TIME_SCALE = 0.20;
    const LANDER_START_X = W * 0.5;
    const LANDER_START_Y = 110;
    const LANDING_SURFACE_Y = H - 50;
    const LANDER_FEET_OFFSET = 44;
    const PAD_CENTER_X = 0.5 * W;
    const PAD_WIDTH = 104;

    const TERRAIN_BASE_Y = LANDING_SURFACE_Y;
    const PAD_TERRAIN_HALF = PAD_WIDTH * 0.65;

    const keys = new Set();
    let mode = 'human';
    let raf = 0;
    let last = performance.now();
    let aiReplay = null;
    let aiIndex = 0;
    let aiTimer = null;
    let aiRequestId = 0;
    let aiVisualOffsetX = 0;
    let aiVisualOffsetY = 0;
    let trainingResults = null;
    const TRAINING_VERSION = 'v2';
    let apiAvailable = false;

    const human = {};
    function resetHuman() {
        human.x = LANDER_START_X;
        human.y = LANDER_START_Y;
        human.vx = 0;
        human.vy = 0;
        human.angle = 0;
        human.fuel = 100;
        human.reward = 0;
        human.done = false;
        human.landed = false;
        episodeState.textContent = 'FLYING';
        message.classList.remove('hide');
        message.innerHTML =
            '<strong>Land softly on the pad.</strong>' +
            '<span>↑ Main engine: slow descent<br>' +
            '← / → Adjust angle toward 0°<br>' +
            'Release controls to stabilize</span>';
        updateTelemetry(human);
        clearQValues();
    }

    function setStatus(text, color = 'ok') {
        status.innerHTML = `<span></span> ${text}`;
        status.querySelector('span').style.background = color === 'warn' ? 'var(--warn)' : color === 'bad' ? 'var(--danger)' : 'var(--accent)';
    }

    function resetAll() {
        cancelAnimationFrame(raf);
        aiRequestId++;
        startAIBtn.disabled = false;
        restartBtn.disabled = false;
        clearInterval(aiTimer);
        aiTimer = null;
        if (mode === 'human') resetHuman(); else resetAIView();
        last = performance.now();
        raf = requestAnimationFrame(loop);
    }

    function resetAIView() {
        aiIndex = 0;
        aiVisualOffsetX = 0;
        aiVisualOffsetY = 0;

        clearInterval(aiTimer);
        aiTimer = null;

        episodeState.textContent = 'READY';

        message.classList.remove('hide');
        message.innerHTML =
            '<strong>DQN evaluation ready.</strong>' +
            '<span>Press Run AI to evaluate the trained agent.</span>';

        clearQValues();
    }

    function actionName(action) {
        return [
            '<i class="fa-solid fa-arrow-left" aria-hidden="true"></i> Left engine',
            '<i class="fa-solid fa-arrow-right" aria-hidden="true"></i> Right engine',
            '<i class="fa-solid fa-fire" aria-hidden="true"></i> Main engine',
            '<i class="fa-regular fa-circle" aria-hidden="true"></i> No action'
        ][Number(action)] ?? 'Unknown';
    }

    function getLegContact(state) {
        const left = Boolean(state?.[6]);
        const right = Boolean(state?.[7]);

        if (left && right) return 'BOTH';
        if (left) return 'LEFT';
        if (right) return 'RIGHT';
        return 'NONE';
    }

    function setTelemetryState(elementId, value, threshold, inverse = false) {
        const element = document.getElementById(elementId);
        if (!element) return;

        const safe = inverse ? Number(value) > threshold : Math.abs(Number(value)) < threshold;

        element.classList.toggle('safe', safe);
        element.classList.toggle('unsafe', !safe);
    }

    function setFuelState(value) {
        const element = document.getElementById('fuel');
        if (!element) return;

        element.classList.remove('safe', 'warning', 'unsafe');

        if (value > 30) {
            element.classList.add('safe');
        } else if (value > 10) {
            element.classList.add('warning');
        } else {
            element.classList.add('unsafe');
        }
    }

    function setContactState(value) {
        const element = document.getElementById('contacts');
        if (!element) return;

        element.classList.remove('safe', 'warning', 'unsafe');

        if (value === 'BOTH') {
            element.classList.add('safe');
        } else if (value === 'LEFT' || value === 'RIGHT') {
            element.classList.add('warning');
        } else {
            element.classList.add('unsafe');
        }
    }

    function updateTelemetryMode() {
        const fuelMetric = document.getElementById('fuelMetric');
        const contactsMetric = document.getElementById('contactsMetric');
    
        if (!fuelMetric || !contactsMetric) return;
    
        const humanMode = mode === 'human';
    
        fuelMetric.style.display = humanMode ? '' : 'none';
        contactsMetric.style.display = humanMode ? 'none' : '';
    }

    function updateTelemetry(s, isAI = false) {
        let altitude;
    
        if (isAI) {
            altitude = Number.isFinite(Number(s.altitude)) ? Number(s.altitude) : 0;
            altitude = Math.max(altitude * 100, 0);

        } else {
            const landerFeetY = s.y + LANDER_FEET_OFFSET;
            altitude = Math.max(0, LANDING_SURFACE_Y - landerFeetY);
        }
    
        document.getElementById('altitude').textContent = `${altitude.toFixed(2)}`;
    
        const fuelElement = document.getElementById('fuel');
    
        if (s.fuel == null) {
            fuelElement.textContent = 'N/A';
        } else {
            fuelElement.textContent = `${Math.max(0, Math.min(s.fuel, 100)).toFixed(0)}%`;
        }
    
        document.getElementById('vy').textContent = `${Number(s.vy).toFixed(2)}`;
        document.getElementById('vx').textContent = `${Number(s.vx).toFixed(2)}`;
        document.getElementById('angle').textContent = `${(Number(s.angle) * 180 / Math.PI).toFixed(1)}°`;
    
        const contacts = s.contacts ?? '—';
        document.getElementById('contacts').textContent = contacts;
        setContactState(contacts);
        document.getElementById('reward').textContent = Number(s.reward ?? 0).toFixed(0);

        setTelemetryState('vy', s.vy, 2.5);
        setTelemetryState('vx', s.vx, 2.5);
        setTelemetryState('angle', s.angle, 0.25);
        setTelemetryState('altitude', altitude, 50, true);
        setFuelState(s.fuel);
    }

    function updateQValues(values, selected, source = 'DQN') {
        if (!Array.isArray(values) || values.length < 4) return clearQValues();
        const max = Math.max(...values.map(Number));
        const min = Math.min(...values.map(Number));
        const span = Math.max(0.001, max - min);
        qRows.forEach((row, i) => {
        const v = Number(values[i]);
        row.querySelector('.bar i').style.width = `${Math.max(5, ((v - min) / span) * 100)}%`;
        row.querySelector('b').textContent = v.toFixed(2);
        row.classList.toggle('selected', Number(selected) === i);
        });
        decision.innerHTML = `${source} selected ${actionName(selected)} with Q = ${Number(values[selected]).toFixed(2)}`;
        qDescription.textContent = 'The selected action is the highest predicted Q-value for this state.';
    }

    function clearQValues() {
        qRows.forEach(row => { row.querySelector('.bar i').style.width = '0%'; row.querySelector('b').textContent = '—'; row.classList.remove('selected'); });
        decision.textContent = 'No AI decision yet';
        qDescription.textContent = mode === 'human'
            ? 'Switch to AI mode to inspect the agent’s action values.'
            : 'Q-values will appear here when the DQN evaluates a state.';
    }

    function physics(dt) {
        if (human.done) return;
        const main = keys.has('ArrowUp');
        const left = keys.has('ArrowLeft');
        const right = keys.has('ArrowRight');
        if (left) human.angle -= 2.3 * dt;
        if (right) human.angle += 2.3 * dt;
            human.angle *= Math.pow(.985, dt * 60);
            human.vy += 0.46 * dt * 60;
            human.vx *= Math.pow(.998, dt * 60);
        if (main && human.fuel > 0) {
            human.vx += Math.sin(human.angle) * 0.12 * dt * 60;
            human.vy -= Math.cos(human.angle) * 0.62 * dt * 60;
            human.fuel -= 8.5 * dt;
        }
        human.x += human.vx * dt * 60;
        human.y += human.vy * dt * 60;
        if (human.x < 32) { human.x = 32; human.vx *= -.45; }
        if (human.x > W - 32) { human.x = W - 32; human.vx *= -.45; }
        if (human.y + LANDER_FEET_OFFSET >= LANDING_SURFACE_Y) {
            human.y = LANDING_SURFACE_Y - LANDER_FEET_OFFSET;
            human.done = true;
            const soft = Math.abs(human.vy) < 2.5 && Math.abs(human.vx) < 2.5 && Math.abs(human.angle) < .25;
            const onPad = Math.abs(human.x - PAD_CENTER_X) < PAD_WIDTH/2;
            human.landed = soft && onPad;
            human.reward = human.landed ? 200 : -100;
            episodeState.textContent = human.landed ? 'LANDED' : 'CRASHED';
            message.classList.remove('hide');
            message.innerHTML = human.landed
                ? '<strong>Successful landing! <i class="fa-solid fa-rocket" aria-hidden="true"></i></strong><span>Restart to try again.</span>'
                : '<strong>Crash landing.</strong>' +
                '<span>↑ Slow your descent.<br>' +
                '← / → Bring angle near 0°.<br>' +
                'Land with low speed on the pad.</span>';
            setStatus(human.landed ? 'Landing successful' : 'Episode ended', human.landed ? 'ok' : 'bad');
        }
        updateTelemetry(human);
    }

    function terrainHeightAt(x) {
        const nx = x / W;
    
        // Large lunar hills and valleys.
        let height =
            30 * Math.sin(nx * Math.PI * 2.1 + 0.4) +
            16 * Math.sin(nx * Math.PI * 5.2 - 0.8) +
            7 * Math.sin(nx * Math.PI * 12.0 + 1.6);
    
        // Keep the landing-pad area relatively flat.
        const padDistance = Math.abs(x - PAD_CENTER_X);
    
        if (padDistance < PAD_TERRAIN_HALF) {
            const t = padDistance / PAD_TERRAIN_HALF;
            height *= t * t;
        }
    
        return TERRAIN_BASE_Y - height;
    }

    function drawStars() {
        const sky = ctx.createLinearGradient(0, 0, 0, H * 0.8);
    
        sky.addColorStop(0, '#02070d');
        sky.addColorStop(0.6, '#06111a');
        sky.addColorStop(1, '#0a1720');
    
        ctx.fillStyle = sky;
        ctx.fillRect(0, 0, W, H);
    
        // Deterministic stars.
        // They stay fixed instead of changing every animation frame.
        for (let i = 0; i < 110; i++) {
            const x = (i * 137 + 31) % W;
            const y = (i * 73 + 19) % 335;
    
            const size =
                i % 11 === 0 ? 1.8 :
                i % 4 === 0 ? 1.2 :
                0.8;
    
            const alpha = i % 7 === 0 ? 0.9 : 0.55;
    
            ctx.fillStyle = `rgba(190, 218, 228, ${alpha})`;
            ctx.fillRect(x, y, size, size);
        }
    
        drawEarth();
    }

    function drawEarth() {
        const x = W - 92;
        const y = 82;
        const r = 41;
    
        // Outer atmospheric glow
        const glow = ctx.createRadialGradient(x, y, r * 0.55, x, y, r * 1.8);
        glow.addColorStop(0, 'rgba(65, 170, 235, 0.18)');
        glow.addColorStop(0.55, 'rgba(40, 120, 190, 0.08)');
        glow.addColorStop(1, 'rgba(0, 0, 0, 0)');
    
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(x, y, r * 1.8, 0, Math.PI * 2);
        ctx.fill();
    
        // Ocean
        const ocean = ctx.createRadialGradient(
            x - 12, y - 14, 5,
            x, y, r
        );
    
        ocean.addColorStop(0, '#3f9bd0');
        ocean.addColorStop(0.55, '#17649a');
        ocean.addColorStop(1, '#092c49');
    
        ctx.fillStyle = ocean;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
    
        // Everything below is clipped to Earth
        ctx.save();
    
        ctx.beginPath();
        ctx.arc(x, y, r - 1, 0, Math.PI * 2);
        ctx.clip();
    
        // --------------------------------------------------
        // LAND MASSES
        // --------------------------------------------------
    
        ctx.fillStyle = '#79a86b';
    
        // North America
        ctx.beginPath();
        ctx.moveTo(x - 29, y - 17);
        ctx.bezierCurveTo(x - 37, y - 25, x - 31, y - 33, x - 20, y - 32);
        ctx.bezierCurveTo(x - 13, y - 35, x - 7, y - 27, x - 9, y - 20);
        ctx.bezierCurveTo(x - 14, y - 17, x - 18, y - 12, x - 24, y - 10);
        ctx.bezierCurveTo(x - 30, y - 10, x - 34, y - 13, x - 29, y - 17);
        ctx.fill();
    
        // South America
        ctx.beginPath();
        ctx.moveTo(x - 10, y - 5);
        ctx.bezierCurveTo(x - 4, y - 2, x - 3, y + 7, x - 7, y + 14);
        ctx.bezierCurveTo(x - 10, y + 20, x - 15, y + 27, x - 17, y + 20);
        ctx.bezierCurveTo(x - 19, y + 12, x - 16, y + 5, x - 13, y);
        ctx.closePath();
        ctx.fill();
    
        // Europe + Asia
        ctx.beginPath();
        ctx.moveTo(x + 2, y - 18);
        ctx.bezierCurveTo(x + 9, y - 23, x + 17, y - 25, x + 25, y - 20);
        ctx.bezierCurveTo(x + 34, y - 19, x + 38, y - 12, x + 34, y - 6);
        ctx.bezierCurveTo(x + 29, y - 1, x + 20, y - 3, x + 14, y - 5);
        ctx.bezierCurveTo(x + 9, y - 7, x + 4, y - 9, x + 0, y - 13);
        ctx.closePath();
        ctx.fill();
    
        // Africa
        ctx.beginPath();
        ctx.moveTo(x + 3, y - 3);
        ctx.bezierCurveTo(x + 11, y - 5, x + 18, y, x + 17, y + 7);
        ctx.bezierCurveTo(x + 16, y + 15, x + 12, y + 23, x + 7, y + 27);
        ctx.bezierCurveTo(x + 1, y + 23, x - 1, y + 15, x + 0, y + 8);
        ctx.bezierCurveTo(x - 1, y + 3, x - 1, y, x + 3, y - 3);
        ctx.fill();
    
        // Australia
        ctx.beginPath();
        ctx.moveTo(x + 24, y + 14);
        ctx.bezierCurveTo(x + 32, y + 12, x + 37, y + 16, x + 34, y + 22);
        ctx.bezierCurveTo(x + 30, y + 28, x + 22, y + 27, x + 19, y + 22);
        ctx.bezierCurveTo(x + 18, y + 18, x + 20, y + 15, x + 24, y + 14);
        ctx.fill();
    
        // Greenland
        ctx.beginPath();
        ctx.moveTo(x - 3, y - 35);
        ctx.bezierCurveTo(x + 4, y - 40, x + 13, y - 39, x + 16, y - 34);
        ctx.bezierCurveTo(x + 12, y - 29, x + 5, y - 28, x - 1, y - 30);
        ctx.closePath();
        ctx.fill();
    
        // Small island groups
        ctx.fillStyle = '#6f9d63';
    
        ctx.beginPath();
        ctx.arc(x + 30, y - 27, 2.2, 0, Math.PI * 2);
        ctx.fill();
    
        ctx.beginPath();
        ctx.arc(x + 34, y - 23, 1.5, 0, Math.PI * 2);
        ctx.fill();
    
        ctx.beginPath();
        ctx.arc(x - 25, y + 5, 1.8, 0, Math.PI * 2);
        ctx.fill();
    
        // --------------------------------------------------
        // CLOUDS
        // --------------------------------------------------
    
        ctx.fillStyle = 'rgba(255, 255, 255, 0.78)';
    
        function cloud(cx, cy, scale) {
            ctx.beginPath();
    
            ctx.arc(
                cx - 7 * scale,
                cy + 1 * scale,
                4 * scale,
                0,
                Math.PI * 2
            );
    
            ctx.arc(
                cx,
                cy - 2 * scale,
                5 * scale,
                0,
                Math.PI * 2
            );
    
            ctx.arc(
                cx + 7 * scale,
                cy + 1 * scale,
                4 * scale,
                0,
                Math.PI * 2
            );
    
            ctx.fillRect(
                cx - 10 * scale,
                cy,
                20 * scale,
                5 * scale
            );
    
            ctx.fill();
        }
    
        cloud(x - 21, y - 22, 0.8);
        cloud(x + 18, y - 11, 0.65);
        cloud(x - 2, y + 16, 0.75);
        cloud(x + 25, y + 7, 0.5);
    
        ctx.restore();
    
        // Subtle atmospheric rim
        ctx.strokeStyle = 'rgba(107, 194, 239, 0.65)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.stroke();
    }

    function drawDistantMountains() {
        const horizon = H * 0.68;
    
        // Far mountain range.
        ctx.fillStyle = '#151f26';
    
        ctx.beginPath();
        ctx.moveTo(0, horizon + 30);
    
        const farPeaks = [
            [0.00, 25],
            [0.08, -18],
            [0.16, 12],
            [0.25, -35],
            [0.34, 5],
            [0.43, -26],
            [0.52, 8],
            [0.62, -22],
            [0.71, 18],
            [0.81, -38],
            [0.90, 2],
            [1.00, -25]
        ];
    
        for (const [ratio, peak] of farPeaks) {
            ctx.lineTo(
                W * ratio,
                horizon + peak
            );
        }
    
        ctx.lineTo(W, TERRAIN_BASE_Y + 20);
        ctx.lineTo(0, TERRAIN_BASE_Y + 20);
        ctx.closePath();
    
        ctx.fill();
    
        // Closer, more dramatic ridge.
        ctx.fillStyle = '#202c33';
    
        ctx.beginPath();
        ctx.moveTo(0, horizon + 75);
    
        const nearPeaks = [
            [0.00, 42],
            [0.06, 15],
            [0.12, 57],
            [0.19, 18],
            [0.25, 68],
            [0.32, 34],
            [0.39, 72],
            [0.47, 27],
            [0.55, 63],
            [0.64, 38],
            [0.72, 71],
            [0.80, 29],
            [0.89, 67],
            [0.95, 18],
            [1.00, 45]
        ];
    
        for (const [ratio, peak] of nearPeaks) {
            ctx.lineTo(
                W * ratio,
                horizon + peak
            );
        }
    
        ctx.lineTo(W, TERRAIN_BASE_Y + 25);
        ctx.lineTo(0, TERRAIN_BASE_Y + 25);
        ctx.closePath();
    
        ctx.fill();
    
        // Mountain highlights.
        ctx.strokeStyle = 'rgba(156, 177, 184, 0.18)';
        ctx.lineWidth = 2;
    
        const ridges = [
            [0.03, horizon + 42, 0.12, horizon + 15],
            [0.19, horizon + 18, 0.25, horizon + 68],
            [0.39, horizon + 72, 0.47, horizon + 27],
            [0.72, horizon + 71, 0.80, horizon + 29],
            [0.89, horizon + 67, 0.95, horizon + 18]
        ];
    
        for (const [x1, y1, x2, y2] of ridges) {
            ctx.beginPath();
    
            ctx.moveTo(
                W * x1,
                y1
            );
    
            ctx.lineTo(
                W * x2,
                y2
            );
    
            ctx.stroke();
        }
    }

    function drawLunarGround() {
        ctx.beginPath();
    
        ctx.moveTo(0, H);
        ctx.lineTo(0, terrainHeightAt(0));
    
        for (let x = 0; x <= W; x += 6) {
            ctx.lineTo(
                x,
                terrainHeightAt(x)
            );
        }
    
        ctx.lineTo(W, H);
        ctx.closePath();
    
        // Ground shading.
        const ground = ctx.createLinearGradient(
            0,
            TERRAIN_BASE_Y - 20,
            0,
            H
        );
    
        ground.addColorStop(0, '#5b6468');
        ground.addColorStop(0.35, '#3d474c');
        ground.addColorStop(1, '#151d21');
    
        ctx.fillStyle = ground;
        ctx.fill();
    
        // Terrain edge.
        ctx.beginPath();
    
        for (let x = 0; x <= W; x += 6) {
            const y = terrainHeightAt(x);
    
            if (x === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
    
        ctx.strokeStyle = 'rgba(174, 185, 188, 0.42)';
        ctx.lineWidth = 1.2;
        ctx.stroke();
    
        // Small rocks.
        for (let i = 0; i < 42; i++) {
            const x = (i * 211 + 53) % W;
    
            const y =
                terrainHeightAt(x) +
                9 +
                ((i * 17) % 32);
    
            const size =
                2 +
                ((i * 13) % 8);
    
            ctx.fillStyle =
                i % 3 === 0
                    ? '#273136'
                    : '#323d42';
    
            ctx.beginPath();
    
            ctx.moveTo(
                x - size,
                y + size * 0.6
            );
    
            ctx.lineTo(
                x - size * 0.45,
                y - size * 0.5
            );
    
            ctx.lineTo(
                x + size * 0.25,
                y - size
            );
    
            ctx.lineTo(
                x + size,
                y + size * 0.45
            );
    
            ctx.closePath();
            ctx.fill();
        }
    
        // Large foreground rocks.
        const boulders = [
            [W * 0.08, 24],
            [W * 0.25, 16],
            [W * 0.73, 20],
            [W * 0.92, 29]
        ];
    
        for (const [x, size] of boulders) {
            const y =
                terrainHeightAt(x) +
                size * 0.75;
    
            ctx.fillStyle = '#242e33';
    
            ctx.beginPath();
    
            ctx.moveTo(
                x - size,
                y + size
            );
    
            ctx.lineTo(
                x - size * 0.75,
                y - size * 0.25
            );
    
            ctx.lineTo(
                x - size * 0.15,
                y - size
            );
    
            ctx.lineTo(
                x + size * 0.65,
                y - size * 0.55
            );
    
            ctx.lineTo(
                x + size,
                y + size
            );
    
            ctx.closePath();
            ctx.fill();
        }
    }

    function drawPad() {
        const y = TERRAIN_BASE_Y;
    
        // Dark landing area.
        ctx.fillStyle = 'rgba(12, 18, 21, 0.9)';
    
        ctx.beginPath();
    
        ctx.ellipse(
            PAD_CENTER_X,
            y - 1,
            PAD_WIDTH * 0.78,
            12,
            0,
            0,
            Math.PI * 2
        );
    
        ctx.fill();
    
        // Main landing ring.
        ctx.strokeStyle = '#73e0b1';
        ctx.lineWidth = 3;
    
        ctx.beginPath();
    
        ctx.ellipse(
            PAD_CENTER_X,
            y - 2,
            PAD_WIDTH / 2,
            7,
            0,
            0,
            Math.PI * 2
        );
    
        ctx.stroke();
    
        // Pad center line.
        ctx.strokeStyle = 'rgba(115, 224, 177, 0.25)';
        ctx.lineWidth = 1;
    
        ctx.beginPath();
    
        ctx.moveTo(
            PAD_CENTER_X - PAD_WIDTH * 0.72,
            y + 1
        );
    
        ctx.lineTo(
            PAD_CENTER_X + PAD_WIDTH * 0.72,
            y + 1
        );
    
        ctx.stroke();
    
        // Label.
        ctx.fillStyle = '#73e0b1';
        ctx.font = '10px system-ui';
        ctx.textAlign = 'center';
    
        ctx.fillText(
            'LANDING PAD',
            PAD_CENTER_X,
            y + 19
        );
    }

    function drawEnvironment() {
        drawStars();
        drawDistantMountains();
        drawLunarGround();
        drawPad();
    }

    function drawLander(
        x,
        y,
        angle,
        thrust = false,
        leftThrust = false,
        rightThrust = false
    ) {
        ctx.save();
    
        ctx.translate(x, y);
        ctx.rotate(angle);
    
        /*
         * Apollo-style Lunar Module
         *
         * Coordinate system:
         *      -Y = top of lander
         *       Y = engine / landing gear
         *
         * The entire vehicle is drawn around (0, 0),
         * so the existing physics and rotation continue
         * to work exactly as before.
         */
    
        // ---------------------------------------------------------
        // 1. Landing legs - draw these first so the body sits over them
        // ---------------------------------------------------------
    
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
    
        ctx.strokeStyle = '#9aa8ad';
        ctx.lineWidth = 2;
    
        // Left rear leg
        ctx.beginPath();
        ctx.moveTo(-11, 12);
        ctx.lineTo(-25, 29);
        ctx.lineTo(-30, 42);
        ctx.stroke();
    
        // Right rear leg
        ctx.beginPath();
        ctx.moveTo(11, 12);
        ctx.lineTo(25, 29);
        ctx.lineTo(30, 42);
        ctx.stroke();
    
        // Front landing legs
        ctx.strokeStyle = '#b4c0c3';
        ctx.lineWidth = 2.5;
    
        ctx.beginPath();
        ctx.moveTo(-8, 15);
        ctx.lineTo(-16, 30);
        ctx.lineTo(-19, 43);
        ctx.stroke();
    
        ctx.beginPath();
        ctx.moveTo(8, 15);
        ctx.lineTo(16, 30);
        ctx.lineTo(19, 43);
        ctx.stroke();
    
        // Struts / cross braces
        ctx.strokeStyle = '#66777e';
        ctx.lineWidth = 1;
    
        ctx.beginPath();
        ctx.moveTo(-24, 29);
        ctx.lineTo(-8, 23);
    
        ctx.moveTo(24, 29);
        ctx.lineTo(8, 23);
    
        ctx.stroke();
    
        // ---------------------------------------------------------
        // 2. Footpads
        // ---------------------------------------------------------
    
        function drawFootpad(px, py) {
            ctx.fillStyle = '#1b252a';
            ctx.strokeStyle = '#9aa8ad';
            ctx.lineWidth = 1.3;
    
            ctx.beginPath();
            ctx.ellipse(
                px,
                py,
                7,
                2.5,
                0,
                0,
                Math.PI * 2
            );
    
            ctx.fill();
            ctx.stroke();
    
            // Small center highlight
            ctx.fillStyle = '#c1cccf';
    
            ctx.beginPath();
            ctx.ellipse(
                px,
                py - 0.4,
                2.5,
                0.7,
                0,
                0,
                Math.PI * 2
            );
    
            ctx.fill();
        }
    
        drawFootpad(-30, 43);
        drawFootpad(30, 43);
        drawFootpad(-19, 44);
        drawFootpad(19, 44);
    
        // ---------------------------------------------------------
        // 3. Gold thermal insulation / descent stage
        // ---------------------------------------------------------
    
        const foilGradient = ctx.createLinearGradient(
            -13,
            5,
            13,
            24
        );
    
        foilGradient.addColorStop(0, '#6f5120');
        foilGradient.addColorStop(0.22, '#b88732');
        foilGradient.addColorStop(0.5, '#e1bd67');
        foilGradient.addColorStop(0.72, '#a87828');
        foilGradient.addColorStop(1, '#624719');
    
        ctx.fillStyle = foilGradient;
        ctx.strokeStyle = '#d6bd7a';
        ctx.lineWidth = 1.5;
    
        ctx.beginPath();
    
        ctx.moveTo(-14, 5);
        ctx.lineTo(14, 5);
        ctx.lineTo(12, 22);
        ctx.lineTo(0, 27);
        ctx.lineTo(-12, 22);
        ctx.closePath();
    
        ctx.fill();
        ctx.stroke();
    
        // Thermal foil panels
        ctx.strokeStyle = 'rgba(255, 231, 157, 0.55)';
        ctx.lineWidth = 0.7;
    
        for (let i = -8; i <= 8; i += 4) {
            ctx.beginPath();
            ctx.moveTo(i, 7);
            ctx.lineTo(i * 0.85, 22);
            ctx.stroke();
        }
    
        // Horizontal foil bands
        ctx.beginPath();
        ctx.moveTo(-13, 12);
        ctx.lineTo(13, 12);
    
        ctx.moveTo(-12, 18);
        ctx.lineTo(12, 18);
    
        ctx.stroke();
    
        // ---------------------------------------------------------
        // 4. Descent engine
        // ---------------------------------------------------------
    
        ctx.fillStyle = '#252f34';
        ctx.strokeStyle = '#8b989c';
        ctx.lineWidth = 1.2;
    
        ctx.beginPath();
        ctx.moveTo(-5, 24);
        ctx.lineTo(5, 24);
        ctx.lineTo(4, 30);
        ctx.lineTo(-4, 30);
        ctx.closePath();
    
        ctx.fill();
        ctx.stroke();
    
        // Engine nozzle
        ctx.fillStyle = '#12191d';
    
        ctx.beginPath();
        ctx.moveTo(-4, 28);
        ctx.lineTo(4, 28);
        ctx.lineTo(3, 33);
        ctx.lineTo(-3, 33);
        ctx.closePath();
    
        ctx.fill();
    
        // ---------------------------------------------------------
        // 5. Main engine flame
        // ---------------------------------------------------------
    
        if (thrust) {
            const flameLength = 20 + Math.random() * 10;
    
            // Outer flame
            const flameGradient = ctx.createLinearGradient(
                0,
                31,
                0,
                31 + flameLength
            );
    
            flameGradient.addColorStop(
                0,
                'rgba(255, 244, 185, 1)'
            );
    
            flameGradient.addColorStop(
                0.35,
                'rgba(255, 188, 69, 0.95)'
            );
    
            flameGradient.addColorStop(
                1,
                'rgba(255, 102, 35, 0)'
            );
    
            ctx.fillStyle = flameGradient;
    
            ctx.beginPath();
            ctx.moveTo(-5, 30);
            ctx.lineTo(
                -2,
                30 + flameLength * 0.45
            );
            ctx.lineTo(
                0,
                30 + flameLength
            );
            ctx.lineTo(
                3,
                30 + flameLength * 0.45
            );
            ctx.lineTo(5, 30);
            ctx.closePath();
    
            ctx.fill();
    
            // Hot inner core
            ctx.fillStyle = 'rgba(255, 246, 196, 0.95)';
    
            ctx.beginPath();
            ctx.moveTo(-2.2, 30);
            ctx.lineTo(
                0,
                30 + flameLength * 0.55
            );
            ctx.lineTo(
                2.2,
                30
            );
            ctx.closePath();
    
            ctx.fill();
        }
    
        // ---------------------------------------------------------
        // 6. Upper ascent/cabin module
        // ---------------------------------------------------------
    
        const cabinGradient = ctx.createLinearGradient(
            -13,
            -21,
            13,
            6
        );
    
        cabinGradient.addColorStop(
            0,
            '#f3f6f4'
        );
    
        cabinGradient.addColorStop(
            0.55,
            '#cbd4d3'
        );
    
        cabinGradient.addColorStop(
            1,
            '#7f8d91'
        );
    
        ctx.fillStyle = cabinGradient;
        ctx.strokeStyle = '#697a80';
        ctx.lineWidth = 1.5;
    
        // Cabin body
        ctx.beginPath();
    
        ctx.moveTo(-12, 5);
        ctx.lineTo(-15, -7);
        ctx.lineTo(-9, -19);
        ctx.lineTo(0, -23);
        ctx.lineTo(9, -19);
        ctx.lineTo(15, -7);
        ctx.lineTo(12, 5);
        ctx.closePath();
    
        ctx.fill();
        ctx.stroke();
    
        // ---------------------------------------------------------
        // 7. Cockpit windows
        // ---------------------------------------------------------
    
        ctx.fillStyle = '#172a35';
        ctx.strokeStyle = '#91a8b0';
        ctx.lineWidth = 1;
    
        // Left window
        ctx.beginPath();
    
        ctx.moveTo(-10, -8);
        ctx.lineTo(-5, -14);
        ctx.lineTo(-2, -12);
        ctx.lineTo(-3, -5);
        ctx.closePath();
    
        ctx.fill();
        ctx.stroke();
    
        // Right window
        ctx.beginPath();
    
        ctx.moveTo(10, -8);
        ctx.lineTo(5, -14);
        ctx.lineTo(2, -12);
        ctx.lineTo(3, -5);
        ctx.closePath();
    
        ctx.fill();
        ctx.stroke();
    
        // Center front window
        ctx.beginPath();
    
        ctx.moveTo(-2, -13);
        ctx.lineTo(2, -13);
        ctx.lineTo(3, -5);
        ctx.lineTo(-3, -5);
        ctx.closePath();
    
        ctx.fill();
        ctx.stroke();
    
        // Window reflections
        ctx.strokeStyle = 'rgba(210, 240, 245, 0.45)';
        ctx.lineWidth = 0.8;
    
        ctx.beginPath();
        ctx.moveTo(-8, -9);
        ctx.lineTo(-5, -12);
    
        ctx.moveTo(5, -12);
        ctx.lineTo(8, -9);
    
        ctx.stroke();
    
        // ---------------------------------------------------------
        // 8. Top docking / antenna assembly
        // ---------------------------------------------------------
    
        ctx.strokeStyle = '#9daeb3';
        ctx.lineWidth = 1.2;
    
        // Antenna mast
        ctx.beginPath();
    
        ctx.moveTo(0, -22);
        ctx.lineTo(0, -31);
    
        ctx.stroke();
    
        // Dish
        ctx.fillStyle = '#d8dddd';
        ctx.strokeStyle = '#7c8c91';
    
        ctx.beginPath();
    
        ctx.arc(
            0,
            -31,
            4.5,
            Math.PI,
            Math.PI * 2
        );
    
        ctx.fill();
        ctx.stroke();
    
        // Antenna tip
        ctx.fillStyle = '#bfc9cc';
    
        ctx.beginPath();
        ctx.arc(
            0,
            -34,
            1.2,
            0,
            Math.PI * 2
        );
    
        ctx.fill();
    
        // ---------------------------------------------------------
        // 9. Side equipment boxes
        // ---------------------------------------------------------
    
        ctx.fillStyle = '#5e6b70';
        ctx.strokeStyle = '#a9b5b8';
        ctx.lineWidth = 0.8;
    
        // Left box
        ctx.fillRect(-18, -1, 5, 7);
        ctx.strokeRect(-18, -1, 5, 7);
    
        // Right box
        ctx.fillRect(13, -1, 5, 7);
        ctx.strokeRect(13, -1, 5, 7);
    
        // Small side antenna rods
        ctx.strokeStyle = '#a8b7bb';
        ctx.lineWidth = 1;
    
        ctx.beginPath();
    
        ctx.moveTo(-18, 2);
        ctx.lineTo(-23, -2);
    
        ctx.moveTo(18, 2);
        ctx.lineTo(23, -2);
    
        ctx.stroke();
    
        // ---------------------------------------------------------
        // 10. RCS / attitude-control flames
        // ---------------------------------------------------------
    
        if (leftThrust) {
            const length = 9 + Math.random() * 5;
    
            ctx.fillStyle = '#ffb94f';
    
            ctx.beginPath();
    
            ctx.moveTo(-17, -1);
            ctx.lineTo(-17 - length, 1);
            ctx.lineTo(-17, 3);
    
            ctx.closePath();
            ctx.fill();
    
            ctx.fillStyle = 'rgba(255, 242, 174, 0.85)';
    
            ctx.beginPath();
    
            ctx.moveTo(-17, 0);
            ctx.lineTo(-17 - length * 0.55, 1);
            ctx.lineTo(-17, 2);
    
            ctx.closePath();
            ctx.fill();
        }
    
        if (rightThrust) {
            const length = 9 + Math.random() * 5;
    
            ctx.fillStyle = '#ffb94f';
    
            ctx.beginPath();
    
            ctx.moveTo(17, -1);
            ctx.lineTo(17 + length, 1);
            ctx.lineTo(17, 3);
    
            ctx.closePath();
            ctx.fill();
    
            ctx.fillStyle = 'rgba(255, 242, 174, 0.85)';
    
            ctx.beginPath();
    
            ctx.moveTo(17, 0);
            ctx.lineTo(17 + length * 0.55, 1);
            ctx.lineTo(17, 2);
    
            ctx.closePath();
            ctx.fill();
        }
    
        // ---------------------------------------------------------
        // 11. Tiny navigation lights
        // ---------------------------------------------------------
    
        ctx.fillStyle = '#73e0b1';
    
        ctx.beginPath();
        ctx.arc(-14, -2, 1.2, 0, Math.PI * 2);
        ctx.fill();
    
        ctx.fillStyle = '#ffcc66';
    
        ctx.beginPath();
        ctx.arc(14, -2, 1.2, 0, Math.PI * 2);
        ctx.fill();
    
        ctx.restore();
    }

    function renderHuman() {
        drawEnvironment();
        drawLander(human.x,human.y,human.angle,keys.has('ArrowUp') && !human.done,keys.has('ArrowLeft') && !human.done,keys.has('ArrowRight') && !human.done);
        if (!human.done) { ctx.fillStyle='#4e6572'; ctx.font='11px system-ui'; ctx.fillText('YOU',human.x-25,human.y-30); }
    }

    function getAIVisualPosition(frame) {
        const s = frame.state || [];

        const rawX = Number.isFinite(frame.x)
            ? Number(frame.x)
            : W / 2 + Number(s[0] || 0) * 120;

        const rawY = Number.isFinite(frame.y)
            ? Number(frame.y)
            : H - 80 - Math.max(0, Number(s[1] || 0)) * 170;

        return {
            x: rawX + aiVisualOffsetX,
            y: rawY + aiVisualOffsetY
        };
    }

    function updateAIFrame(frame) {
        if (!frame) return;
        const s = frame.state || {};
        const { x, y } = getAIVisualPosition(frame);
        const angle = Number(frame.angle ?? s[4] ?? 0);
        const vx = Number(frame.vx ?? s[2] ?? 0);
        const vy = Number(frame.vy ?? s[3] ?? 0);
        const altitude = Number(frame.altitude ?? s[1] ?? 0);
        const contacts = getLegContact(s);
    
        updateTelemetry({
            x: x,
            y: y,
            altitude: altitude,
            vx: vx,
            vy: vy,
            angle: angle,
            fuel: null,
            contacts: contacts,
            reward: Number(frame.reward ?? 0)
        }, true);
    
        updateQValues(frame.q_values, frame.action, 'DQN');
    
        if (frame.done) {
            episodeState.textContent = frame.success ? 'LANDED' : 'ENDED';
            message.classList.remove('hide');
            message.innerHTML = frame.success
                ? '<strong>Agent landed successfully. <i class="fa-solid fa-rocket" aria-hidden="true"></i></strong><span>Evaluation complete.</span>'
                : '<strong>Episode ended.</strong><span>Evaluation complete.</span>';
        } else {
            message.classList.add('hide');
        }
    
        renderAI(frame);
    }

    function renderAI(frame) {
        drawEnvironment();
        const s = frame.state || [];
        const { x, y } = getAIVisualPosition(frame);

        const drawX = Math.max(30, Math.min(W - 30, x));
        const drawY = Math.max(40, Math.min(H - 50, y));
        drawLander(
            drawX,
            drawY,
            Number(frame.angle ?? s[4] ?? 0),
            Number(frame.action) === 2,
            Number(frame.action) === 0,
            Number(frame.action) === 1
        );
        ctx.fillStyle = '#73e0b1';
        ctx.font = '11px system-ui';

        ctx.fillText(
            'DQN',
            Math.max(20, Math.min(W - 50, drawX - 25)),
            Math.max(20, drawY - 34)
        );
    }

    function loop(now) {
        const dt = Math.min(.032,(now-last)/1000); last=now;
        if (mode === 'human') { physics(dt * HUMAN_TIME_SCALE); renderHuman(); }
        raf=requestAnimationFrame(loop);
    }

    function playAIResult() {
        if (!aiReplay?.frames?.length) {
            episodeState.textContent = 'ERROR';
            setStatus('No evaluation frames received', 'bad');
            return;
        } 

        aiIndex = 0;

        const firstFrame = aiReplay.frames[0];

        if (firstFrame) {
            const rawX = Number.isFinite(firstFrame.x)
                ? Number(firstFrame.x)
                : W / 2 + Number(firstFrame.state?.[0] || 0) * 120;

            const rawY = Number.isFinite(firstFrame.y)
                ? Number(firstFrame.y)
                : H - 80 - Math.max(0, Number(firstFrame.state?.[1] || 0)) * 170;

            aiVisualOffsetX = LANDER_START_X - rawX;
            aiVisualOffsetY = LANDER_START_Y - rawY;
        } else {
            aiVisualOffsetX = 0;
            aiVisualOffsetY = 0;
        }


        const fps = Number(aiReplay.fps || 18);
        const interval = 1000 / fps;

        setStatus('AI evaluation running');

        aiTimer = setInterval(() => {
            const frame = aiReplay.frames[aiIndex];

            if (!frame) {
                clearInterval(aiTimer);
                aiTimer = null;
                return;
            }

            updateAIFrame(frame);
            aiIndex++;

            if (aiIndex >= aiReplay.frames.length) {
                clearInterval(aiTimer);
                aiTimer = null;

                setStatus(
                    aiReplay.success
                        ? 'AI landing successful'
                        : 'AI episode complete',
                    aiReplay.success ? 'ok' : 'warn'
                );

                // Replay is now completely finished.
                if (mode === 'ai') {
                    startAIBtn.disabled = false;
                    restartBtn.disabled = false;
                }
            }
        }, interval);
    }

    function renderTrainingChart() {
        const container = document.getElementById('rewardChart');

        if (!trainingResults?.episode_history?.rewards?.length) {
            container.innerHTML = `
                <div class="chart-loading">
                    Training data unavailable.
                </div>
            `;
            return;
        }

        const rawRewards = trainingResults.episode_history.rewards;
        const rawEpisodes = trainingResults.episode_history.episodes;
        const rawSuccess = trainingResults.episode_history.success;
        
        if (!Array.isArray(rawRewards) || rawRewards.length === 0) {
            container.innerHTML = `
                <div class="chart-loading">
                    Training data unavailable.
                </div>
            `;
            return;
        }

        const rewards = rawRewards.map(Number);
        const success = Array.isArray(rawSuccess) ? rawSuccess.map(Boolean) : null;
        const episodes = Array.isArray(rawEpisodes)
            ? rawEpisodes.map(Number)
            : rewards.map((_, index) => index + 1);

        if (
            rewards.length !== episodes.length || !success ||
            success.length !== episodes.length ||
            rewards.some(value => !Number.isFinite(value)) ||
            episodes.some(value => !Number.isFinite(value))
        ) {
            console.error('Invalid training chart data:', {
                rewards,
                episodes,
                success
            });

            container.innerHTML = `
                <div class="chart-loading">
                    Training data is invalid.
                </div>
            `;
            return;
        }

        const successRate = [];

        let successfulEpisodes = 0;

        success.forEach((successful, index) => {
            if (successful) {
                successfulEpisodes++;
            }

            successRate.push(
                (successfulEpisodes / (index + 1)) * 100
            );
        });

        const width = 1000;
        const height = 430;

        const margin = {top: 30, right: 24, bottom: 55, left: 65};

        const chartWidth = width - margin.left - margin.right;
        const chartHeight = height - margin.top - margin.bottom;

        const minReward = Math.min(...rewards);
        const maxReward = Math.max(...rewards);

        const padding = Math.max(20, (maxReward - minReward) * 0.08);

        const yMin = minReward - padding;
        const yMax = maxReward + padding;

        const x = episode => {
            return margin.left +
                ((episode - 1) / Math.max(1, episodes.length - 1)) *
                chartWidth;
        };

        const yRange = Math.max(1, yMax - yMin);
        const y = reward => {
            return margin.top +
                (1 - (reward - yMin) / yRange) *
                chartHeight;
        };

        const rewardPoints = rewards
            .map((reward, index) =>
                `${x(episodes[index])},${y(reward)}`
            )
            .join(' ');

        const configuredWindow = Number(trainingResults.episode_history.average_window);
        
        const windowSize = Math.max(
            1,
            Math.min(
                Number.isFinite(configuredWindow)
                    ? configuredWindow
                    : 100,
                rewards.length
            )
        );

        const movingAverage = [];

        for (let i = windowSize - 1; i < rewards.length; i++) {
            let sum = 0;

            for (let j = i - windowSize + 1; j <= i; j++) {
                sum += rewards[j];
            }
            movingAverage.push({episode: episodes[i], reward: sum / windowSize});
        }

        const averagePoints = movingAverage
            .map(point => `${x(point.episode)},${y(point.reward)}`
            )
            .join(' ');

        const zeroY = yMin <= 0 && yMax >= 0 ? y(0) : null;

        const successY = value => margin.top + (1 - value / 100) * chartHeight;

        const successPoints = successRate.map((value, index) => {
            const x =
                margin.left +
                (index / Math.max(1, successRate.length - 1)) *
                chartWidth;
        
            const y = successY(value);
        
            return `${x.toFixed(2)},${y.toFixed(2)}`;
        }).join(' ');

        const gridLines = 5;

        let grid = '';

        for (let i = 0; i <= gridLines; i++) {
            const value =
                yMin +
                ((yMax - yMin) * i / gridLines);

            const py = y(value);

            grid += `
                <line
                    x1="${margin.left}"
                    y1="${py}"
                    x2="${width - margin.right}"
                    y2="${py}"
                    class="chart-grid"
                />

                <text
                    x="${margin.left - 10}"
                    y="${py + 4}"
                    text-anchor="end"
                    class="chart-axis-label"
                >
                    ${Math.round(value)}
                </text>

                <text
                    x="${width - 18}"
                    y="${margin.top + 4}"
                    text-anchor="end"
                    class="chart-axis-label"
                >
                    100%
                </text>

                <text
                    x="${width - 18}"
                    y="${margin.top + chartHeight / 2 + 4}"
                    text-anchor="end"
                    class="chart-axis-label"
                >
                    50%
                </text>

                <text
                    x="${width - 18}"
                    y="${margin.top + chartHeight + 4}"
                    text-anchor="end"
                    class="chart-axis-label"
                >
                    0%
                </text>
            `;
        }

        let xLabels = '';

        const labelCount = 6;

        for (let i = 0; i < labelCount; i++) {
            const index = Math.round(
                (episodes.length - 1) *
                (i / (labelCount - 1))
            );

            const episode = episodes[index];
            const px = x(episode);

            xLabels += `
                <text
                    x="${px}"
                    y="${height - 20}"
                    text-anchor="middle"
                    class="chart-axis-label"
                >
                    ${episode}
                </text>
            `;
        }

        container.innerHTML = `
            <svg
                class="reward-chart-svg"
                viewBox="0 0 ${width} ${height}"
                preserveAspectRatio="none"
                role="img"
                aria-label="Lunar Lander training reward curve"
            >
                ${grid}

                ${
                    zeroY !== null
                        ? `
                            <line
                                x1="${margin.left}"
                                y1="${zeroY}"
                                x2="${width - margin.right}"
                                y2="${zeroY}"
                                class="chart-zero"
                            />
                        `
                        : ''
                }

                <polyline
                    points="${rewardPoints}"
                    class="reward-line"
                />

                <polyline
                    points="${averagePoints}"
                    class="average-line"
                />

                <polyline
                    points="${successPoints}"
                    class="success-line"
                />

                ${xLabels}

                <text
                    x="${width / 2}"
                    y="${height - 12}"
                    text-anchor="middle"
                    class="chart-axis-title"
                >
                    Episode
                </text>

                <text
                    x="16"
                    y="${height / 2}"
                    text-anchor="middle"
                    transform="rotate(-90 16 ${height / 2})"
                    class="chart-axis-title"
                >
                    Total reward
                </text>

                <text
                    x="${width - 16}"
                    y="${margin.top - 10}"
                    text-anchor="end"
                    class="chart-axis-title"
                >
                    Success rate
                </text>

                <g class="chart-legend">
                    <line
                        x1="${margin.left}"
                        y1="18"
                        x2="${margin.left + 24}"
                        y2="18"
                        class="reward-line"
                    />
                    <text
                        x="${margin.left + 32}"
                        y="22"
                        class="chart-legend-label"
                    >
                        Episode reward
                    </text>

                    <line
                        x1="${margin.left + 150}"
                        y1="18"
                        x2="${margin.left + 174}"
                        y2="18"
                        class="average-line"
                    />
                    <text
                        x="${margin.left + 182}"
                        y="22"
                        class="chart-legend-label"
                    >
                        ${windowSize}-episode moving average
                    </text>

                    <line
                        x1="${margin.left + 360}"
                        y1="18"
                        x2="${margin.left + 384}"
                        y2="18"
                        class="success-line"
                    />

                    <text
                        x="${margin.left + 392}"
                        y="22"
                        class="chart-legend-label"
                    >
                        Cumulative success rate
                    </text>
                </g>
            </svg>
        `;
    }

    async function runAI() {
        clearInterval(aiTimer);
        aiTimer = null;

        const requestId = ++aiRequestId;

        message.classList.add('hide');
        episodeState.textContent = 'RUNNING';
        setStatus('Running DQN evaluation...');

        startAIBtn.disabled = true;
        restartBtn.disabled = true;

        try {
            const response = await fetch(`/api/evaluate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    seed: 0
                })
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(
                    error.detail || `Evaluation failed (${response.status})`
                );
            }

            const result = await response.json();

            if(requestId !== aiRequestId || mode !== 'ai'){
                return;
            }

            aiReplay = result;
            aiIndex = 0;

            playAIResult();

        } catch (error) {
            console.error('AI evaluation failed:', error);

            if (requestId !== aiRequestId || mode !== 'ai') {
                return;
            }

            episodeState.textContent = 'ERROR';
            setStatus('AI evaluation failed', 'bad');

            message.classList.remove('hide');
            message.innerHTML =
                '<strong>AI evaluation failed.</strong>' +
                `<span>${error.message}</span>`;

            startAIBtn.disabled = false;
            restartBtn.disabled = false;
        }
    }

    async function loadTrainingResults() {
        try {
            const response = await fetch(
                `/api/training/${TRAINING_VERSION}`,
                { cache: 'no-store' }
            );

            if (!response.ok) {
                throw new Error('Training results not found');
            }

            trainingResults = await response.json();

            document.getElementById('episodes').textContent = trainingResults.training_results?.episodes_completed ?? '—';

            document.getElementById('finalAverage').textContent =
                trainingResults.training_results?.final_average_reward != null
                    ? Number(
                        trainingResults.training_results.final_average_reward
                    ).toFixed(2)
                    : '—';

            document.getElementById('successRate').textContent =
                trainingResults.training_results?.success_rate != null
                    ? `${Number(
                        trainingResults.training_results.success_rate
                    ).toFixed(1)}%`
                    : '—';

            document.getElementById('trainTime').textContent =
                trainingResults.training_results?.training_time_minutes != null
                    ? `${Number(
                        trainingResults.training_results.training_time_minutes
                    ).toFixed(1)} min`
                    : '—';

            renderTrainingChart();

        } catch (error) {
            console.error('Training results could not be loaded:', error);

            trainingResults = null;

            const container =
                document.getElementById('rewardChart');

            container.innerHTML = `
                <div class="chart-loading">
                    Training results unavailable.
                </div>
            `;
        }
    }

    async function checkAPI() {
        try {
            const response = await fetch(
                `/api/health`,
                { cache: 'no-store' }
            );

            if (!response.ok) {
                throw new Error('API unavailable');
            }

            const result = await response.json();

            if (result.status !== 'ok') {
                throw new Error('API health check failed');
            }

            qDescription.textContent = 'Real Q-values from the trained DQN will appear frame by frame.';

            if (mode === 'ai') {
                resetAIView();
            }

            apiAvailable = true;
            return true;

        } catch (error) {
            console.warn('DQN API unavailable:', error);

            if (mode === 'ai') {
                episodeState.textContent = 'API OFFLINE';
                message.classList.remove('hide');

                message.innerHTML =
                    '<strong>DQN API is offline.</strong>' +
                    '<span>Start the FastAPI server to evaluate the trained agent.</span>';

                setStatus('DQN API offline', 'warn');
            }

            apiAvailable = false;
            return false;
        }
    }

    document.querySelectorAll('.mode').forEach(btn=>btn.addEventListener('click',()=>{
        mode=btn.dataset.mode;
        updateTelemetryMode();

        document.querySelectorAll('.mode').forEach(b=>{const active=b===btn;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active));});
        modeLabel.textContent=mode==='human'?'HUMAN CONTROL':'DQN EVALUATION';
        startAIBtn.classList.toggle('hidden',mode!=='ai');
        restartBtn.innerHTML = mode === 'human'
            ? '<i class="fa-solid fa-rotate-right" aria-hidden="true"></i> Restart'
            : '<i class="fa-solid fa-rotate-right" aria-hidden="true"></i> Reset replay';
        
        cancelAnimationFrame(raf);
        aiRequestId++;

        startAIBtn.disabled = false;
        restartBtn.disabled = false;
        clearInterval(aiTimer);
        aiTimer=null;
        last=performance.now();
        
        if (mode === 'human') {
            setStatus('Human mode ready', 'ok');
            resetHuman();
        } else {
            startAIBtn.disabled = false;
            restartBtn.disabled = false;

            if (apiAvailable) {
                setStatus('DQN evaluation ready', 'ok');
            } else {
                setStatus('DQN API offline', 'warn');
            }
        
            resetAIView();
        }
        raf=requestAnimationFrame(loop);
    }));
    restartBtn.addEventListener('click',resetAll);
    startAIBtn.addEventListener('click',runAI);
    window.addEventListener('keydown',e=>{if(['ArrowUp','ArrowLeft','ArrowRight'].includes(e.key)){keys.add(e.key);e.preventDefault();}});
    window.addEventListener('keyup',e=>keys.delete(e.key));

    updateTelemetryMode();
    resetHuman();
    loadTrainingResults();
    checkAPI();
    raf=requestAnimationFrame(loop);
})();
