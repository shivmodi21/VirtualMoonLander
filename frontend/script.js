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
    const LANDER_FEET_OFFSET = 18;
    const PAD_CENTER_X = 0.55 * W;
    const PAD_WIDTH = 104;
    const keys = new Set();
    let mode = 'human';
    let raf = 0;
    let last = performance.now();
    let aiReplay = null;
    let aiIndex = 0;
    let aiTimer = null;
    let aiVisualOffsetX = 0;
    let aiVisualOffsetY = 0;
    let trainingResults = null;
    const TRAINING_VERSION = 'v1';
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
        return ['← Left', '→ Right', '🔥 Main', '○ No action'][Number(action)] ?? 'Unknown';
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

    function updateTelemetry(s) {
        const landerFeetY = s.y + LANDER_FEET_OFFSET;
        const altitude = Math.max(0, LANDING_SURFACE_Y - landerFeetY);

        document.getElementById('altitude').textContent = `${altitude.toFixed(1)} px`;
        const fuelElement = document.getElementById('fuel');
        if (s.fuel == null) {
        fuelElement.textContent = 'N/A';
        } else {
        fuelElement.textContent = `${Math.max(0, Math.min(s.fuel, 100)).toFixed(0)}%`;
        }
        document.getElementById('vy').textContent = `${s.vy.toFixed(2)}`;
        document.getElementById('vx').textContent = `${s.vx.toFixed(2)}`;
        document.getElementById('angle').textContent = `${(s.angle * 180 / Math.PI).toFixed(1)}°`;
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
        decision.textContent = `${source} selected ${actionName(selected)} with Q = ${Number(values[selected]).toFixed(2)}`;
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
                ? '<strong>Successful landing! 🚀</strong><span>Restart to try again.</span>'
                : '<strong>Crash landing.</strong>' +
                '<span>↑ Slow your descent.<br>' +
                '← / → Bring angle near 0°.<br>' +
                'Land with low speed on the pad.</span>';
            setStatus(human.landed ? 'Landing successful' : 'Episode ended', human.landed ? 'ok' : 'bad');
        }
        updateTelemetry(human);
    }

    function drawStars() {
        ctx.fillStyle = '#050b12'; ctx.fillRect(0,0,W,H);
        for (let i=0;i<75;i++) { const x=(i*137)%W, y=(i*73)%310; ctx.fillStyle=i%7===0?'#b8d4dc':'#536875'; ctx.fillRect(x,y,1.4,1.4); }
    }

    function drawPad() {
        const y = LANDING_SURFACE_Y;
        ctx.strokeStyle = '#2e4855'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke();
        ctx.strokeStyle = '#73e0b1'; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(PAD_CENTER_X-PAD_WIDTH/2,y-2); ctx.lineTo(PAD_CENTER_X+PAD_WIDTH/2,y-2); ctx.stroke();
        ctx.fillStyle = '#73e0b1'; ctx.font = '10px system-ui'; ctx.textAlign='center'; ctx.fillText('LANDING PAD',PAD_CENTER_X,y+17);
    }

    function drawLander(x,y,angle,thrust=false, leftThrust=false, rightThrust=false) {
        ctx.save(); ctx.translate(x,y); ctx.rotate(angle);
        ctx.fillStyle='#eaf4f5'; ctx.strokeStyle='#7c949e'; ctx.lineWidth=2;
        ctx.beginPath(); ctx.moveTo(0,-20); ctx.lineTo(16,-6); ctx.lineTo(12,13); ctx.lineTo(-12,13); ctx.lineTo(-16,-6); ctx.closePath(); ctx.fill(); ctx.stroke();
        ctx.fillStyle='#182a35'; ctx.beginPath(); ctx.arc(0,-5,6,0,Math.PI*2); ctx.fill();
        ctx.strokeStyle='#9cb1ba'; ctx.beginPath(); ctx.moveTo(-12,8); ctx.lineTo(-23,18); ctx.moveTo(12,8); ctx.lineTo(23,18); ctx.stroke();
        if(thrust){ctx.fillStyle='#ffcc66';ctx.beginPath();ctx.moveTo(-7,14);ctx.lineTo(0,34+Math.random()*7);ctx.lineTo(7,14);ctx.closePath();ctx.fill();}
        // Left side engine
        if (leftThrust) {
            ctx.fillStyle = '#ffcc66';
            ctx.beginPath();
            ctx.moveTo(-15, -3);
            ctx.lineTo(-27 - Math.random() * 4, -1);
            ctx.lineTo(-15, 3);
            ctx.closePath();
            ctx.fill();
        }

        // Right side engine
        if (rightThrust) {
            ctx.fillStyle = '#ffcc66';
            ctx.beginPath();
            ctx.moveTo(15, -3);
            ctx.lineTo(27 + Math.random() * 4, -1);
            ctx.lineTo(15, 3);
            ctx.closePath();
            ctx.fill();
        }
        ctx.restore();
    }

    function renderHuman() {
        drawStars(); drawPad();
        drawLander(human.x,human.y,human.angle,keys.has('ArrowUp') && !human.done,keys.has('ArrowLeft') && !human.done,keys.has('ArrowRight') && !human.done);
        if (!human.done) { ctx.fillStyle='#4e6572'; ctx.font='11px system-ui'; ctx.fillText('YOU',human.x-10,human.y-30); }
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
        const contacts = getLegContact(s);
        updateTelemetry({x: x, y: y, vx, vy, angle, fuel: null, contacts, reward: Number(frame.reward || 0)});
        updateQValues(frame.q_values, frame.action, 'DQN');
        if (frame.done) {
            episodeState.textContent = frame.success ? 'LANDED' : 'ENDED';
            message.classList.remove('hide');
            message.innerHTML = frame.success ? '<strong>Agent landed successfully. 🚀</strong><span>Evaluation complete.</span>' : '<strong>Episode ended.</strong><span>Evaluation complete.</span>';
        }
        else message.classList.add('hide');
        renderAI(frame);
    }

    function renderAI(frame) {
        drawStars(); drawPad();
        const s = frame.state || [];
        const { x, y } = getAIVisualPosition(frame);

        const drawX = Math.max(30, Math.min(W - 30, x));
        const drawY = Math.max(35, Math.min(H - 75, y));
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
            Math.max(20, Math.min(W - 50, drawX - 15)),
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

        const rewards = trainingResults.episode_history.rewards.map(Number);
        const episodes = trainingResults.episode_history.episodes?.map(Number) ?? rewards.map((_, index) => index + 1);

        const width = 1000;
        const height = 430;

        const margin = {top: 30, right: 24, bottom: 55, left: 65};

        const chartWidth = width - margin.left - margin.right;
        const chartHeight = height - margin.top - margin.bottom;

        const minReward = Math.min(...rewards);
        const maxReward = Math.max(...rewards);

        const padding = Math.max(
            20,
            (maxReward - minReward) * 0.08
        );

        const yMin = minReward - padding;
        const yMax = maxReward + padding;

        const x = episode => {
            return margin.left +
                ((episode - 1) / Math.max(1, episodes.length - 1)) *
                chartWidth;
        };

        const y = reward => {
            return margin.top +
                (1 - (reward - yMin) / (yMax - yMin)) *
                chartHeight;
        };

        const rewardPoints = rewards
            .map((reward, index) =>
                `${x(episodes[index])},${y(reward)}`
            )
            .join(' ');

        const windowSize = Math.min(
            trainingResults.episode_history.average_window || 100,
            rewards.length
        );

        const movingAverage = [];

        for (let i = windowSize - 1; i < rewards.length; i++) {
            let sum = 0;

            for (
                let j = i - windowSize + 1;
                j <= i;
                j++
            ) {
                sum += rewards[j];
            }

            movingAverage.push({
                episode: episodes[i],
                reward: sum / windowSize
            });
        }

        const averagePoints = movingAverage
            .map(point =>
                `${x(point.episode)},${y(point.reward)}`
            )
            .join(' ');

        const zeroY =
            yMin <= 0 && yMax >= 0
                ? y(0)
                : null;

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
                </g>
            </svg>
        `;
    }

    async function runAI() {
        clearInterval(aiTimer);
        aiTimer = null;

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

            aiReplay = result;
            aiIndex = 0;

            playAIResult();

        } catch (error) {
            console.error('AI evaluation failed:', error);

            episodeState.textContent = 'ERROR';
            setStatus('AI evaluation failed', 'bad');

            message.classList.remove('hide');
            message.innerHTML =
                '<strong>AI evaluation failed.</strong>' +
                `<span>${error.message}</span>`;

        } finally {
            startAIBtn.disabled = false;
            restartBtn.disabled = false;
        }
    }

    async function loadTrainingResults() {
        try {
            const response = await fetch(
                `../results/training_result_${TRAINING_VERSION}.json`,
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
        document.querySelectorAll('.mode').forEach(b=>{const active=b===btn;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active));});
        modeLabel.textContent=mode==='human'?'HUMAN CONTROL':'DQN EVALUATION';
        startAIBtn.classList.toggle('hidden',mode!=='ai');
        restartBtn.textContent=mode==='human'?'↻ Restart':'↻ Reset replay';
        cancelAnimationFrame(raf); clearInterval(aiTimer); aiTimer=null; last=performance.now();
        if (mode === 'human') {
            setStatus('Human mode ready', 'ok');
            resetHuman();
        } else {
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

    resetHuman(); loadTrainingResults(); checkAPI();
    raf=requestAnimationFrame(loop);
})();
