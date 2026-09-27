# VirtualMoonLander — Deep Q-Learning Lunar Lander

An end-to-end Deep Q-Learning project that trains a neural-network agent to control the Gymnasium Lunar Lander environment and exposes the trained agent through an interactive web application.

The project combines:

- **Deep Reinforcement Learning** with TensorFlow/Keras
- **Gymnasium** for the Lunar Lander environment
- **FastAPI** for live model evaluation
- **HTML/CSS/JavaScript** for the interactive frontend
- **Live Q-value visualization** so users can inspect the agent's decisions
- **Training analytics** rendered directly from the saved training-results JSON

> **Portfolio demo:** train the DQN, start the FastAPI backend, open the frontend, and watch the real trained model evaluate a Lunar Lander episode in real time.

---

## Demo

The project has two interactive modes.

### 🎮 Human mode

Human mode is a lightweight browser-based Lunar Lander simulation.

Controls:

| Key | Action       |
| --- | ------------ |
| `↑` | Main engine  |
| `←` | Rotate left  |
| `→` | Rotate right |

Human mode is implemented entirely in JavaScript using the HTML canvas. It is a visualization/game layer and **is not the Gymnasium physics environment used for training the DQN**.

### 🤖 AI mode

AI mode uses the **actual trained TensorFlow model** through the FastAPI backend.

When the user starts an AI evaluation:

```text
Browser
   │
   │ POST /api/evaluate
   ▼
FastAPI Backend
   │
   ├── Load versioned Keras model
   ├── Create LunarLander-v3
   ├── Evaluate state → Q-values
   ├── Select argmax(Q)
   ├── Step environment
   └── Collect episode frames
   │
   ▼
JSON response
   │
   ▼
Browser animation + live Q-values
```

The browser therefore does **not** invent a precomputed trajectory or calculate fake Q-values. The Q-values and actions shown in AI mode come from the trained model evaluated by the Python backend.

---

# Architecture

The project uses a three-layer architecture:

```text
┌──────────────────────────────────────────────┐
│                  Frontend                    │
│                                              │
│ HTML + CSS + JavaScript + Canvas             │
│                                              │
│ Human Simulation                             │
│ AI Visualization                             │
│ Telemetry                                    │
│ Q-value Visualization                        │
│ Training Results                             │
└──────────────────────┬───────────────────────┘
                       │
                       │ HTTP / JSON
                       ▼
┌──────────────────────────────────────────────┐
│                  Backend                     │
│                                              │
│ FastAPI                                      │
│                                              │
│ GET  /api/health                             │
│ POST /api/evaluate                           │
│                                              │
│ Loads versioned TensorFlow model             │
│ Runs Gymnasium LunarLander-v3                │
│ Generates live evaluation frames             │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│              ML / RL Layer                   │
│                                              │
│ TensorFlow / Keras                           │
│ DQN                                          │
│ Experience Replay                            │
│ ε-greedy Exploration                         │
│ Target Network                               │
│ Bellman Loss                                 │
│ Gymnasium LunarLander-v3                     │
└──────────────────────────────────────────────┘
```

This architecture separates **model training**, **model inference**, and **frontend presentation**.

Training is performed offline using `src/train.py`. The resulting versioned `.keras` model is consumed by the FastAPI inference layer. The browser communicates with the inference layer through HTTP rather than running TensorFlow itself.

---

# What the project demonstrates

## Reinforcement learning

- Deep Q-Learning (DQN)
- Experience replay
- ε-greedy exploration
- Target Q-network
- Soft target-network updates
- Bellman loss
- Discounted future rewards
- Custom TensorFlow training loop with `GradientTape`
- Greedy policy evaluation

## ML engineering

- Modular Python implementation
- Configurable command-line training
- Versioned saved TensorFlow/Keras model
- Unified structured training-results JSON
- Reproducible random seeds
- Separate training and inference workflows
- REST API for model evaluation

## Backend engineering

- FastAPI application
- Pydantic request model
- Health-check endpoint
- Evaluation endpoint
- HTTP error handling
- CORS configuration
- JSON serialization of model evaluation results

## Frontend / visualization

- HTML, CSS, and JavaScript
- Interactive canvas simulation
- Human/AI mode switching
- Live backend-driven AI evaluation
- Live Q-value visualization
- Runtime telemetry
- Training reward chart generated from JSON data
- API availability detection
- Responsive portfolio UI

---

# How the agent learns

## Environment

The agent interacts with:

```text
LunarLander-v3
```

from Gymnasium.

Each observation contains 8 state values:

```text
[x position,
 y position,
 x velocity,
 y velocity,
 angle,
 angular velocity,
 left leg contact,
 right leg contact]
```

The agent chooses among four discrete actions:

| Action  |    Description    |
| ------- | ----------------- |
|   `0`   | Fire left engine  |
|   `1`   | Fire right engine |
|   `2`   | Fire main engine  |
|   `3`   | Do nothing        |


The training loop follows this cycle:

```text
Current state
     │
     ▼
 Q-network
     │
     ▼
ε-greedy action
     │
     ▼
Gymnasium environment
     │
     ├── reward
     └── next state
            │
            ▼
       Replay buffer
            │
            ▼
        Mini-batch
            │
            ▼
      Gradient update
            │
            ▼
       Target network
        soft update
```

A stored experience has the form:

```text
(state, action, reward, next_state, done)
```

---

# DQN architecture

The Q-network maps the 8-dimensional state to four predicted action values.

```text
8 state values
      │
      ▼
Dense(64, ReLU)
      │
      ▼
Dense(64, ReLU)
      │
      ▼
Dense(4, Linear)
      │
      ▼
Q-values for 4 actions
```

The implementation constructs the network dynamically from the environment's observation and action spaces.

The current network consists of:

```python
Dense(64, activation="relu")
Dense(64, activation="relu")
Dense(num_actions, activation="linear")
```

A second network with the same architecture is used as the target network.

---

# Bellman loss

For each transition, the target value is calculated from the target network:

```text
y = reward + (1 - done) × gamma × max Q_target(next_state, action)
```

The implementation then compares the target with the Q-network value corresponding to the action actually taken.

The loss is mean-squared error:

```text
MSE(y_target, Q(state, action))
```

The project uses:

```text
gamma = 0.995
```

---

# Experience replay

The training process stores experiences in a replay buffer with a maximum capacity of:

```text
100,000
```

A random mini-batch of:

```text
64 experiences
```

is sampled whenever an update is due.

Updates occur every:

```text
4 environment steps
```

The replay mechanism helps reduce correlation between consecutive experiences and makes DQN training more stable.

---

# ε-greedy exploration

During training, the agent uses ε-greedy exploration.

At high ε values, random actions are selected more frequently. As training progresses, ε decays toward a minimum value, allowing the agent to increasingly exploit the learned policy.

Current configuration:

| Parameter | Value   |
| --------- | ------- |
| Initial ε | `1.0`   |
| Decay     | `0.995` |
| Minimum ε | `0.01`  |

The decay is implemented as:

```python
epsilon = max(E_MIN, E_DECAY * epsilon)
```

---

# Target network

DQN maintains two networks:

- **Q-network** — updated using gradient descent.
- **Target Q-network** — used to compute the Bellman target.

The target network starts with the same weights as the Q-network and is then updated using a soft update:

```text
target = tau × q_network + (1 - tau) × target
```

The project uses:

```text
tau = 1e-3
```

This allows the target values to change gradually during training.

---

# Training configuration

The current training implementation uses:

| Parameter | Value |
| --- | --- |
| Environment | `LunarLander-v3` |
| Replay buffer | `100,000` |
| Mini-batch size | `64` |
| Discount factor (`gamma`) | `0.995` |
| Learning rate | `1e-3` |
| Steps between updates | `4` |
| Initial epsilon | `1.0` |
| Epsilon decay | `0.995` |
| Minimum epsilon | `0.01` |
| Target update (`tau`) | `1e-3` |
| Maximum timesteps / episode | `1000` |
| Moving-average window | `100` |
| Solved threshold | `200` average reward |
| Default maximum episodes | `2000` |
| Default seed | `0` |

Training stops early when the average reward over the most recent 100 episodes reaches `200`.

---

# Project structure

```text
VirtualMoonLander/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── backend/
│   └── main.py
│
├── src/
│   ├── dqn.py
│   ├── train.py
│   ├── utils.py
│   └── evaluate.py
│
├── models/
│   └── lunar_lander_v1.keras
│
└── results/
    └── training_result_v1.json
```

The training version is intentionally hard-coded in the training script, backend, and frontend so that the model and training-results file always refer to the same trained version.

For example:

```text
Training version: v1

Model:
models/lunar_lander_v1.keras

Training results:
results/training_result_v1.json
```

---

# File responsibilities

## `src/train.py`

Responsible for:

- creating the Gymnasium environment
- initializing the Q-network and target network
- running the DQN training loop
- managing the replay buffer
- selecting actions using ε-greedy exploration
- performing gradient updates
- soft-updating the target network
- saving the versioned trained model
- saving the unified training-results JSON

The training version is defined inside the script.

Training output:

```text
models/lunar_lander_v1.keras
results/training_result_v1.json
```

There is no separate training summary file or training log file in the new output format.

## `src/dqn.py`

Contains the reusable DQN components:

- Q-network architecture
- Bellman loss
- Adam optimizer creation

## `src/utils.py`

Contains reusable training utilities:

- replay-buffer sampling
- update scheduling
- ε decay
- ε-greedy action selection
- target-network soft updates

## `src/evaluate.py`

Runs the trained model directly inside Gymnasium with graphical rendering.

It loads the versioned Keras model and selects:

```text
argmax(Q(s, a))
```

for every environment state.

## `backend/main.py`

Provides the live inference API.

Responsibilities include:

- loading the versioned trained model
- creating `LunarLander-v3`
- running a complete greedy evaluation episode
- collecting Q-values and state information
- converting environment coordinates for the frontend
- returning the complete evaluation as JSON

## `frontend/index.html`

Defines the browser interface:

- navigation
- hero section
- interactive lab
- human/AI mode selector
- canvas
- telemetry
- Q-value panel
- learning explanation
- training results
- GitHub link

## `frontend/script.js`

Controls the browser application:

- human-mode physics
- canvas rendering
- telemetry
- AI evaluation requests
- AI replay playback
- Q-value visualization
- API health checking
- versioned training-results loading
- training reward chart rendering
- keyboard controls

The frontend uses the same hard-coded training version as the backend and training script.

## `frontend/style.css`

Contains the visual design system and responsive layout for the portfolio application.

---

# Training results

The training script now saves **one versioned JSON file** containing both the information previously split between the training summary and training log.

For version `v1`:

```text
results/training_result_v1.json
```

The JSON contains:

```json
{
  "training_version": "v1",
  "environment": "LunarLander-v3",
  "model": {
    "path": "models/lunar_lander_v1.keras"
  },
  "training_configuration": {
    "episodes_configured": 2000,
    "seed": 0,
    "replay_buffer_size": 100000,
    "batch_size": 64,
    "gamma": 0.995,
    "learning_rate": 0.001,
    "steps_per_update": 4,
    "initial_epsilon": 1.0,
    "epsilon_decay": 0.995,
    "minimum_epsilon": 0.01,
    "target_network_tau": 0.001,
    "maximum_timesteps": 1000,
    "solved_reward": 200.0,
    "average_reward_window": 100
  },
  "training_results": {
    "episodes_completed": 2000,
    "environment_solved": true,
    "final_epsilon": 0.01,
    "final_average_reward": 205.32,
    "training_time_seconds": 1234.56,
    "training_time_minutes": 20.58
  },
  "episode_history": {
    "episodes": [1, 2, 3],
    "rewards": [12.4, -31.2, 48.6],
    "average_window": 100
  }
}
```

The exact numerical values depend on the training run.

The `episode_history` section is important because the frontend uses the saved episode rewards to build the training chart. The frontend calculates the moving average from the raw rewards and the saved averaging window.

---

# Training visualization

The frontend builds the training reward chart directly from:

```text
results/training_result_v1.json
```

The chart displays:

- episode reward
- moving-average reward
- episode axis
- total reward axis
- chart grid
- chart legend

The moving-average window is read from the saved training-results JSON, so the visualization follows the configuration used by the training run.

This makes the training visualization part of the frontend rather than a separate generated image asset.

---

# Backend API

The current implementation uses FastAPI as the model-serving layer.

## Start the API

From the **project root**:

```bash
uvicorn backend.main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

FastAPI also provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

---

## Health endpoint

```http
GET /api/health
```

Example response:

```json
{
  "status": "ok",
  "service": "VirtualMoonLander"
}
```

The frontend calls this endpoint when it initializes to determine whether the DQN backend is available.

---

## Evaluation endpoint

```http
POST /api/evaluate
```

Request body:

```json
{
  "seed": 0
}
```

The backend then:

1. Loads the versioned Keras model.
2. Creates `LunarLander-v3`.
3. Resets the environment using the requested seed.
4. Evaluates the current state through the Q-network.
5. Selects `argmax(Q-values)`.
6. Steps the environment.
7. Records the resulting state, Q-values, action, reward, and telemetry.
8. Continues until the episode ends.
9. Returns the episode data as JSON.

The response contains the evaluation metadata and the frame data used by the frontend.

---

# Frontend AI workflow

When AI mode is selected, the frontend checks the backend status.

If the API is available:

```text
DQN evaluation ready
```

When **Run AI** is pressed:

```text
1. Disable evaluation controls
2. POST /api/evaluate
3. Wait for backend evaluation
4. Receive episode JSON
5. Store result in aiReplay
6. Play the returned frames
7. Display telemetry
8. Display Q-values
9. Highlight selected action
10. Show landing / episode completion
```

The frontend animates the returned frames at the FPS value supplied by the backend.

If the API is unavailable, the UI reports:

```text
DQN API offline
```

and instructs the user to start the FastAPI server.

---

# Live Q-value visualization

For every AI frame, the interface displays four action values:

```text
← Left
→ Right
🔥 Main
○ No action
```

The selected action is:

```text
argmax(Q(s, a))
```

The corresponding row is highlighted and the UI displays:

```text
DQN selected <action> with Q = <value>
```

This makes the demo inspectable: users can see not only what the agent did, but also the relative action values produced by the Q-network for that state.

---

# Telemetry

The browser displays model/environment information including:

- altitude
- vertical velocity
- horizontal velocity
- angle
- leg contact
- reward
- fuel in human mode

Telemetry states are visually updated based on the current values.

For AI mode, the displayed values are derived from the state returned by the backend evaluation.

For human mode, they come from the JavaScript simulation.

---

# Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd VirtualMoonLander
```

Replace the placeholder with the final GitHub repository URL.

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dependency file should contain the packages required by the current Python implementation, including TensorFlow, Gymnasium, NumPy, FastAPI, Uvicorn, and the other runtime dependencies used by the project.

---

# Train the agent

Run training from the **project root**:

```bash
python src/train.py
```

The default training configuration allows up to:

```text
2000 episodes
```

but training stops early when the 100-episode moving average reaches:

```text
200 reward
```

The trained model is saved to:

```text
models/lunar_lander_v1.keras
```

The unified training results are saved to:

```text
results/training_result_v1.json
```

The training version is defined inside `src/train.py`.

---

## Custom training

Train for a specific number of episodes:

```bash
python src/train.py --episodes 500
```

Specify a seed:

```bash
python src/train.py --episodes 2000 --seed 0
```

The current training interface intentionally keeps the model path, result path, and training version inside `src/train.py`. They are not command-line arguments.

---

# Evaluate the trained agent directly

To open the actual Gymnasium environment and watch the trained agent:

```bash
python src/evaluate.py
```

The script:

1. Loads the versioned trained model.
2. Creates `LunarLander-v3` with human rendering.
3. Computes Q-values for each state.
4. Selects the highest-valued action.
5. Steps the environment until completion.
6. Prints the episode reward.

The model version used by `evaluate.py` is configured in the script.

---

# Run the complete web application locally

The web application requires both:

1. the **FastAPI backend**
2. the **static frontend**

## Terminal 1 — Start FastAPI

From the **project root**:

```bash
uvicorn backend.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Terminal 2 — Start the frontend

From the **project root**, start the static server:

```bash
python -m http.server 5500
```

Then open:

```text
http://localhost:5500/frontend/
```

Serving from the project root is important because the frontend loads the versioned training-results file from:

```text
../results/training_result_v1.json
```

relative to the frontend application.

The frontend uses the configured API URL in `frontend/script.js` to communicate with the FastAPI backend.

---

# Complete development workflow

For a fresh installation:

```bash
# Clone
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd VirtualMoonLander

# Create environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Or on macOS/Linux
# source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Train the DQN
python src/train.py

# Optional: watch the agent directly in Gymnasium
python src/evaluate.py
```

Then start the web application.

### Terminal 1

```bash
uvicorn backend.main:app --reload
```

### Terminal 2

```bash
python -m http.server 5500
```

Open:

```text
http://localhost:5500/frontend/
```

Then:

1. Select **AI**.
2. Confirm the status shows the DQN API is available.
3. Press **Run AI**.
4. Watch the backend evaluate the trained model.
5. Inspect the Q-values and selected action frame by frame.

---

# Quick start with an existing trained model

If the repository already contains:

```text
models/lunar_lander_v1.keras
```

you do not need to retrain the agent.

Make sure the matching training-results file exists:

```text
results/training_result_v1.json
```

Then start the API:

```bash
uvicorn backend.main:app --reload
```

Start the frontend from the project root:

```bash
python -m http.server 5500
```

Open:

```text
http://localhost:5500/frontend/
```

Select **AI** and run the evaluation.

The backend will load the existing versioned model and perform a fresh Gymnasium evaluation.

---

# Deployment

## Frontend

The frontend consists of static:

```text
HTML
CSS
JavaScript
```

It can be hosted by a static hosting provider.

However, **AI mode requires the FastAPI backend to be running and reachable by the browser** because the current architecture performs live model inference on the backend.

Therefore, the complete AI demo is not a frontend-only deployment.

The deployment architecture is:

```text
Static Frontend
     │
     │ HTTPS / JSON
     ▼
Hosted FastAPI API
     │
     ▼
TensorFlow / Keras Model
     │
     ▼
Gymnasium LunarLander-v3
```

Possible deployment approaches include:

- static frontend + separately hosted FastAPI service
- a platform capable of serving both frontend and Python backend
- containerized deployment

Before production deployment, update the frontend API URL from the local development address to the deployed API endpoint.

---

# CORS

The FastAPI application currently enables CORS for local development.

For production deployment, the allowed origins should be restricted to the actual frontend domain rather than allowing every origin.

---

# Model inference architecture

The trained model is stored as:

```text
models/lunar_lander_v1.keras
```

The backend loads this versioned model when an evaluation request is received.

For every state:

```python
q_values = model(tensor, training=False,).numpy()[0]
action = int(np.argmax(q_values))
```

The action is therefore selected using the maximum predicted Q-value.

The backend then sends the state and decision information back to the frontend.

---

# Design principles

The current system is intentionally separated into independent responsibilities.

## Training is offline

`src/train.py` is responsible for learning the policy.

It does not depend on the web frontend.

## Evaluation is a backend responsibility

`backend/main.py` is responsible for serving the trained model and running live evaluations.

The frontend does not load TensorFlow.

## Visualization is a frontend responsibility

`frontend/script.js` takes the backend response and turns it into:

- animation
- telemetry
- Q-value bars
- action selection feedback
- episode status
- training reward visualization

## Human mode is independent

The browser simulation provides an interactive experience without requiring the ML backend.

This means the frontend remains usable even when the DQN API is unavailable, although AI mode requires the backend.

---

# Error handling

The FastAPI layer distinguishes common evaluation failures.

### Model missing

If:

```text
models/lunar_lander_v1.keras
```

does not exist, the API reports that the required versioned model is unavailable.

### Evaluation failure

Unexpected errors during model/environment evaluation return an HTTP `500` response.

The frontend catches API failures and displays an appropriate error state instead of silently failing.

---

# Reproducibility

The training script accepts a seed:

```bash
python src/train.py --seed 0
```

The seed is applied to the relevant training components and Gymnasium episode resets.

The API also accepts a seed in the evaluation request:

```json
{
  "seed": 0
}
```

This allows evaluation episodes to be reproduced using the same environment seed.

Exact numerical results may still vary across machines, library versions, hardware, and underlying numerical operations.

---

# Portfolio value

This project demonstrates more than implementing a reinforcement-learning algorithm.

It connects the complete workflow:

```text
RL Research / Algorithm
          ↓
Model Training
          ↓
Versioned Saved Model
          ↓
Backend Inference API
          ↓
HTTP / JSON Interface
          ↓
Interactive Frontend
          ↓
Visual Explanation of Model Decisions
```

This makes the project representative of an ML application rather than only a training notebook.

A recruiter can inspect:

- the DQN implementation
- the training loop
- the neural-network architecture
- the inference API
- the frontend
- the system architecture
- the model's actual Q-values
- the training metrics

---

# Key implementation files

| File          | Responsibility                                        |
| ------------- | ----------------------------------------------------- |
| `train.py`    | DQN training pipeline                                 |
| `dqn.py`      | Network, Bellman loss, optimizer                      |
| `utils.py`    | Replay, exploration, target updates                   |
| `evaluate.py` | Direct Gymnasium visual evaluation                    |
| `api.py`      | FastAPI model-serving and live evaluation             |
| `index.html`  | Frontend structure                                    |
| `style.css`   | Frontend styling and responsive layout                |
| `script.js`   | Browser simulation, API integration, AI visualization |

---

# References

- Mnih et al., *Human-level control through deep reinforcement learning*, Nature, 2015.
- Mnih et al., *Playing Atari with Deep Reinforcement Learning*, 2013.
- Lillicrap et al., *Continuous Control with Deep Reinforcement Learning*, ICLR, 2016.

---



# Project background

The original implementation was developed as part of a DeepLearning.AI reinforcement-learning course assignment.

The project was subsequently reorganized into standalone Python modules and extended with a web-based application so that the trained reinforcement-learning model can be demonstrated interactively.

The current architecture separates:

```text
Training
Inference
API
Frontend
```

making the project easier to run, inspect, and present.

---



# Author

**Shiv Modi**

B.Tech. + M.Tech. — IIT Bombay

- GitHub: [https://github.com/shivmodi21](https://github.com/shivmodi21)
- Portfolio: [https://shivmodi21.github.io/](https://shivmodi21.github.io/)
- LinkedIn: [https://www.linkedin.com/in/shivmodi210/](https://www.linkedin.com/in/shivmodi210/)
