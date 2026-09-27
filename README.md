# VirtualMoonLander — Deep Q-Learning Lunar Lander

An interactive Deep Reinforcement Learning project where a **Deep Q-Network (DQN)** learns to control the Gymnasium Lunar Lander environment.

## 🚀 Live Demo

[**Launch VirtualMoonLander →**](https://virtualmoonlander.onrender.com/)

Open the deployed application to interact with the project directly in your browser.

### What you can explore

* 🎮 **Human Mode** — control the lander yourself.
* 🤖 **AI Mode** — watch the trained DQN control the lander.
* 📊 **Live Q-values** — see how the agent evaluates each available action.
* 📡 **Live telemetry** — inspect position, velocity, angle, reward, and leg contact.
* 📈 **Training results** — explore episode rewards and the moving average from the recorded training run.
* 🧠 **DQN architecture and learning process** — understand how the agent learns from experience.

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

## 🏗️ Project Structure

```text
VirtualMoonLander/
├── backend/
│   └── main.py
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── models/
│   └── lunar_lander_v1.keras
│
├── results/
│   └── training_result_v1.json
│
├── src/
│   ├── dqn.py
│   ├── evaluate.py
│   ├── train.py
│   └── utils.py
│
├── requirements.txt
├── .python-version
├── .gitignore
└── README.md
```

### File responsibilities

**`src/train.py`**
Trains the DQN agent and records the training trajectory.

**`src/dqn.py`**
Defines the Q-network, loss calculation, optimizer, and target-network updates.

**`src/utils.py`**
Contains replay-buffer, action-selection, epsilon-decay, and target-network utilities.

**`src/evaluate.py`**
Runs local graphical evaluation of the trained model using Gymnasium.

**`backend/main.py`**
Provides the FastAPI service used by the deployed application for DQN evaluation.

**`frontend/`**
Contains the interactive browser interface and visualization.

**`models/`**
Contains the trained Keras model.

**`results/`**
Contains the recorded training results used by the application.

---

## ⚙️ Technical Highlights

### Experience Replay

The agent stores up to **100,000 transitions** and samples random batches during training. This reduces correlation between consecutive experiences and improves learning stability.

### Epsilon-Greedy Exploration

The agent begins with a high exploration rate and gradually shifts toward exploiting its learned Q-values.

### Target Network

A separate target network is softly updated from the online network, providing a more stable target during Q-learning updates.

### End-to-End Inference

The deployed application does not simulate the trained agent in JavaScript. AI evaluation runs on the Python backend using the trained TensorFlow model and the Gymnasium environment, while the browser visualizes the resulting trajectory.

---

## 🛠️ Technology Stack

**Machine Learning**

* Python
* TensorFlow / Keras
* NumPy
* Gymnasium

**Reinforcement Learning**

* Deep Q-Learning
* Experience Replay
* Epsilon-Greedy Exploration
* Target Network
* Bellman Loss

**Backend**

* FastAPI
* Uvicorn
* Pydantic

**Frontend**

* HTML
* CSS
* JavaScript
* SVG-based training visualization
* Canvas-based lander visualization

**Deployment**

* Render
* GitHub

---

## 🔬 DQN Architecture

The agent receives an 8-dimensional state:

1. X position
2. Y position
3. X velocity
4. Y velocity
5. Angle
6. Angular velocity
7. Left leg contact
8. Right leg contact

The neural network is:

```text
8 input features
       ↓
Dense(64) + ReLU
       ↓
Dense(64) + ReLU
       ↓
Dense(4) + Linear
       ↓
Q-values for 4 actions
```

### Training configuration

| Parameter             |            Value |
| --------------------- | ---------------: |
| Environment           | `LunarLander-v3` |
| Replay buffer         |          100,000 |
| Batch size            |               64 |
| Discount factor γ     |            0.995 |
| Learning rate         |            0.001 |
| Update frequency      |    Every 4 steps |
| Initial ε             |              1.0 |
| ε decay               |            0.995 |
| Minimum ε             |             0.01 |
| Target-network τ      |            0.001 |
| Maximum timesteps     |            1,000 |
| Moving-average window |     100 episodes |
| Solved threshold      |       200 reward |
| Configured episodes   |            2,000 |
| Seed                  |                0 |

Training stops early when the average reward over the most recent 100 episodes reaches 200.

---

## 📈 Training Results

The project records the training trajectory and exposes the results through the application.

The deployed website visualizes:

* Episode reward
* 100-episode moving average
* Number of completed episodes
* Final average reward
* Training time

The trained model is versioned using the training version, currently:

```text
v1
```
---

## 📂 Source Code

The complete implementation is available on GitHub:

**[View the GitHub Repository →](https://github.com/shivmodi21/VirtualMoonLander)**

---

## 🎯 Purpose

This project was built to demonstrate the complete lifecycle of a reinforcement-learning application:

```text
Design
  ↓
Train
  ↓
Evaluate
  ↓
Save model
  ↓
Build inference API
  ↓
Create interactive visualization
  ↓
Deploy
```

Rather than presenting only a trained model or notebook, VirtualMoonLander packages the reinforcement-learning system into an interactive application that allows users to observe the agent's decisions and understand the underlying learning process.

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
