from pathlib import Path

import gymnasium as gym
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_VERSION = "v2"
DEFAULT_MODEL_PATH = (BASE_DIR / "models" / f"lunar_lander_{TRAINING_VERSION}.keras")
MODEL = None

app = FastAPI(
    title="VirtualMoonLander API",
    description="API for live DQN Lunar Lander evaluation.",
    version="1.0.0",
)

@app.on_event("startup")
def load_model():
    global MODEL

    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{DEFAULT_MODEL_PATH}'."
        )

    MODEL = tf.keras.models.load_model(DEFAULT_MODEL_PATH)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluationRequest(BaseModel):
    seed: int = 0

def to_canvas(state: np.ndarray) -> tuple[float, float]:
    x = 450.0 + float(state[0]) * 360.0
    y = 470.0 - float(state[1]) * 300.0
    return x, y

def is_successful_landing(state: np.ndarray, terminated: bool) -> bool:
    if not terminated:
        return False

    left_leg_contact = bool(state[6])
    right_leg_contact = bool(state[7])

    return left_leg_contact and right_leg_contact


def evaluate_episode(model, seed: int = 0):
    env = gym.make("LunarLander-v3")

    state, _ = env.reset(seed=seed)

    frames = []
    total_reward = 0.0

    terminated = False
    truncated = False

    while not (terminated or truncated):
        state_array = np.asarray(state, dtype=np.float32)

        tensor = tf.convert_to_tensor(
            state_array[None, :],
            dtype=tf.float32,
        )

        q_values = model(
            tensor,
            training=False,
        ).numpy()[0]

        action = int(np.argmax(q_values))

        next_state, reward, terminated, truncated, _ = env.step(action)

        total_reward += float(reward)

        x, y = to_canvas(state_array)

        next_state_array = np.asarray(next_state, dtype=np.float32)

        frames.append(
            {   
                # State used by the DQN to choose this action.
                "state": state_array.tolist(),

                # State produced after taking the action.
                "next_state": next_state_array.tolist(),

                "q_values": q_values.astype(float).tolist(),
                "action": action,
                "reward": float(reward),

                "x": x,
                "y": y,

                "altitude": float(state_array[1]),
                "vx": float(state_array[2]),
                "vy": float(state_array[3]),
                "angle": float(state_array[4]),
                "left_leg_contact": bool(state_array[6]),
                "right_leg_contact": bool(state_array[7]),

                # This action ended the environment episode.
                "done": bool(terminated or truncated),
                "success": False,
            }
        )

        state = next_state

    if frames:
        terminal_state = np.asarray(state, dtype=np.float32)
        frames[-1]["terminal_state"] = terminal_state.tolist()

    terminal_state = np.asarray(state, dtype=np.float32)
    if frames:
        frames[-1]["terminal_state"] = terminal_state.tolist()
    
    success = is_successful_landing(terminal_state, terminated,)

    if frames:
        frames[-1]["success"] = success

    env.close()

    return {
        "environment": "LunarLander-v3",
        "seed": seed,
        "episode_reward": total_reward,
        "success": success,
        "frames": frames,
        "fps": 18.0,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "VirtualMoonLander",
    }



@app.get("/api/training/{training_version}")
def get_training_result(training_version: str):
    result_path = BASE_DIR / "results" / f"training_result_{training_version}.json"

    if not result_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Training result '{training_version}' not found.",
        )

    return FileResponse(result_path)


@app.post("/api/evaluate")
def evaluate(request: EvaluationRequest):
    try:
        return evaluate_episode(model=MODEL, seed=request.seed,)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {exc}",
        ) from exc

FRONTEND_DIR = BASE_DIR / "frontend"

app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend",
)

@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")