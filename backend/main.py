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
TRAINING_VERSION = "v1"
DEFAULT_MODEL_PATH = (BASE_DIR / "models" / f"lunar_lander_{TRAINING_VERSION}.keras")

app = FastAPI(
    title="VirtualMoonLander API",
    description="API for live DQN Lunar Lander evaluation.",
    version="1.0.0",
)

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


def evaluate_episode(model_path: Path, seed: int = 0):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'."
        )

    model = tf.keras.models.load_model(model_path)

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

        frames.append(
            {
                "state": state_array.tolist(),
                "q_values": q_values.astype(float).tolist(),
                "action": action,
                "reward": float(reward),

                # Gymnasium state
                "x": x,
                "y": y,
                "vx": float(state_array[2]),
                "vy": float(state_array[3]),
                "angle": float(state_array[4]),
                "left_leg_contact": bool(state_array[6]),
                "right_leg_contact": bool(state_array[7]),

                "done": False,
                "success": False,
            }
        )

        state = next_state

    if frames:
        frames[-1]["done"] = True

    success = total_reward >= 200.0

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


@app.post("/api/evaluate")
def evaluate(request: EvaluationRequest):
    try:
        return evaluate_episode(
            model_path=DEFAULT_MODEL_PATH,
            seed=request.seed,
        )

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