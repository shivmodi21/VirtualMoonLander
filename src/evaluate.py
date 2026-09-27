import argparse
from pathlib import Path

import gymnasium as gym
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_VERSION = "v1"

DEFAULT_MODEL_PATH = (PROJECT_ROOT / "models" / f"lunar_lander_{TRAINING_VERSION}.keras")

def evaluate(model_path=DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found. Please run `python src/train.py` first."
        )

    env = gym.make("LunarLander-v3", render_mode="human")
    model = tf.keras.models.load_model(model_path)

    state, _ = env.reset()
    terminated = truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        q_values = model(tf.expand_dims(state, axis=0), training=False)
        action = int(tf.argmax(q_values[0]).numpy())
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    print(f"Episode reward: {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch the trained Lunar Lander agent.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    evaluate(args.model)