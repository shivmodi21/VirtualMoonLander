import argparse
import time
from collections import deque, namedtuple
from pathlib import Path
import json
import gymnasium as gym
import numpy as np
import tensorflow as tf

from dqn import build_q_network, compute_loss, create_optimizer
from utils import (
    E_MIN,
    E_DECAY,
    MINIBATCH_SIZE,
    TAU,
    get_action,
    get_experiences,
    get_new_eps,
    check_update_conditions,
    update_target_network,
)

TRAINING_VERSION = "v1"

MEMORY_SIZE = 100_000
GAMMA = 0.995
ALPHA = 1e-3
NUM_STEPS_FOR_UPDATE = 4
DEFAULT_EPISODES = 2000
DEFAULT_SEED = 0
MAX_TIMESTEPS = 1000
AVERAGE_WINDOW = 100
SOLVED_REWARD = 200.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = (PROJECT_ROOT / "models" / f"lunar_lander_{TRAINING_VERSION}.keras")
DEFAULT_RESULTS_PATH = (PROJECT_ROOT / "results" / f"training_result_{TRAINING_VERSION}.json")

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


def save_training_results(
    output_path,
    training_version,
    environment,
    episodes_configured,
    rewards,
    seed,
    final_epsilon,
    final_average_reward,
    training_time,
    model_path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    episodes = list(range(1, len(rewards) + 1))
    environment_solved = (final_average_reward >= SOLVED_REWARD)

    results = {
        "training_version": training_version,
        "environment": environment,

        "model": {
            "path": str(Path(model_path).relative_to(PROJECT_ROOT))
        },

        "training_configuration": {
            "episodes_configured": episodes_configured,
            "seed": seed,
            "replay_buffer_size": MEMORY_SIZE,
            "batch_size": MINIBATCH_SIZE,
            "gamma": GAMMA,
            "learning_rate": ALPHA,
            "steps_per_update": NUM_STEPS_FOR_UPDATE,
            "initial_epsilon": 1.0,
            "epsilon_decay": E_DECAY,
            "minimum_epsilon": E_MIN,
            "target_network_tau": TAU,
            "maximum_timesteps": MAX_TIMESTEPS,
            "solved_reward": SOLVED_REWARD,
            "average_reward_window": AVERAGE_WINDOW,
        },

        "training_results": {
            "episodes_completed": len(rewards),
            "environment_solved": environment_solved,
            "final_epsilon": float(final_epsilon),
            "final_average_reward": float(final_average_reward),
            "training_time_seconds": float(training_time),
            "training_time_minutes": float(training_time / 60),
        },

        "episode_history": {
            "episodes": episodes,
            "rewards": [float(reward) for reward in rewards],
            "average_window": AVERAGE_WINDOW,
        },
    }

    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8",)

def train(
    num_episodes=DEFAULT_EPISODES, 
    seed=DEFAULT_SEED,
    model_path=DEFAULT_MODEL_PATH,
    results_path=DEFAULT_RESULTS_PATH,
):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    env = gym.make("LunarLander-v3")
    state_size = int(env.observation_space.shape[0])
    num_actions = int(env.action_space.n)

    q_network = build_q_network(state_size, num_actions)
    target_q_network = build_q_network(state_size, num_actions)
    target_q_network.set_weights(q_network.get_weights())
    optimizer = create_optimizer(ALPHA)

    @tf.function
    def agent_learn(experiences):
        with tf.GradientTape() as tape:
            loss = compute_loss(experiences, GAMMA, q_network, target_q_network)
        gradients = tape.gradient(loss, q_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
        update_target_network(q_network, target_q_network)
        return loss

    memory_buffer = deque(maxlen=MEMORY_SIZE)
    point_history = []
    epsilon = 1.0
    start = time.time()

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        total_points = 0.0

        for t in range(MAX_TIMESTEPS):
            q_values = q_network(np.expand_dims(state, axis=0), training=False)
            action = get_action(q_values, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory_buffer.append(Experience(state, action, reward, next_state, done))

            if check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer):
                experiences = get_experiences(memory_buffer)
                agent_learn(experiences)

            state = next_state
            total_points += reward

            if done:
                break

        point_history.append(total_points)
        epsilon = get_new_eps(epsilon)
        average = float(np.mean(point_history[-AVERAGE_WINDOW:]))

        print(
            f"\rEpisode {episode + 1}/{num_episodes} | "
            f"Average reward ({min(AVERAGE_WINDOW, len(point_history))}): {average:7.2f} | "
            f"epsilon: {epsilon:.3f}",
            end="",
        )

        if len(point_history) >= AVERAGE_WINDOW and average >= SOLVED_REWARD:
            print(f"\n\nEnvironment solved in {episode + 1} episodes.")
            break
    else:
        print()

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    q_network.save(model_path)
    env.close()

    elapsed = time.time() - start
    final_average = float(np.mean(point_history[-AVERAGE_WINDOW:]))

    save_training_results(
        output_path=results_path,
        training_version=TRAINING_VERSION,
        environment="LunarLander-v3",
        episodes_configured=num_episodes,
        rewards=point_history,
        seed=seed,
        final_epsilon=epsilon,
        final_average_reward=final_average,
        training_time=elapsed,
        model_path=model_path,
    )

    print(f"Training time: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
    print(f"Model saved to: {model_path}")
    print(f"Training results saved to: {results_path}")
    print(f"Final {AVERAGE_WINDOW}-episode average: {final_average:.2f}")

    return q_network, point_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Lunar Lander DQN agent.")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results", default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args()
    train(args.episodes, args.seed, args.model, args.results)
