import random
import numpy as np
import tensorflow as tf

MINIBATCH_SIZE = 64
TAU = 1e-3
E_DECAY = 0.995
E_MIN = 0.01


def get_experiences(memory_buffer):
    """Sample a random mini-batch from the replay buffer."""
    experiences = random.sample(memory_buffer, k=MINIBATCH_SIZE)
    states = tf.convert_to_tensor(np.array([e.state for e in experiences]), dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([e.action for e in experiences]), dtype=tf.float32)
    rewards = tf.convert_to_tensor(np.array([e.reward for e in experiences]), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([e.next_state for e in experiences]), dtype=tf.float32)
    done_vals = tf.convert_to_tensor(np.array([e.done for e in experiences], dtype=np.float32), dtype=tf.float32)
    return states, actions, rewards, next_states, done_vals


def check_update_conditions(t, num_steps_upd, memory_buffer):
    """Return True when it is time and there is enough data for an update."""
    return (t + 1) % num_steps_upd == 0 and len(memory_buffer) >= MINIBATCH_SIZE


def get_new_eps(epsilon):
    """Decay epsilon while keeping a small amount of exploration."""
    return max(E_MIN, E_DECAY * epsilon)


def get_action(q_values, epsilon=0.0, rng=None):
    """Choose an action using an epsilon-greedy policy."""
    rng = rng or random
    if rng.random() > epsilon:
        return int(np.argmax(q_values.numpy()[0]))
    return rng.randrange(int(q_values.shape[-1]))


def update_target_network(q_network, target_q_network):
    """Soft-update target-network weights."""
    for target_weights, q_net_weights in zip(target_q_network.weights, q_network.weights):
        target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)
