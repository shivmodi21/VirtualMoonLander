import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam


def build_q_network(state_size: int, num_actions: int) -> Sequential:
    """Build the Q-network used by the agent."""
    return Sequential(
        [
            tf.keras.Input(shape=(state_size,)),
            Dense(64, activation="relu", name="L1"),
            Dense(64, activation="relu", name="L2"),
            Dense(num_actions, activation="linear", name="L3"),
        ]
    )


def compute_loss(experiences, gamma: float, q_network, target_q_network):
    """Compute the DQN mean-squared Bellman loss."""
    states, actions, rewards, next_states, done_vals = experiences

    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)
    y_targets = rewards + (1.0 - done_vals) * gamma * max_qsa

    q_values = q_network(states)
    indices = tf.stack(
        [tf.range(tf.shape(q_values)[0]), tf.cast(actions, tf.int32)], axis=1
    )
    q_values = tf.gather_nd(q_values, indices)

    return MSE(y_targets, q_values)


def create_optimizer(learning_rate: float) -> Adam:
    """Create the Adam optimizer used by the training loop."""
    return Adam(learning_rate=learning_rate)
