from dqn import *

def train_dqn_agent(env, agent, num_episodes, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, update_target_freq=10):
    epsilon = epsilon_start
    all_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            action = agent.act(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            agent.train()

            if done or truncated:
                break

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        all_rewards.append(episode_reward)

        if episode % update_target_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")

    return all_rewards