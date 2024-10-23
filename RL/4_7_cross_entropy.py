# 4_7_cross_entropy.py
import gym
import numpy as np


def run_episode(env, w, b):
    state, _ = env.reset()

    step_count, done = 0, False
    while not done:
        # (2) = (1, 4) @ (4, 2)
        actions = np.dot(state, w) + b  # dot 행렬곱셈
        best = np.argmax(actions)

        state, reward, done, _, _ = env.step(best)
        step_count += 1
        if step_count > 500:
            break
    return step_count


def make_theta(theta_mean, theta_sd):
    return np.random.multivariate_normal(mean=theta_mean, cov=np.diag(theta_sd))


# env = gym.make('CartPole-v1')
# i_size, o_size = 4, 2
# n_sample = 32
# w_size = i_size * o_size  # 8개
# b_size = o_size  # 2개
#
# theta_mean = np.zeros(w_size + b_size)
# theta_sd = np.ones(w_size + b_size)
#
# # print(make_theta(theta_mean, theta_sd))
# # [ 0.31243822  0.0228355   0.5248056   1.75803453 -0.57078932 -0.42182691
# #   0.2152496   2.36402668  0.00544605 -0.43856867]
#
# for episode in range(100):
#     population = [make_theta(theta_mean, theta_sd) for _ in range(n_sample)]  # (32, 10)
#
#     rewards = [run_episode(env, w=p[:8].reshape(4, 2), b=p[8:]) for p in population]
#     # print(sorted(rewards, reverse=True))
#
#     indices = np.argsort(rewards)[-int(n_sample*0.2):]
#     # print(indices)
#     # print(rewards[indices[-1]], rewards[indices[-2]])
#
#     elite = [population[idx] for idx in indices]
#
#     theta_mean = np.mean(elite, axis=0)
#     theta_sd = np.std(elite, axis=0)
#
#     elite_rewards = [rewards[idx] for idx in indices]
#     avg_rewards = np.mean(elite_rewards)
#     print(episode, avg_rewards, elite_rewards)
#
# env.close()
# ------------------------------------------------------- #

env = gym.make('CartPole-v1', render_mode='human')
# env = gym.make('CartPole-v1')
# p = elite[-1]
# print(p)
p = [-1.44648131, -0.03654486, -4.49853192, 2.7878456, -3.13549209, 10.0746595,
     -8.72544225, 5.31647421, -1.64691165, -1.1458293]
p = np.array(p)
run_episode(env, w=p[:8].reshape(4, 2), b=p[8:])

env.close()
