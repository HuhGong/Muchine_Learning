# 1_4_frozen_lake_game.py
import gym
import readchar


# menu -> run ->  edit Configuration -> modify options -> emulate terminal in output console
# 퀴즈
# x를 입력 할 때 까지 반복 입력 받는 코드를 만드세요
# 퀴즈
# asdw키를 사용해서 frozen lake로부터
# 목적지까지 캐릭터를 이동하는 코드를 구현하세요

# print(readchar.readkey())

def game_basic():
    env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=False)

    env.reset()
    env.render()

    LEFT, DOWN, RIGHT, UP = 'a', 's', 'd', 'w'

    done = False

    while not done:
        c = readchar.readkey()

        if c != LEFT and c != DOWN and c != RIGHT and c != UP:
            continue
        action = 3
        if c == LEFT:
            action = 0
        elif c == DOWN:
            action = 1
        elif c == RIGHT:
            action = 2

        _, _, done, _, _ = env.step(action)
        env.render()

    env.close()


def advanced_game():
    env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=True)

    env.reset()
    env.render()

    LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
    # actions = {'a': LEFT, 's': DOWN, 'd': RIGHT, 'w': UP}

    actions = {
        '\x00H': UP,
        '\x00P': DOWN,
        '\x00M': RIGHT,
        '\x00K': LEFT,
    }
    done = False

    while not done:
        c = readchar.readkey()

        if c not in actions:
            continue
        else:
            action = actions[c]

        _, _, done, _, _ = env.step(action)
        env.render()

    env.close()


advanced_game()
