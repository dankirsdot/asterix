import os
import retro

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
retro.data.Integrations.add_custom_path(
    os.path.join(SCRIPT_DIR, "custom_integrations")
)

def main():
    env = retro.make("CustomAsterix-Sms", inttype=retro.data.Integrations.ALL)
    env.reset()

    total_reward = 0
    i = 0

    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        print('action', action)
        total_reward += reward
        print('total_reward', total_reward)

        i += 1
        print(i)
        print('observation', type(observation), observation.shape)
        print('reward', reward)
        print('terminated', terminated)
        print('truncated', truncated)
        print('info', info)

        if terminated or truncated:
            print('RESET')
            env.reset()
            input()

    env.close()


if __name__ == "__main__":
    main()
