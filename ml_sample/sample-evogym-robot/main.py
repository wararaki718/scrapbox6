import gym
from utils import factory_create


def main() -> None:
    body, connections = factory_create(5, 5)
    env = gym.make("Walker-v0", body=body)
    env.reset()

    while True:
        action = env.action_space.sample() - 1
        ob, reward, done, info = env.step(action)
        print(ob, reward, done, info)
        env.render()
        
        if done:
            env.reset()
    env.close()
    print("DONE")


if __name__ == "__main__":
    main()
