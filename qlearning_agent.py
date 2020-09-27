import random
import time
import q_agent
from tqdm import tqdm

class QLearningAgent(q_agent.QAgent):
    def __init__(self, game, default_value=0.5, alpha=0.5, alpha_decay=0.98, gamma=0.98, epsilon=0.9, epsilon_decay=0.99, decay_start=0, watch_train=False, watch_run=True, wait=0.1):
        q_agent.QAgent.__init__(self, game, default_value, alpha, alpha_decay, gamma, epsilon, epsilon_decay, decay_start, watch_train, watch_run, wait)

    def learn(self, iterations):
        e = self.epsilon
        a = self.alpha

        if self.decay_start > iterations:
            raise ValueError("decay_start value is set for after training finishes")

        for i in tqdm(range(iterations)):
            
            state = self.game.get_state()
            act = self.get_epsilon_max(state, e)

            reward = self.game.step(act)
            if reward < self.game.normal:
                reward = reward*self.game.manhatten_to_apple()*0.1

            nxt_state = self.game.get_state()
            nxt_act = self.get_max_act(nxt_state)

            state_act = (state, act)
            next_state_act = (nxt_state, nxt_act)

            self.q_update(state_act, reward, next_state_act, a)

            if i > self.decay_start:
                e = e*self.epsilon_decay
                a = a*self.alpha_decay

            if self.watch_train:
                time.sleep(self.wait)

    def execute(self, epsilon=None):
        self.game.reset()
        done = False
        total_reward = 0
        while not done:
            score = self.game.score
            state = self.game.get_state()
            if epsilon:
                act = self.get_epsilon_max(state, epsilon)
            else:
                act = self.get_max_act(state)
            reward = self.game.step(act)
            total_reward += reward
            if reward == self.game.loss:
                done = True
                snk, apl = state
                self.game.set_state(snk, apl)
                self.game.draw()
                print(f"Total reward received {total_reward}")
                print(f"Total score {score}")
            if self.watch_run:
                time.sleep(self.wait)
