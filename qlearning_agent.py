import random
import time

class QLearningAgent():
    def __init__(self, game, default_value=0.5, alpha=0.5, gamma=0.98, epsilon=0.9, epsilon_decay=0.99, watch_train=False, watch_run=True, wait=0.1):
        self.game = game
        self.q_table = dict()
        self.default_value = default_value
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.watch_train = watch_train
        self.watch_run = watch_run
        self.wait = wait
        self.game.reset()

    def get_state(self):
        return game.get_state()

    def set_value(self, state_act, value):
        if state_act not in self.q_table:
            self.q_table[state_act] = self.default_value
        else:
            self.q_table[state_act] = value

    def get_value(self, state_act):
        if state_act not in self.q_table:
            self.q_table[state_act] = self.default_value

        return self.q_table[state_act]
    
    def get_max_act(self, state):
        mq = float("-inf")
        ma = None
        actions = self.game.get_applicable_actions_from(state)
        random.shuffle(actions)
        for act in actions:
            q = self.get_value((state, act))
            if q > mq:
                mq = q
                ma = act
        return ma
    
    # def get_max_act(self, state):
    #     mq = float("-inf")
    #     ma = None
    #     actions = self.game.get_applicable_actions_from(state)
    #     random.shuffle(actions)
    #     for act in actions:
    #         q = self.get_value[(state, act)]
    #         if q > mq:
    #             mq = q
    #             ma = act
    #     return ma
    
    def get_epsilon_max(self, state, epsilon):
        prb = random.random()
        if prb < epsilon:
            return random.choice(self.game.get_applicable_actions_from(state))
        else:
            return self.get_max_act(state)
    
    def q_update(self, state_act, reward, next_state_act):
        curr_q = self.get_value(state_act)
        nxt_q = self.get_value(next_state_act)
        q = curr_q + self.alpha*(reward+self.gamma*nxt_q - curr_q)
        self.set_value(state_act, q)

    def learn(self, iterations):
        e = self.epsilon

        for _ in range(iterations):
            
            state = self.game.get_state()
            act = self.get_epsilon_max(state, e)

            reward = self.game.step(act)

            nxt_state = self.game.get_state()
            nxt_act = self.get_max_act(nxt_state)

            state_act = (state, act)
            next_state_act = (nxt_state, nxt_act)

            self.q_update(state_act, reward, next_state_act)

            e = e*self.epsilon_decay

            if self.watch_train:
                time.sleep(self.wait)

    def execute(self):
        self.game.reset()
        done = False
        total_reward = 0
        while not done:
            score = self.game.score
            state = self.game.get_state()
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
