import game
import qlearning_agent
import time
game = game.Snake(h=5, w=5, live_draw=False)

agent = qlearning_agent.QLearningAgent(game=game,
                                        default_value=0.5, 
                                        alpha=0.4,
                                        alpha_decay=1,
                                        gamma=0.98, 
                                        epsilon=0.99, 
                                        epsilon_decay=0.99,
                                        decay_start=500000,
                                        watch_train=False, 
                                        wait=0.1)

agent.learn(1000)

print("Executing learned parameters...")
game.live_draw = True
time.sleep(2)

while True:
    print("Running")
    agent.execute()
    time.sleep(2)

