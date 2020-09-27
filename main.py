import game
import qlearning_agent
import time
game = game.Snake(h=5, w=5, live_draw=False)

agent = qlearning_agent.QLearningAgent(game=game, watch_train=False, wait=0.1)

agent.learn(1000000)

print("Executing learned parameters...")
game.live_draw = True
time.sleep(2)
agent.execute()

i=0
while i<1:
    game.draw()
    time.sleep(4)
    i += 1
