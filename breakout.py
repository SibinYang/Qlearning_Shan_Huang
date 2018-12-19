import gym
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from keras import backend as K
from keras.models import load_model

env = gym.make('Breakout-ram-v0')
OBSERVATION_SPACE = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.n
LEARNING_RATE = 0.001
LEARNING_RATE_MIN = 0.00001
LEARNING_RATE_DECAY = 1
BATCH_SIZE = 32
TOTAL_EPISODE = 100000
GAME_PER_EPISODE = 1
MEMORY_SIZE = 100000
GAMMA = 0.99
EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.05
EXPLORATION_DECAY = 0.999

def Create_Q():
	model = Sequential()
	model.add(Dense(1024, input_shape = (OBSERVATION_SPACE,), activation = 'relu'))
	model.add(Dense(ACTION_SPACE, activation = 'linear'))
	return model
	
def Compile_Q():
	model = Create_Q()
	model.compile(loss = 'mse', optimizer = Adam(lr = LEARNING_RATE))
	return model

def Gameplay(QNN, current_state, previous_lives):
	#env.render()
	current_state = np.reshape(current_state, [1, OBSERVATION_SPACE])
	if random.random() > exploration_rate:
		action = np.argmax(QNN.predict(current_state)[0])
	else:
		action = random.randrange(ACTION_SPACE)
	next_state, reward, terminal, info = env.step(action)
	current_lives = info['ale.lives']
	next_state = np.reshape(next_state, [1, OBSERVATION_SPACE])
	if (current_lives < previous_lives):
		reward = -reward
		terminal = True
	else:
		reward = reward
		terminal = False
	record.append((current_state/255, action, reward, next_state/255, terminal))
	return next_state, reward, terminal, current_lives

def Train(QNN, record):
	batch = random.sample(record, BATCH_SIZE)
	for state, action, reward, next_state, terminal in batch:
		if (terminal):
			q = reward
		else:
			q = reward + GAMMA * np.amax(QNN.predict(next_state)[0])
		q_values = QNN.predict(state)
		q_values[0][action] = q
		QNN.fit(state, q_values, verbose = 0)
	return QNN
	
state = env.reset()
record = deque(maxlen=MEMORY_SIZE)
last_ten_performance = deque(maxlen = 10)
last_ten_average = 0
last_thousand_performance = deque(maxlen = 1000)
last_thousand_average = 0
last_ten_step = deque(maxlen = 10)
last_ten_average_step = 0
exploration_rate = EXPLORATION_MAX
episode = 0
learning = LEARNING_RATE
start = True
log = open("performance_curve_1024_1.txt", "a")
if start:
	QNN = Compile_Q()
else:
	QNN = load_model('breakoutmodel_1024_1.h5')
while True:
	if episode % 1000 ==0:
		QNN.save('breakoutmodel_1024_1.h5')
		last_thousand_average = np.mean(last_thousand_performance)
		log.write(str(episode) + ' '+str(last_thousand_average)+"\n")
	performance = 0
	total_step = 0
	for games in range(GAME_PER_EPISODE):
		step = 0
		state = env.reset()
		state = env.step(1)[0]
		state = np.reshape(state, [1, OBSERVATION_SPACE])
		lives = 5
		terminal = False
		while not terminal:
			state, reward, terminal, lives = Gameplay(QNN, state, lives)
			step = step + 1
			performance = performance + reward
		total_step += step
	performance = performance / float(GAME_PER_EPISODE)
	last_ten_performance.append(performance)
	last_thousand_performance.append(performance)
	total_step = total_step / float(GAME_PER_EPISODE)
	last_ten_step.append(total_step)
	episode = episode + 1
	if episode%10 == 0:
		last_ten_average_step = np.mean(last_ten_step)
		last_ten_average = np.mean(last_ten_performance)
		print("Episode"+ str(episode)+": Average score:"+str(last_ten_average)+", step:" + str(last_ten_average_step) + ", exploration rate "+str(exploration_rate))
	for i in range(GAME_PER_EPISODE):
		QNN = Train(QNN, record)
		exploration_rate *= EXPLORATION_DECAY
		exploration_rate = max(exploration_rate, EXPLORATION_MIN)