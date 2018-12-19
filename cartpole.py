import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque


env = gym.make('CartPole-v1')
OBSERVATION_SPACE = env.observation_space.shape[0]
ACTION_SPACE = env.action_space.n
LEARNING_RATE = 0.01
BATCH_SIZE = 16
TOTAL_EPISODE = 2
GAME_PER_EPISODE = 2
MEMORY_SIZE = 100000
EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.05
EXPLORATION_DECAY = 0.095
GAMMA = 0.99

def Create_Q():
	model = Sequential()
	model.add(Dense(64, input_shape = (OBSERVATION_SPACE,), activation = 'relu'))
	#model.add(Dense(8, activation = 'relu'))
	model.add(Dense(ACTION_SPACE, activation = 'linear'))
	return model
	
def Compile_Q():
	model = Create_Q()
	model.compile(loss = 'mse', optimizer = Adam(lr = LEARNING_RATE))
	return model

def Gameplay(QNN, current_state):
	current_state = np.reshape(current_state, [1, OBSERVATION_SPACE])
	if random.random() > exploration_rate:
		action = np.argmax(QNN.predict(current_state)[0])
	else:
		action = random.randrange(ACTION_SPACE)
	next_state, reward, terminal, info = env.step(action)
	next_state = np.reshape(next_state, [1, OBSERVATION_SPACE])
	reward = reward if not terminal else -reward
	record.append((current_state, action, reward, next_state, terminal))
	return next_state, terminal

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
		
record = deque(maxlen=MEMORY_SIZE)
QNN = Compile_Q()		

log = open("performance_curve_cartpole_64_1_ave.txt", "a")

for round in range(100):
	exploration_rate = EXPLORATION_MAX
	last_fifty_performance = deque(maxlen = 50)
	last_fifty_average = 0
	QNN = Compile_Q()
	for episode in range(1001):
		if episode % 50 ==0:
			QNN.save('cartpole_64_1.h5')
			last_fifty_average = np.mean(last_fifty_performance)
			print("Episode" + str(episode) + ":Average score for last 50 episode is" + str(last_fifty_average))
			log.write(str(episode) + ' '+str(last_fifty_average)+"\n")
		performance = 0
		for games in range(GAME_PER_EPISODE):
			terminal = False
			step = 0
			state = env.reset()
			state = np.reshape(state, [1, OBSERVATION_SPACE])
			while not terminal:
				state, terminal = Gameplay(QNN, state)
				step += 1
			performance = performance + step
		performance = performance / float(GAME_PER_EPISODE)
		last_fifty_performance.append(performance)
		for batch in range(GAME_PER_EPISODE):
			Train(QNN, record)
			exploration_rate *= EXPLORATION_DECAY
			exploration_rate = max(exploration_rate, EXPLORATION_MIN)