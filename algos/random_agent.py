def run(env, max_episodes=100, max_steps=100):

	for epi in range(max_episodes):
		env.reset()
		epi_reward = 0
		for step in range(max_steps):

			action = env.action_space.sample()
			obs, reward, done, _ = env.step(action)		

			epi_reward += reward
			if(done):
				print('Done episode {:3d}, with total reward: {:2.2f}'.format(epi, epi_reward))
				break


