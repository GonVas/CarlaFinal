import numpy as np
from PIL import Image
import time

def run(env, until, save_dir, rank=0, max_episodes=1000, max_steps=10000):

    total_steps = 0

    start_time = time.time()

    for epi in range(max_episodes):
        env.reset()
        epi_reward = 0
        for step in range(max_steps):

            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)     

            epi_reward += reward
            if(done):
                #print('Done episode {:3d}, with total reward: {:2.2f}'.format(epi, epi_reward))
                break

            total_steps += 1
            #import pudb; pudb.set_trace()
            #cv2.imshow("Soccer Env", env.unwrapped.observation()[0]['frame'][:, :, ::-1])
            to_save = save_dir + "env_"+ str(rank) + '_' + str(total_steps) + '_' + str(int(time.time() * 1000)) + ".png"
            #import pudb; pudb.set_trace()

            #img_tosave = cv2.resize(obs, (300, 900), interpolation = cv2.INTER_AREA)
            #cv2.imwrite(to_save, img_tosave, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            #import pudb; pudb.set_trace()



            img = Image.fromarray((obs[0]*255).astype(np.uint8))

            r, g, b = img.split()
            final_data = Image.merge("RGB", (b, g, r))

            final_data.save(to_save)

            if(time.time() > until + start_time):
                print('Finished gathering data')
                return

            #cv2.imwrite(to_save, obs*255, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            #cv2.imshow('input'+str(rank), obs)
            #cv2.waitKey(1)


