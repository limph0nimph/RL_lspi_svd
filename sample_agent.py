def Sample_agent_policy(env, agent, is_agent = False):

  actions = {'left': 0, 'stop': 1, 'right': 2}
  def policy(obs, t):
    # Write the code for your policy here. You can use the observation
    # (a tuple of position and velocity), the current time step, or both,
    # if you want.
    position, velocity = obs
    if velocity<0:
      return actions['left']
    else:
      return actions['right']  

  def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
  def grey_and_scale(img):
    return cv2.resize(rgb2gray(img), dsize=(150, 100), interpolation=cv2.INTER_CUBIC)


  state_img2_sequence=[]
  action_sequence=[]
  reward_seqeunce=[]
  TIME_LIMIT=1000
  obs = env.reset()
  obs_array=env.render('rgb_array')

  res =grey_and_scale(obs_array)
  plt.imshow(res)
  prev_state_img = res

  action = policy(obs, 0)  # Call your policy

  obs, reward, done, _ = env.step(action)
  obs_array=env.render('rgb_array')
  res=grey_and_scale(obs_array)
      #make a tensor of 2 pictures to incorporate velocity tensor
  state_img2=np.zeros([prev_state_img.shape[0],prev_state_img.shape[1],2])
  state_img2[:,:,0] = prev_state_img
  state_img2[:,:,1] = res

  # saving more wisely  is desired
   #flatten
  state_img2_sequence.append(state_img2.flatten())
  for t in range(1, TIME_LIMIT):
      plt.gca().clear()
      
      #agent.predict(obs)
      if is_agent:
        action = agent.predict(state_img2_sequence[-1])
      else:
        action = policy(obs, t)
        # Call your policy
      obs, reward, done, _ = env.step(action)  # Pass the action chosen by the policy to the environment
      
      action_sequence.append(action)
      reward_seqeunce.append(reward)
      # We don't do anything with reward here because MountainCar is a very simple environment,
      # and reward is a constant -1. Therefore, your goal is to end the episode as quickly as possible.

      # Draw game image on display.
      obs_array=env.render('rgb_array')
      res = grey_and_scale(obs_array)

      #make a tensor of 2 pictures to incorporate velocity tensor
      state_img2=np.zeros([prev_state_img.shape[0],prev_state_img.shape[1],2])
      #flatten at this stage
      state_img2[:,:,0] = prev_state_img
      state_img2[:,:,1] = res
      # saving more wisely  is desired
      state_img2_sequence.append(state_img2.flatten())

      #visualisation
      plt.imshow(res)
      
      display.display(plt.gcf())
      display.clear_output(wait=True)

      #next state:
      prev_state_img = res.copy()
      if done:
          print("Well done!")
          break
  else:
      print("Time limit exceeded. Try again.")

  display.clear_output(wait=True)
  return state_img2_sequence, action_sequence, reward_seqeunce
