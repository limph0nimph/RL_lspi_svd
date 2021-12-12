state_img2_sequence=[]
action_sequence=[]
reward_seqeunce=[]

obs = env.reset()
obs_array=env.render('rgb_array')
img_grey=rgb2gray(obs_array)
res = cv2.resize(img_grey, dsize=(150, 100), interpolation=cv2.INTER_CUBIC)
plt.imshow(res)
prev_state_img = res
plt.savefig('/content/drive/MyDrive/RL_proj_car_images/car_img_iter'+str(t)+'.png')


##

action = policy(obs, t)  # Call your policy
obs, reward, done, _ = env.step(action)
obs_array=env.render('rgb_array')
img_grey=rgb2gray(obs_array)
res = cv2.resize(img_grey, dsize=(150, 100), interpolation=cv2.INTER_CUBIC)
    #make a tensor of 2 pictures to incorporate velocity tensor
state_img2=np.zeros([prev_state_img.shape[0],prev_state_img.shape[1],2])
state_img2[:,:,0] = prev_state_img
state_img2[:,:,1] = res

    # saving more wisely  is desired
    #flatten
state_img2_sequence.append(state_img2.flatten())
for t in range(1, TIME_LIMIT):
    plt.gca().clear()
    
    action = policy(obs, t)  # Call your policy
    obs, reward, done, _ = env.step(action)  # Pass the action chosen by the policy to the environment
    
    action_sequence.append(action)
    reward_seqeunce.apppend(reward)
    # We don't do anything with reward here because MountainCar is a very simple environment,
    # and reward is a constant -1. Therefore, your goal is to end the episode as quickly as possible.

    # Draw game image on display.
    obs_array=env.render('rgb_array')
    img_grey=rgb2gray(obs_array)
    res = cv2.resize(img_grey, dsize=(150, 100), interpolation=cv2.INTER_CUBIC)

    #make a tensor of 2 pictures to incorporate velocity tensor
    state_img2=np.zeros([prev_state_img.shape[0],prev_state_img.shape[1],2])
    #flatten at this stage
    state_img2[:,:,0] = prev_state_img
    state_img2[:,:,1] = res
    # saving more wisely  is desired
    state_img2_sequence.append(state_img2.flatten())

    #visualisation
    plt.imshow(res)
    plt.savefig('/content/drive/MyDrive/RL_proj_car_images/car_img_iter'+str(t)+'.png')
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




Actions_set=[0,1,2]
def make_A(action_sequence, reward_seqeunce, state_img2_sequence):

  feature_dict_action={}
  feature_next_action={}
  reward_action_dict={}
  for action in Actions_set:
    indices=np.arange(len(action_sequence))
    indices_act=indices[action_sequence==action]
    features = state_img2_sequence[indices_act]
    features_next = state_img2_sequence[indices_act+1]
    rewards_a=reward_seqeunce[action_sequence==action]


    feature_dict_action[action]=features
    feature_next_action[action]=features_next
    reward_action_dict[action] = rewards_a



  return feature_dict_action, feature_next_action, reward_action_dict
feature_dict_action, feature_next_action, reward_action_dict = make_A(action_sequence, reward_seqeunce, state_img2_sequence)




