import torch
import numpy as np
import random

from replaybuffer import ReplayBuffer
from network import Actor, Critic, AttentionCritic
from noisegenerator import OUNoise
from evaluator import PolicyEvaluator
from environment import UnicyclePushBoxEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
import numpy as np
import torch


def training_loop(env,buffer,noise,evaluator,
                              actor,critic,target_actor,target_critic,actor_opt,critic_opt,
                              tau,gamma,sigma,
                              num_agents,sigma_decay_rate,emb):
    terminator = False
    evaluation_counter =0
    episodes = 10001
    agent_radius =1
    max_steps = 500
    batch_size = 64
    epis_saved=0
    done_once = False
    start_steps = 500# minimum number of samples required for starting the game
    w=[]
    goal = torch.as_tensor(env.goal, dtype=torch.float32, device=device)
    print("model: centralized actor, centralized attentaion-based critic")
    print(f"Params: Tau:{tau}, Gamma:{gamma}, Starting Sigma:{sigma}, Sigma DR:{sigma_decay_rate}, NumAgents:{num_agents}, TokenLengths:{emb}")
    params_to_animate=[tau,gamma,sigma,sigma_decay_rate,num_agents,emb]
    for episode in range(episodes):
        if episode%200 ==0:
            print(f"episode : {episode}")

        state = env.reset(agent_radius)
        noise.reset()
        total_reward = 0

        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy().flatten()
            action += noise.sample(sigma) 
            for i in range(5): # action changes every 5 steps in the game
                next_state, reward, done,_,dist= env.step(action)
                if reward>0.1:
                    for _ in range(20)   : 
                        
                        buffer.add(state, action, reward, next_state, done)
                else:
                    buffer.add(state, action, reward, next_state, done)
                        
                state = next_state
                total_reward += reward
                if done:
                    break 

            # Learn
            if len(buffer) > start_steps :
                ran = np.random.randint(1,batch_size)
                batch = buffer.sample(batch_size-ran) # batch=[ states, actions, rewards, next_states, dones]

                s = torch.FloatTensor(batch['states']).to(device)
                a = torch.FloatTensor(batch['actions']).to(device)
                r = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(device)
                s2 = torch.FloatTensor(batch['next_states']).to(device)
                d = torch.FloatTensor(batch['dones']).unsqueeze(1).to(device)


                with torch.no_grad():
                    next_actions = target_actor(s2)
                    q_next, _ = target_critic(
                                                s2,
                                                next_actions,
                                                goal)
                    q_target = r + gamma * (1 - d) * q_next

                q_val , _ = critic(s, a,goal)
                critic_loss = torch.nn.functional.mse_loss(q_val, q_target)
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                q_critic,_ = critic(s, actor(s),goal)
                actor_loss = -q_critic.mean()
                actor_opt.zero_grad()
                actor_loss.backward()
                actor_opt.step()

                # Soft update targets
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if done:
                break
        sigma *=sigma_decay_rate
        trajectories, box_start_and_end, box_positions, box_theta, goal_reached ,box_reached= evaluator.evaluate(max_steps,printer =0)
        # Flatten the weight matrix
#        if episode%100 ==0:
        # Flatten the weight matrix
    #        flat_weights = actor.net[4].weight.view(-1)

            # Pick weights at indices 5, 8, 12, 19
#            indices = 5
 #           w1 = flat_weights[indices]
#
 #           # Optional: convert to NumPy
  #          w1 = w1.detach().cpu().numpy()
   #         w.append(w1)
        


        if (any(box_reached) and episode-epis_saved>50):    
            print(f"Results saved at episode {episode}")
            evaluator.animate(trajectories, box_positions, box_theta, params_to_animate, env.goal,env.box_size, env.env_size, episode=episode)
            evaluator.save(trajectories, box_start_and_end,env.goal, episode)
            epis_saved = episode

        # if we reach box for the first time wwe delete buffer:
        if sum(box_reached) ==num_agents and done_once == False:
            buffer.delete()
            
            done_once = True

        if goal_reached:                
            print(f"Goal reached at episode {episode}...\n")
            if sum(box_reached)==num_agents:
                evaluator.animate(trajectories, box_positions, box_theta,params_to_animate, env.goal,env.box_size, env.env_size, episode=episode)
                evaluator.save(trajectories, box_start_and_end,env.goal, episode)

                buffer.delete()    

            
                evaluation_counter+=1

                terminator= True  # or return to end training

        if terminator:
            print("learning completed. There is nothing to add")
            break

    return  0