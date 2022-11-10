import numpy as np
import gym

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

import chainerrl
from chainerrl import replay_buffer
from chainerrl.agent import AttributeSavingMixin
from chainerrl.misc.batch_states import batch_states

class Actor(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        self.beta = 1.0
        self.min_prob = 0.0
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        h = self.l2(h)
        return chainerrl.distribution.SoftmaxDistribution(
            h, beta=self.beta, min_prob=self.min_prob)

class Critic(chainer.Chain):
    def __init__(self, obs_size, n_hidden_channels=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, 1)

    def __call__(self, x):
        batchsize = x.shape[0]
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        r = self.l2(h)
        return r.reshape(batchsize)

def disable_train(chain):
    call_orig = chain.__call__

    def call_test(self, x):
        with chainer.using_config('train', False):
            return call_orig(self, x)

    chain.__call__ = call_test


class AWRModel(chainer.Chain):
    def __init__(self, actor, critic):
        super().__init__(actor=actor, critic=critic)
        

class AWR(AttributeSavingMixin):
    ADV_EPS = 1e-5
        
    def __init__(self, 
                 env, 
                 model, 
                 actor_optimizer, 
                 critic_optimizer, 
                 replay_buffer,
                 gamma=0.95,
                 minibatch_size=32,
                 gpu=None,
                 phi=lambda x: x,
                 batch_states=batch_states):
        self.env = env
        self.model = model
        self.xp = self.model.xp
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.replay_buffer = replay_buffer
        self.minibatch_size = minibatch_size
        self.gpu = gpu
        self.phi = phi
        self.gamma = gamma
        self.batch_states = batch_states
        self.actor = self.model['actor']
        self.critic = self.model['critic']
        self.temp = 1.0
        self.weight_clip = 20

        self.total_actor_loss = 0
        self.total_critic_loss = 0
        
        
    def act(self, obs, test=True):
        with chainer.using_config('train', not test):
            s = self.batch_states([obs], self.xp, self.phi)
            r = self.actor(s)
            
            if test:
                a = F.argmax(r.logits, axis=1)
            else:
                a = r.sample()            

        return cuda.to_cpu(a.array[0])
    
    def get_statistics(self):
        return [
            ('average_q', self.average_q),
            ('average_loss', self.average_loss)
        ]
    
    def stop_episode(self):
        self.replay_buffer.stop_current_episode()
    
    def update(self, n_sample_size=128, errors_out=None):
        # Sample episodes
        batchsize = self.minibatch_size
        n_episodes = self.replay_buffer.n_episodes

        if n_sample_size > n_episodes:
            n_sample_size = n_episodes

        episodes = self.replay_buffer.sample_episodes(n_sample_size)
        
        # Compute Rewards with td-lambda
        data = self.sample_data(episodes)
        nb_data = len(data)
        perm  = np.random.permutation(nb_data)

        # Update critic
        self.total_critic_count = 0
        self.total_critic_loss = 0
        for i in range(0, nb_data, batchsize):
            batch_data = [data[idx] for idx in perm[i:i+batchsize]]
            states = np.array([d['episode']['state'] for d in batch_data])
            new_vals = np.array([d['new_val'] for d in batch_data])

            # create batch from states and new_rewards
            val = self.critic(states.astype(np.float32))            
            critic_loss = F.sum(F.square(val - new_vals))
            
            self.critic.cleargrads()
            critic_loss.backward()
            self.critic_optimizer.update()
    
            self.total_critic_count += batchsize
            self.total_critic_loss += critic_loss.data
        self.total_critic_loss /= self.total_critic_count
        

        # Re-compute Rewards with td-lambda
        data = self.sample_data(episodes)
        nb_data = len(data)
        perm  = np.random.permutation(nb_data)

        # Update actor
        self.total_actor_count = 0
        self.total_actor_loss = 0
        for i in range(0, nb_data, batchsize):
            states = np.array([d['episode']['state'] for d in batch_data])
            vals = np.array([d['val'] for d in batch_data])
            new_vals = np.array([d['new_val'] for d in batch_data])
            actions = np.array([d['episode']['action'] for d in batch_data])

            # create batch from states and new_rewards
            adv = new_vals - vals
            #adv = (adv - adv.mean()) / (adv.std() + self.ADV_EPS)
            weights = np.exp(adv / self.temp)
            weights = np.minimum(weights, self.weight_clip)

            val = self.actor(states.astype(np.float32))
            actor_loss = -F.sum(val.log_prob(actions) * weights) 
            
            self.actor.cleargrads()
            actor_loss.backward()
            self.actor_optimizer.update()
                        
            self.total_actor_count += batchsize
            self.total_actor_loss += actor_loss.data
        self.total_actor_loss /= self.total_actor_count
        
    def rollout_path(self, test=True, max_length=None):
        obs = self.env.reset()
        done = False
        R = 0
        t = 0
        
        path = []
        while not done:
            if max_length is not None and t >= max_length:
                break
                
            action = self.act(obs, test)
            new_obs, r, done, _ = env.step(action)            
            path.append({
                'state': obs,
                'reward': r,
                'action': action,
                'next_state': new_obs,
                'is_state_terminal': done
            })            
            R += r
            t += 1            
            obs = new_obs
        
        return path, R
    
    def rollout_train(self, num_samples, max_length=None):
        new_sample_count = 0
        total_return = 0
        path_count = 0
        
        while (new_sample_count < num_samples):
            path, path_return = self.rollout_path(test=False, max_length=max_length)                        
            self.store_path(path)
            
            new_sample_count += len(path)
            total_return += path_return
            path_count += 1

        avg_return = total_return / path_count
        return avg_return, path_count, new_sample_count
    
    def rollout_test(self, num_episodes, max_length=None, print_info=False):
        total_return = 0
        for e in range(num_episodes):
            path, path_return = self.rollout_path(test=True, max_length=max_length)
            total_return += path_return

            if (print_info):
                print("Episode: {:d}".format(e))
                print("Curr_Return: {:.3f}".format(path_return))
                print("Avg_Return: {:.3f}\n".format(total_return / (e + 1)))

        avg_return = total_return / num_episodes
        return avg_return, num_episodes    
    
    def stop_episode(self):
        self.replay_buffer.stop_current_episode()
        
    def store_path(self, path):
        for p in path:
            self.replay_buffer.append(**p)
        self.stop_episode()
            
    def get_total_samples(self):
        return len(self.replay_buffer)     
    
    def compute_return(self, episode, val_t, td_lambda=0.9):
        path_len = len(episode)

        return_t = np.zeros(path_len)
        last_val = episode[-1]['reward']
        return_t[-1] = last_val

        for i in reversed(range(0, path_len - 1)):    
            curr_r = episode[i]['reward']
            next_ret = return_t[i+1]
            curr_val = curr_r + self.gamma * ((1.0 - td_lambda) * val_t[i+1] + td_lambda * next_ret)
            return_t[i] = curr_val

        return return_t
    
    def sample_data(self, episodes):
        vals = []
        new_vals = []

        for episode in episodes:
            states = np.array([e['state'] for e in episode])
            val_t = self.critic(states.astype(np.float32))
            val_t = val_t.data
            new_val_t = self.compute_return(episode, val_t) 

            vals.append(val_t)
            new_vals.append(new_val_t)

        data = []
        for episode, val, new_val in zip(episodes, vals, new_vals):
            for e, v, nv in zip(episode, val, new_val):
                data.append({'episode': e, 'val': v, 'new_val': nv})


        return data

env = gym.make('Humanoid-v2')

obs_size = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

actor = Actor(obs_size, n_actions)
critic = Critic(obs_size)

model = AWRModel(actor=actor, critic=critic)

opt_a = chainer.optimizers.SGD(lr=0.00005)
opt_c = chainer.optimizers.SGD(lr=0.001)
opt_a.setup(actor)
opt_c.setup(critic)

rbuf = replay_buffer.EpisodicReplayBuffer(10 ** 6)
phi = lambda x: x.astype(np.float32, copy=False)

samples_per_iter=256
max_episode_length = 200

awr = AWR(env, model, opt_a, opt_c, rbuf, phi=phi)

for itr in range(10):
    for i in range(10):
        awr.rollout_train(samples_per_iter, max_episode_length)
        awr.update(n_sample_size=128)    
    ret = awr.rollout_test(1, max_episode_length, print_info=False)
    print('itr: ', itr, ret)

for i in range(100):
    obs = env.reset()
    done = False
    R = 0
    t = 0
    while not done and t < 200:
        env.render()
        action = awr.act(obs)
        obs, r, done, _ = env.step(action)
        R += r
        t += 1
    print('test episode:', i, 'R:', R)
    awr.stop_episode()
