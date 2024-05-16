import sys
import random
from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QDesktopWidget, QPushButton, QGraphicsPathItem
from PyQt5.QtWidgets import QGraphicsTextItem, QGraphicsItem, QMessageBox, QDialog
from PyQt5.QtCore import Qt, QTimer, QRectF, pyqtSignal, QPointF, QObject
from PyQt5.QtGui import QPen, QColor, QPainterPath, QBrush, QPainter, QPixmap

from datetime import datetime, timedelta
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.distributions import Independent, Normal, Categorical
from torch.nn.utils.convert_parameters import _check_param_device
import torch.nn as nn
from collections import OrderedDict
import gym
from datetime import datetime, timezone
from copy import deepcopy
from PIL import Image
import torchvision.transforms as transforms

import torch.multiprocessing as mp
import threading
import asyncio
import time
from functools import reduce
from operator import mul
from gym import spaces
from gym.vector import SyncVectorEnv as SyncVectorEnv_
from gym.vector.utils import concatenate, create_empty_array
from gym.envs.registration import load
from gym.wrappers import TimeLimit
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.distributions.kl import kl_divergence


class make_env(object):
    """
    这样做的好处是以后在创建环境的时候可以更加方便：
    env_carpole = make_env('CartPole-v1')
    然后后面创建环境的时候都使用: env = env_carpole()
    """
    def __init__(self, env_name, env_kwargs={}, seed=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.seed = seed

    def __call__(self):
        env = gym.make(self.env_name, **self.env_kwargs)
        if hasattr(env, 'seed'):
            env.seed(self.seed)
        return env

class Sampler(object):
    def __init__(self, env_name, env_kwargs, batch_size, policy, env=None, seed=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.batch_size = batch_size
        self.policy = policy
        self.seed = seed
        if env is None:
            env = gym.make(env_name, **env_kwargs)
        self.env = env
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        self.env.close()
        self.closed = False
    
    # 需要子类实现的方法，这里只是定义了接口
    def sample_async(self, args, **kargs):
        raise NotImplementedError()
    def sample(self, args, **kargs):
        return self.sample_async(args, **kargs)

def mujoco_wrapper(entry_point, **kwargs):
    normalization_scale = kwargs.pop('normalization_scale', 1.0)
    max_episode_steps = kwargs.pop('max_episode_steps', 200)
    env_class = load(entry_point)
    env = env_class(**kwargs)
    env = NormalizedActionWrapper(env, scale=normalization_scale) # 添加动作归一化
    env = TimeLimit(env, max_episode_steps=max_episode_steps) # 添加最大步数限制
    return env

class policy_gradient:
    def __init__(self, model, memory, config):
        self.gamma = config.gamma
        self.device = config.device
        self.policy_net = model.to(self.device)
        self.memory = memory
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=config.lr) # 能缓解多变量下降速度不平衡的问题
    
    def sample_action(self, state):
        state = torch.Tensor(state, dtype=torch.float32)
        state = torch.autograd.Variable(state) # 将tensor转换为Variable， variable是pytorch中的一种数据类型，可以进行自动求导
        probs = self.policy_net(state) # 通过神经网络得到动作的概率
        action = torch.distributions.Bernoulli(probs).sample() # 从服从 bernoulli分布的概率中随机抽取, 返回相同 shape的 0 or 1
        action = action.data.numpy().astype(int)[0]
        return action
    
    def update(self):
        state_pool, action_pool, reward_pool = self.memory.sample()
        state_pool, action_pool, reward_pool = list(state_pool), list(action_pool), list(reward_pool)
        running_add = 0
        for i in reversed(range(len(reward_pool))):
            reward_pool[i] = (self.gamma * reward_pool[i-1] + reward_pool[i]) if reward_pool[i] !=0 else 0
        # normalize
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std
        # gradient desent
        self.optimizer.zero_grad()
        for i in range(len(reward_pool)):
            state = torch.autograd.Variable(torch.Tensor(state_pool[i], dtype=torch.float32))
            action = torch.autograd.Variable(torch.Tensor(action_pool[i], dtype=torch.float32))
            reward = reward_pool[i]
            probs = self.policy_net(state)
            loss = - torch.distributions.Bernoulli(probs).log_prob(action) * reward # log (在给定的 Bernoulli分布中选中 action的概率)
            loss.backward()
        self.optimizer.step()
        self.memory.clear()

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.Net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.Net(x)

class Qlearning(object):
    def __init__(self, config):
        self.explore_type = config.explore_type
        self.n_actions = config.n_actions
        self.lr = config.lr
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.sample_cnt = 0
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay =  config.epsilon_decay
        self.epsilon_decay_flag = config.epsilon_decay_flag
        self.Q_table = np.zeros((config.n_states, self.n_actions)) # 实际上每一个staste的action个数都是不一样的, 这个地方不准确
        self.__ucb_init()
    
    def __ucb_init(self):
        self.ucb_sa_visit_cnt_arr = np.array([])
        self.ucb_cnt = 0
        self.ucb_sa_visit_cnt_arr = np.ones(self.Q_table.shape) # (s,a)
    
    @staticmethod
    def explore_type_space():
        return {'epsilon_greedy', 'boltzmann', 'ucb', 'special_ucb', 'softmax', 'thompson'}
    
    def __softmax(self, actions_v):
        Exp = np.exp(actions_v + 1e-3) # 小小的优化
        return Exp / np.sum(Exp, axis=0)
    
    def __softmax_policy_init(self):
        self.Q_table = np.random.random(self.Q_table.shape)
    
    def _e_greedy(self, state):
        self.epsilon = self.epsilon_end
        if self.epsilon_decay_flag:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.sample_cnt / self.epsilon_decay) # 选择指数递减
        if np.random.uniform(0, 1) < self.epsilon or sum(self.Q_table[state]) == 0:
            # 选择随机动作, 使用sum函数有点吃时间复杂度, 删掉问题不大 (自我感觉), 或者初始化为极小值也行
            # return np.random.randint(len(self.Q_table[state])) # number of actions in state
            return np.random.randint(self.n_actions) # 因为 actions个数都是一样的, 所以这个地方就不用上面的那种方法了
        return np.argmax(self.Q_table[state])

    def _sp_ucb_policy(self, s): # special_ucb 再前期用 epsilon-greedy进行一个优化
        # 在 RL^2中说可以把初始化 -> ucb_sa_visit_cnt 改为 1
        self.ucb_cnt += 1 # 迭代的轮数
        # 当使用过 nS种 action之前(使用同一种 action, 不用 state视为不同)使用 epsilon-greedy算法 (个人觉得不合理, 跑一次所有的 action不就已经有 nS 次了吗)
        not_state_once = np.sum(self.ucb_sa_visit_cnt_arr > 1, axis=1).sum() < self.ucb_sa_visit_cnt_arr.shape[0] # 先验动作
        if not_state_once:
            a_final = self._e_greedy(s)
            self.ucb_sa_visit_cnt_arr[s, a_final] += 1 # 用来记录在每个状态下选择每个行动的次数
            return a_final
        # softmax 可以使 sum=1 并且保持之前的大小关系, 这个地方相当于将两个部分分别进行了一个归一化
        b_t= self.__softmax(self.Q_table[s]) + self.__softmax(np.sqrt(2 * np.log(self.ucb_cnt) / self.ucb_sa_visit_cnt_arr[s]))
        a_final = np.argmax(b_t)
        self.ucb_sa_visit_cnt_arr[s, a_final] += 1
        return a_final
    
    def sample_action(self, state):
        self.sample_cnt += 1
        if self.explore_type == 'epsilon_greedy':
            return self._e_greedy(state)
        elif self.explore_type == 'boltzmann' or self.explore_type == 'softmax': # 属于 gradient bandit algorithm (sutton p37)
            if self.sample_cnt == 1 and self.explore_type == 'softmax': # 唯一不同就是有一个初始化
                self.__softmax_policy_init() # Q_table 为 (0,1)的随机数
            Ht = np.exp(self.Q_table[state] / self.epsilon)
            action_probs = Ht / np.sum(Ht)
            return np.random.choice(self.n_actions, p=action_probs)
        elif self.explore_type == 'ucb':
            if self.sample_cnt < self.n_actions:
                return self.sample_cnt # 这样就把所有的动作都试了一遍
            else:
                return np.argmax(self.__softmax(self.Q_table[state]) + self.__softmax(self.epsilon * np.sqrt(np.log(self.sample_cnt) / self.sample_cnt))) # 注意这个地方分母不是 cntA
        elif self.explore_type == 'special_ucb':
            return self._sp_ucb_policy(state)
        elif self.explore_type == 'thompson':
            alpha = self.__softmax(self.Q_table[state]) # 根据q值确定, q越大选择的概率越大, 在这里就是 alpha越大
            return np.argmax(np.random.beta(alpha, 1 - alpha))
        else:
            raise ValueError

    def predict_action(self, state):
        if self.explore_type in ['epsilon_greedy', 'ucb', 'special_ucb']:
            return np.argmax(self.Q_table[state]) # predict 的时候一直 exploitation
        elif self.explore_type == 'boltzmann' or self.explore_type == 'softmax':
            Ht = np.exp(self.Q_table[state] / self.epsilon)
            action_probs = Ht / np.sum(Ht)
            return np.random.choice(self.n_actions, p=action_probs)
        elif self.explore_type == 'thompson':
            alpha = self.__softmax(self.Q_table[state]) # 根据q值确定, q越大选择的概率越大, 在这里就是 alpha越大
            return np.argmax(np.random.beta(alpha, 1 - alpha))
    
    def update(self, state, action, reward, next_state, next_action, terminated):
        Q_predict = self.Q_table[state][action]
        Q_target = reward if terminated else (reward + self.gamma * self.Q_table[next_state][next_action])
        self.Q_table[state][action] += self.lr * (Q_target - Q_predict) # 采用的是固定学习率的方法

class Networks(nn.Module):
    """
    this class is used to create the network for maml
    we need special network to deal with the second order derivative
    """
    def __init__(self, net, config):
        super(Networks, self).__init__()
        self.params = nn.ParameterList()
        self.params_batch_norm = nn.ParameterList()
        self.net = net
        self.config = config
        for name, param in net:
            if name == 'conv2d':
                w = nn.Parameter(torch.ones(param[:4]))
                b = nn.Parameter(torch.zeros(param[0]))
                torch.nn.init.kaiming_normal_(w)
                self.params.append(w)
                self.params.append(b)
            elif name == 'linear':
                w = nn.Parameter(torch.ones(*param))
                b = nn.Parameter(torch.zeros(param[0]))
                torch.nn.init.kaiming_normal_(w)
                self.params.append(w)
                self.params.append(b)
            elif name == 'batch_norm':
                w = nn.Parameter(torch.ones(param[0]))
                b = nn.Parameter(torch.zeros(param[0]))
                self.params.append(w)
                self.params.append(b)
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.params_batch_norm.extend([running_mean, running_var])
            elif name in ['flatten', 'relu', 'tanh', 'sigmoid', 'max_pool2d']:
                continue
            else:
                raise NotImplementedError
    
    def forward(self, x, params=None, bn_training=True):
        if params is None:
            params = self.params
        idx, bn_idx = 0, 0
        for name, param in self.net:
            if name == 'conv2d':
                x = F.conv2d(x, params[idx], params[idx+1], stride=param[4], padding=param[5])
                idx += 2
            elif name == 'linear':
                x = F.linear(x, params[idx], params[idx+1])
                idx += 2
            elif name == 'batch_norm':
                x = F.batch_norm(x, self.params_batch_norm[bn_idx], self.params_batch_norm[bn_idx+1], weight=params[idx], bias=params[idx+1], training=bn_training)
                idx += 2
                bn_idx += 2
            elif name == 'flatten':
                x = x.view(x.size(0), -1) # (batch, -1)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = F.sigmoid(x)
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            else:
                raise NotImplementedError
        return x
    
    def zero_grad(self, params=None):
        with torch.no_grad():
            if params is None:
                params = self.params
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

    def parameters(self):
        return self.params

class REINFORCE:
    def __init__(self, config):
        self.policy_net = PolicyNet(config.nS, config.hidden_dim, config.nA).to(config.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.gamma = config.gamma
        self.device = config.device
    
    def sample_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        probs = self.policy_net(state.reshape(1,-1))
        action_dist = torch.distributions.Categorical(probs)
        return action_dist.sample().item() # sample返回一个tensor, item返回一个数字 (第几个action)

    def update(self, transition_dict):
        reward_list = transition_dict['reward']
        state_list = transition_dict['state']
        action_list = transition_dict['action']
        
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor(state_list[i], dtype=torch.float32).to(self.device)
            action = torch.tensor(action_list[i]).to(self.device)
            log_prob = torch.log(self.policy_net(state.reshape(1,-1))[0, action])
            G = self.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()

class SyncVectorEnv(SyncVectorEnv_):
    def __init__(self, env_fns, observation_space=None, action_space=None, **kwargs):
        super(SyncVectorEnv, self).__init__(env_fns, observation_space=observation_space, action_space=action_space, **kwargs)
        for env in self.envs:
            if not hasattr(env.unwrapped, 'reset_task'):
                raise ValueError('envs must contain a reset_task method for SyncVectorEnv')
        self._dones = np.zeros(len(self.envs), dtype=np.bool_) # 记录是不是每一个环境都结束了

    @property
    def dones(self):
        return self._dones
    
    def reset_task(self, task):
        # self._dones[:] = False
        for env in self.envs:
            env.unwrapped.reset_task(task)
    
    def step_wait(self): # SyncVectorEnv的step会调用step_wait，让每一个环境都step一步
        observations_list, infos = [], []
        batch_ids, j = [], 0
        num_actions = self.action_space.shape[0]
        rewards = np.zeros((num_actions,), dtype=np.float32)
        for i, (env, action) in enumerate(zip(self.envs, self._actions)): # step一轮
            if self._dones[i]:
                continue
            observation, rewards[j], self._dones[i], truncated, info = env.step(action)
            self.dones[i] = self.dones[i] or truncated
            batch_ids.append(i)
            if not self._dones[i]:
                observations_list.append(observation)
                infos.append(info)
            j += 1
        assert num_actions == j
        return (np.array(observations_list), rewards, np.copy(self._dones), {'batch_ids': batch_ids, 'infos': infos})

class NormalizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, scale=1.0):
        super(NormalizedActionWrapper, self).__init__(env)
        self.scale = scale
        self.action_space = spaces.Box(low=-self.scale, high=self.scale, shape=self.env.action_space.shape, dtype=np.float32)
    
    def action(self, action):
        """
        把一个动作剪切到[-scale, scale]之间，然后再映射回去[lb, ub]
        """
        action = np.clip(action, -self.scale, self.scale) # 剪切到[-scale, scale]之间
        lb, ub = self.env.action_space.low, self.env.action_space.high # 找到action的最大值和最小值
        if np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)): # 如果最大值和最小值都是有限的
            action = lb + (action + self.scale) * (ub - lb) / (2 * self.scale) # 将action加上scale防止小于0
            # 然后乘上(ub-lb)除以2*scale，这样就可以将action映射到[0,1]*(ub-lb)之间，再加上lb映射回去了相当于
            action = np.clip(action, lb, ub)
        return action

class GradientBasedMetaLearner(object):
    def __init__(self, policy, device='cpu'):
        super(GradientBasedMetaLearner, self).__init__()
        self.device = torch.device(device)
        self.policy = policy
        self.policy.to(self.device)
        self._event_loop = asyncio.get_event_loop()
    
    def adapt(self, episodes, *args, **kwargs):
        raise NotImplementedError
    
    def step(self, train_episodes, valid_episodes, *args, **kwargs):
        raise NotImplementedError
    
    def _async_gather(self, coros): # 传入多个task, 然后运行等待所有完成, 返回多个task的结果
        coro = asyncio.gather(*coros)
        return zip(*self._event_loop.run_until_complete(coro))

class MAMLTRPO(GradientBasedMetaLearner):
    def __init__(self, policy, fast_lr=0.5, first_order=False, device='cpu'):
        super(MAMLTRPO, self).__init__(policy, device)
        self.fast_lr = fast_lr
        self.first_order = first_order

    async def adapt(self, train_episode_futures, first_order=None):
        # inner_loop更新一次参数, 然后给最外层outerloop使用loss更新
        if first_order is None:
            first_order = self.first_order
        params = None
        for train_episode in train_episode_futures: # 把一个trajectory拿来更新
            inner_loss = reinforce_loss(self.policy, await train_episode, params=params)
            params = self.policy.update_params(inner_loss, params, step_size = self.fast_lr, first_order=first_order)
        return params
    
    def hessian_vector_product(self, kl, damping=1e-2): # 返回一个函数
        # kl是衡量两个分布之间的差异的指标, kl越大,两个分布越不相似
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flag_grad_kl = parameters_to_vector(grads)
        def _product(vector, retain_graph=True):
            grad_kl_v = torch.dot(flag_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters(), retain_graph=retain_graph)
            flat_grad2_kl = parameters_to_vector(grad2s)
            return flat_grad2_kl + damping * vector
        return _product #!看看这个函数在那里use
    
    async def surrogate_loss(self, train_futures, valid_futures, old_pi=None):
        # 计算一个替代的loss, 用来更新参数, 使用新旧两个分布之间的差异来计算, 会更加稳定
        # 如果没有old_pi就会将valid_episodes丢进policy算一个pi出来
        first_order = (old_pi is not None) or self.first_order # 如果old_pi为空, 就只能依赖于当前的策略梯度, 所以我们不能first_order, 但是如果为不为空,那么说明可以用就策略来计算新策略, 也就是通过一阶梯度策略改进策略
        params = await self.adapt(train_futures, first_order) # 先进行内循环更新
        with torch.set_grad_enabled(old_pi is None): # 如果old_pi为空, 则需要计算梯度
            # 因为在更新参数的时候必须使用到old_pi, 也就是说其他的时候计算surrogate_loss的时候都不需要计算梯度
            # 在更新新的参数的时候必须是从old_pi那里更新过来的(old_pi那里计算的时候相当于是一阶梯度了)
            valid_episodes = await valid_futures 
            pi = self.policy(valid_episodes.observations, params=params)
            if old_pi is None:
                old_pi = detach_distribution(pi) # 如果没有的话就复制一份, 但是不需要梯度
            log_ratio = (pi.log_prob(valid_episodes.actions) - old_pi.log_prob(valid_episodes.actions))
            ratio = torch.exp(log_ratio)
            # 和reinforce_loss相比, 这个地方的ratio是两个分布之间的差值, 而不是一个分布下的概率
            losses = - weighted_mean(ratio * valid_episodes.advantages, lengths=valid_episodes.lengths)
            kls = weighted_mean(kl_divergence(pi, old_pi), lengths=valid_episodes.lengths)
        return losses.mean(), kls.mean(), old_pi
    
    def step(self, train_futures, valid_futures, max_kl=1e-3, cg_iters=10, cg_damping=1e-2, ls_max_steps=10, ls_backtrack_ratio=0.5):
        num_tasks = len(train_futures[0]) # 也就是num_steps
        logs = {}
        old_losses, old_kls, old_pis = self._async_gather([self.surrogate_loss(train, valid, old_pi=None) for (train, valid) in zip(zip(*train_futures), valid_futures)])
        # train_futures是一个列表, shape为 (m, n)表示, 每个任务有m个trajectory, 一共有n个不同的任务
        # [ [traj1_task1, traj1_task2, ..., traj1_taskn] ...
        # [trajm_taskm, trajm_task2, ..., trajm_taskn] ]
        # 这里使用 zip(* train_futures)就可以把每个任务的trajectory放在一起, 形成一个列表
        logs['loss_before'] = to_numpy(old_losses)
        logs['kl_before'] = to_numpy(old_kls)
        old_loss = sum(old_losses) / num_tasks 
        old_kl = sum(old_kls) / num_tasks
        grads = torch.autograd.grad(old_loss, self.policy.parameters(), retain_graph=True)
        grads = parameters_to_vector(grads)
        hessian_vector_product = self.hessian_vector_product(old_kl, damping=cg_damping) # 定义的是一个函数
        stepdir = conjugate_gradient(hessian_vector_product, grads, cg_iters=cg_iters)
        
        lagrange_multiplier = torch.sqrt(0.5 * torch.dot(stepdir, hessian_vector_product(stepdir, False)) / max_kl)
        step = stepdir / lagrange_multiplier
        old_params = parameters_to_vector(self.policy.parameters())
        
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step, self.policy.parameters()) # 就是更新一下参数
            losses, kls, _ = self._async_gather([self.surrogate_loss(train, valid, old_pi=old_pi) for (train, valid, old_pi) in zip(zip(*train_futures), valid_futures, old_pis)])
            improve = (sum(losses) / num_tasks) - old_loss
            kl = sum(kls) / num_tasks
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                logs['loss_after'] = to_numpy(losses)
                logs['kl_after'] = to_numpy(kls)
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters()) # 如果for循环正常结束, 则不更新参数
        return logs

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class omniglot:
    """
    This class is used to load the omniglot dataset.
    the data set is divided into two parts: train and test.
    train: images_background
    test: images_evaluation
    the structure of the folders is : language/character/png
    the main content of the data is the alphabet of each language
    """
    def init_data(self, args):
        def read_data(name):
            data = []
            resize_method = transforms.Resize((args.image_size, args.image_size))
            for language_index, language_label in enumerate(os.listdir(os.path.join('data', name))):
                for label_index, label in enumerate(os.listdir(os.path.join('data', name, language_label))):
                    tempdata = []
                    for img_index, img in enumerate(os.listdir(os.path.join('data', name, language_label, label))):
                        filename = os.path.join('data', name, language_label, label, img)
                        img = resize_method(Image.open(filename).convert('L'))
                        tempdata.append(np.array(img).astype(float).reshape(1, args.image_size, args.image_size) / 255.0)
                    data.append(np.array(tempdata))
            return data
        
        train_data = np.array(read_data('images_background')) # (964, 20, 1, x, x)
        test_data = np.array(read_data('images_evaluation')) # (659, 20, 1, x, x)
        print(train_data.shape, test_data.shape)
        return train_data, test_data
    
    def __init__(self, args):
        if os.path.exists(os.path.join(args.data_path, 'data.npy')):
            self.train_data, self.test_data = np.load(os.path.join(args.data_path, 'data.npy'), allow_pickle=True)
        else:
            self.train_data, self.test_data = self.init_data(args)
            np.save(os.path.join(args.data_path, 'data.npy'), (self.train_data, self.test_data)) # save the data
        self.batch_size = args.task_num
        self.image_size = args.image_size
        self.n_class = self.train_data.shape[0] + self.test_data.shape[0]
        self.n_way = args.n_way # n-way means the taks have n-way classes
        self.k_shot = args.k_spt # k-shot means the task have k-shot samples for each class
        self.k_query = args.k_qry # k-query means the task have k-query samples for each class
        self.batch_index = {"train":0, "test":0}
        self.data_cache = {"train":self.load_data_cache(self.train_data), "test":self.load_data_cache(self.test_data)}
        #TODO: can dynamically save the training model without having to train from sctrach

    def load_data_cache(self, data):
        """
        This function is used to prepare the N-shot learning data
        you will receive support_x, support_y, query_x, query_y
        (support_x) the return value's shape is (sample, batch_size, self.n_way * self.k_shot, 1, self.image_size, self.image_size)
        "sample" is used for self.batch_index
        (support_y) the return value's shape is (sample, batch_size, self.n_way * self.k_shot)
        """
        data_cache = []
        for sample in range(10):
            support_set_feature, support_set_label, query_set_feature, query_set_label = [], [], [], []
            for i in range(self.batch_size): # batch_size is the number of tasks, also known as meta_batch_size (default is 256)
                support_x, support_y, query_x, query_y = [], [], [], []
                selected_class = np.random.choice(data.shape[0], self.n_way, replace=False)
                for j, current_class in enumerate(selected_class): # for each selected class
                    selected_image = np.random.choice(20, self.k_shot+self.k_query, replace=False)
                    support_x.append(data[current_class][selected_image[:self.k_shot]])
                    support_y.append([j for _ in range(self.k_shot)])
                    query_x.append(data[current_class][selected_image[self.k_shot:]])
                    query_y.append([j for _ in range(self.k_query)])
                
                permutation = np.random.permutation(self.n_way * self.k_shot) # (self.n_way * self.k_shot) total of data
                support_x = np.array(support_x).reshape(self.n_way * self.k_shot, 1, self.image_size, self.image_size)[permutation] # shuffle the support set
                support_y = np.array(support_y).reshape(self.n_way * self.k_shot)[permutation]
                permutation = np.random.permutation(self.n_way * self.k_query) # (self.n_way * self.k_query) total of data
                query_x = np.array(query_x).reshape(self.n_way * self.k_query, 1, self.image_size, self.image_size)[permutation] # shuffle the query set
                query_y = np.array(query_y).reshape(self.n_way * self.k_query)[permutation]
                # after the operation above, the shape of the data is (self.n_way * self.k_shot or self.k_query, 1, self.image_size, self.image_size)
                # after all the operation this loop, the shape of the data add new dimension: batch_size
                support_set_feature.append(support_x), support_set_label.append(support_y)
                query_set_feature.append(query_x), query_set_label.append(query_y)
            support_set_feature = np.array(support_set_feature).astype(np.float32).reshape(self.batch_size, self.n_way * self.k_shot, 1, self.image_size, self.image_size)
            support_set_label = np.array(support_set_label).astype(np.int).reshape(self.batch_size, self.n_way * self.k_shot)
            query_set_feature = np.array(query_set_feature).astype(np.float32).reshape(self.batch_size, self.n_way * self.k_query, 1, self.image_size, self.image_size)
            query_set_label = np.array(query_set_label).astype(np.int).reshape(self.batch_size, self.n_way * self.k_query)
            data_cache.append([support_set_feature, support_set_label, query_set_feature, query_set_label])
        return data_cache

    def next(self, mode):
        """
        get the next batch of data
        mode: train or test
        """
        if self.batch_index[mode] >= len(self.data_cache[mode]):
            self.batch_index[mode] = 0
            self.data_cache[mode] = self.load_data_cache(self.train_data if mode == "train" else self.test_data)
        next_batch = self.data_cache[mode][self.batch_index[mode]]
        self.batch_index[mode] += 1
        return next_batch

class Policy(nn.Module):
    def __init__(self, input_size,output_size):
        # 传入了一个module，他的数据都在self.named_parameters()里面
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.named_meta_parameters = self.named_parameters
        self.meta_parameters = self.parameters
    
    def update_params(self, loss, params=None, step_size=0.5, first_order=False):
        # 需要单独把params拿出来，是因为方便计算first_order,second_order的梯度
        if params is None: # 默认是policy自己的参数
            params = OrderedDict(self.named_parameters()) # deep copy, 有序字典
        grads = torch.autograd.grad(loss, params.values(), create_graph=not first_order)
        # 如果只需要一阶导数，那么就不需要创建计算图，这样可以节省内存
        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad
        return updated_params

class NormalMLPPolicy(Policy):
    def __init__(self, input_size, output_size, hidden_sizes=(), nonlinearity=F.relu, init_std=1.0, min_std=1e-6):
        # hidden_sizes 是一个元组，表示隐藏层的神经元个数 比如(8, 16, 32, 16)
        super(NormalMLPPolicy, self).__init__(input_size, output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.min_log_std = np.log(min_std) # 也就是std的最小值不能小于 -6
        self.num_layers = len(hidden_sizes) + 1 # 还有一个输入层

        layer_sizes = (input_size, ) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        self.mu = nn.Linear(layer_sizes[-1], output_size)    # 自己学习到均值的合适取值 （比较奇怪） 
        self.sigma = nn.Parameter(torch.Tensor(output_size)) # 标准差
        self.sigma.data.fill_(np.log(init_std)) # 用对数的形式初始化标准差
        self.apply(weight_init)

    def forward(self, input, params=None):
        if params is None: # 和Policy的一样，如果没有传入参数，就用自己的参数
            params = OrderedDict(self.named_parameters())
        if isinstance(input, np.ndarray):
            input = torch.tensor(input, dtype=torch.float32)
        output = input.to(torch.float32) # 初始化输入
        for i in range(1, self.num_layers):
            output = F.linear(output, weight=params['layer{0}.weight'.format(i)], bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        mu = F.linear(output, weight=params['mu.weight'], bias=params['mu.bias'])
        scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std)) # 限制标准差的最小值，避免不稳定
        """
        Normal ：是正态分布, 输入是均值和标准差, 输出是一个正态分布
        Independent ：是把一个分布变成独立的分布, 这里的mu和scale都是一个(n,)的向量, 但是Normal输出的东西不是独立的
                    他们之间还有一个相关系数, 所以用Independent把他们变成独立的分布
        """
        return Independent(Normal(loc=mu, scale=scale), 1)

class CategoricalMLPPolicy(Policy):
    def __init__(self, input_size, output_size, hidden_sizes=(), nonlinearity=F.relu):
        super(CategoricalMLPPolicy, self).__init__(input_size, output_size)
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.num_layers = len(hidden_sizes) + 1
        layer_sizes = (input_size, ) + hidden_sizes + (output_size, )
        for i in range(1, self.num_layers+1):
            self.add_module('layer{0}'.format(i), nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        self.apply(weight_init)
    
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        output = input
        for i in range(1, self.num_layers):
            output = F.linear(output, weight=params['layer{0}.weight'.format(i)], bias=params['layer{0}.bias'.format(i)])
            output = self.nonlinearity(output)
        logits = F.linear(output, weight=params['layer{0}.weight'.format(self.num_layers)], bias=params['layer{0}.bias'.format(self.num_layers)])
        # logits 就是最后一层的输出，即将丢到 softmax 中的值
        # Catelogical 就是 Samples are integers from {0,…,K−1} based on the logits provided. K是输出的维度
        return Categorical(logits=logits)

def get_policy_for_env(env, hidden_sizes=(100, 100), nonlinearity='relu'):
    input_size = get_input_size(env)
    # 举个例子 recduce(mul, (4, 5)), 1) = 4 * 5 * 1 = 20
    nonlinearity = getattr(torch, nonlinearity)
    if isinstance(env.action_space, gym.spaces.Box):
        output_size = reduce(mul, env.action_space.shape, 1)
        policy = NormalMLPPolicy(input_size, output_size, hidden_sizes=tuple(hidden_sizes), nonlinearity=nonlinearity)
    else:
        output_size = env.action_space.n
        policy = CategoricalMLPPolicy(input_size, output_size, hiddensizes=tuple(hidden_sizes), nonlinearity=nonlinearity)
    return policy

def get_input_size(env):# reduce 就是把 observation里面的全部元素相乘
    return reduce(mul, env.observation_space.shape, 1)

def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    # 共轭梯度, 用于求解Ax=b的解x, 输入的是一个函数f_Ax, 
    p = b.clone().detach()
    r = b.clone().detach()
    x = torch.zeros_like(b).float() # 初始解x通常取0向量
    rdotr = torch.dot(r, r)
    for i in range(cg_iters):
        z = f_Ax(p).detach()
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr.item() < residual_tol:
            break
    return x.detach()

def get_returns(episodes):
    # returns 是一系列动作之后获得的奖励总和
    return to_numpy([episode.rewards.sum(dim=0) for episode in episodes])

def reinforce_loss(policy, episodes, params=None):
    # policy是一个函数, 输入是observation, 输出是动作的分布, 形状是 [action, ]
    pi = policy(episodes.observations.view((-1, *episodes.observation_shape)), params=params) # 每个动作的分布
    # episode.observation的形状是 [trajectory_length, batch_size, observation_dim] -> [trajectory_length * batch_size, observation_dim]
    # print(pi.sample().shape) # torch.Size([maxlen * meta_batch(2000), 6])
    log_probs = pi.log_prob(episodes.actions.view((-1, *episodes.action_shape))) # (maxlen * meta_batch, )
    # log_probs就是计算在给定分布下的概率密度,log_probs的形状是 [trajectory_length * batch_size, ]
    # 因为是那action_dim(6个)同时取到的概率, 所以加起来最后一维action_dim就没有了
    log_probs = log_probs.view(len(episodes), episodes.batch_size)
    losses = - weighted_mean(log_probs * episodes.advantages, lengths=episodes.lengths) #我们希望增加优势高的动作的概率，减少优势低的动作的概率 
    # 因为我们想要最大化log_probs * advantages, 然而pytorch只有最小化的loss, 所以加个负号
    return losses.mean()

class SamplerWorker(mp.Process):
    # 采样器进程
    def __init__(self, index, env_name, env_kwargs, batch_size, observation_space, action_space, policy, baseline, task_queue, train_queue, valid_queue, policy_lock, seed=None):
        super(SamplerWorker, self).__init__()
        env_functions = [make_env(env_name, env_kwargs) for _ in range(batch_size)]
        self.envs = SyncVectorEnv(env_functions, observation_space=observation_space, action_space=action_space)
        self.batch_size = batch_size
        self.policy = policy
        self.baseline = baseline
        self.task_queue = task_queue
        self.train_queue = train_queue
        self.valid_queue = valid_queue
        self.policy_lock = policy_lock
        self.envs.seed(None if (seed is None) else seed + index * batch_size)

    def sample_trajectories(self, params=None):
        # 一个yield的生成器，每次返回一个轨迹
        observations, info = self.envs.reset()
        self.envs.dones[:] = False
        with torch.no_grad():
            while not self.envs.dones.all(): # 不是while True
                observations_tensor = torch.from_numpy(observations.astype('float32'))
                pi = self.policy(observations_tensor, params=params)
                actions_tensor = pi.sample()
                actions = actions_tensor.cpu().numpy()
                new_observations, rewards, _, infos = self.envs.step(actions)
                batch_ids = infos['batch_ids']
                yield (observations, actions, rewards, batch_ids)
                observations = new_observations
    
    def create_episode(self, params=None, gamma=0.95, gae_lambda=1.0, device='cpu'):
        episodes = BatchEpisodes(self.batch_size, gamma, device)
        for item in self.sample_trajectories(params):
            episodes.append(*item)
        self.baseline.fit(episodes)
        episodes.compute_advantages(self.baseline, gae_lambda, True)
        return episodes

    def sample(self, index, num_steps=1, fast_lr=0.5, gamma=0.95, gae_lambda=1.0, device='cpu'):
        params = None # 新开一个param, 在采样的时候单独使用
        for step in range(num_steps): # 采样num_steps个轨迹
            train_episode = self.create_episode(params, gamma, gae_lambda, device) # 创建一个episode
            self.train_queue.put((index, step, deepcopy(train_episode))) # deepcopy是为了防止多个进程之间的episode共享内存
            with self.policy_lock:
                loss = reinforce_loss(self.policy, train_episode, params)
                params = self.policy.update_params(loss, params, fast_lr, True) # 在这个地方更新param并不会影响到policy的param
        valid_episode = self.create_episode(params, gamma, gae_lambda, device)
        self.valid_queue.put((index, None, deepcopy(valid_episode)))
    
    def run(self):
        while True:
            data = self.task_queue.get()
            if data is None:
                self.envs.close()
                self.task_queue.task_done()
                break
            index, task, kwargs = data
            self.envs.reset_task(task)
            self.sample(index, **kwargs)
            self.task_queue.task_done() # 通知主进程任务完成



def _create_consumer(queue, futures, loop):
    while True:
        data = queue.get() # 等待采样器进程的数据, 如果没有数据, 就会阻塞在这里
        if data is None: # 传入None就代表结束
            break
        index, step, episode = data
        future = futures if (step is None) else futures[step]
        if not future[index].cancelled():
            loop.call_soon_threadsafe(future[index].set_result, episode)


class MultiTaskSampler(Sampler):
    def __init__(self, env_name, env_kwargs, batch_size, policy, baseline, env=None, num_workers=1, seed=None):
        super(MultiTaskSampler, self).__init__(env_name, env_kwargs, batch_size, policy, env, seed=seed)
        self.num_workers = num_workers

        self.task_queue = mp.JoinableQueue()
        self.train_episodes_queue = mp.Queue()
        self.valid_episodes_queue = mp.Queue()
        policy_lock = mp.Lock()
        self.workers = [
            SamplerWorker(
                index, env_name, env_kwargs, batch_size, self.env.observation_space, self.env.action_space, policy,
                deepcopy(baseline), self.task_queue, self.train_episodes_queue, self.valid_episodes_queue, policy_lock, seed
            ) for index in range(num_workers)
        ]
        for worker in self.workers:
            worker.daemon = True # 设置为守护进程
            worker.start()
        self._waiting_sample = False
        self._event_loop = asyncio.get_event_loop() # 创建一个事件循环
        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    def sample_tasks(self, num_tasks):
        return self.env.unwrapped.sample_tasks(num_tasks)

    def _start_consumer_threads(self, tasks, num_steps=1):
        # train
        train_episodes_futures = [[self._event_loop.create_future() for _ in tasks] for _ in range(num_steps)]
        self._train_consumer_thread = threading.Thread(target=_create_consumer, args=(self.train_episodes_queue, train_episodes_futures), kwargs={'loop':self._event_loop})
        self._train_consumer_thread.daemon = True # 设置为守护进程, 因为当主进程结束了, 就不需要继续了
        self._train_consumer_thread.start()
        # valid
        valid_episodes_futures = [self._event_loop.create_future() for _ in tasks]
        self._valid_consumer_thread = threading.Thread(target=_create_consumer, args=(self.valid_episodes_queue, valid_episodes_futures), kwargs={'loop':self._event_loop})
        self._valid_consumer_thread.daemon = True
        self._valid_consumer_thread.start()
        return (train_episodes_futures, valid_episodes_futures)

    def sample_async(self, tasks, **kwargs): # 传入task, 然后传入task_queue进行sample, 最后调用consumer进行set future
        if self._waiting_sample:
            raise RuntimeError('Already sampling!')
        for index, task in enumerate(tasks):
            self.task_queue.put((index, task, kwargs)) # SamplerWorker 已经开始采样了
        num_steps = kwargs.get('num_steps', 1)
        futures = self._start_consumer_threads(tasks, num_steps)
        self._waiting_sample = True
        return futures
    
    @property
    def valid_consumer_thread(self):
        return self._valid_consumer_thread
    
    @property
    def train_consumer_thread(self):
        return self._train_consumer_thread
    
    def sample_wait(self, episodes_futures): # 用来等待所有的异步采样操作完成
        if not self._waiting_sample:
            raise RuntimeError('Not sampling!')
        
        async def _wait(train_futures, valid_futures):
            train_episodes = await asyncio.gather(*[asyncio.gather(*futures) for futures in train_futures])
            valid_episodes = await asyncio.gather(*valid_futures)
            return train_episodes, valid_episodes
        
        samples = self._event_loop.run_until_complete(_wait(*episodes_futures)) # Run the event loop until a Future is done.
        self._join_consumer_threads()
        self._waiting_sample = False
        return samples
    
    def sample(self, tasks, **kwargs):
        futures =  self.sample_async(tasks, **kwargs)
        return self.sample_wait(futures)
    
    def _join_consumer_threads(self): # 等待所有的消费者线程结束, 分别关闭train和valid的消费者线程
        if self._train_consumer_thread is not None:
            self.train_episodes_queue.put(None) # 通知采样器进程结束
            self.train_consumer_thread.join()
        if self._valid_consumer_thread is not None:
            self.valid_episodes_queue.put(None)
            self.valid_consumer_thread.join()
        self._train_consumer_thread = None
        self._valid_consumer_thread = None

    def close(self):
        if self.closed:
            return
        for _ in range(self.num_workers):
            self.task_queue.put(None) # Put None之后join就不会阻塞了
        self.task_queue.join() # 等待所有的任务完成
        self._join_consumer_threads() # 等待所有的消费者线程结束
        self.closed = True

class make_env(object):
    """
    这样做的好处是以后在创建环境的时候可以更加方便：
    env_carpole = make_env('CartPole-v1')
    然后后面创建环境的时候都使用: env = env_carpole()
    """
    def __init__(self, env_name, env_kwargs={}, seed=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.seed = seed

    def __call__(self):
        env = gym.make(self.env_name, **self.env_kwargs)
        if hasattr(env, 'seed'):
            env.seed(self.seed)
        return env

class Sampler(object):
    def __init__(self, env_name, env_kwargs, batch_size, policy, env=None, seed=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.batch_size = batch_size
        self.policy = policy
        self.seed = seed
        if env is None:
            env = gym.make(env_name, **env_kwargs)
        self.env = env
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        self.env.close()
        self.closed = False
    
    # 需要子类实现的方法，这里只是定义了接口
    def sample_async(self, args, **kargs):
        raise NotImplementedError()
    def sample(self, args, **kargs):
        return self.sample_async(args, **kargs)

class First_Vist_MonteCarlo:
    def __init__(self, config):
        self.n_actions = config.n_actions
        self.epsilon = config.epsilon
        self.gamma = config.gamma
        self.Q_table = []
        self.return_sum = []
        self.return_cnt = []
    
    def sample_action(self, state):
        state = str(state)
        if state in self.Q_table.keys():
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, self.n_actions)
            else:
                return np.argmax(self.Q_table[state]) 
        else:
            return np.random.randint(0, self.n_actions)
    
    def predict_action(self, state):
        state = str(state)
        if state in self.Q_table.keys():
            return np.argmax(self.Q_table[state])
        else:
            return np.random.randint(0, self.n_actions)
    
    def update(self, one_ep_transition):
        sa_in_episode = set([(str(x[0]), x[1]) for x in one_ep_transition])
        for state, action in sa_in_episode:
            sa_pair = (state, action)
            # 可以空间换时间, 可以从前往后枚举 one_ep_transition, 然后 book掉没用的
            first_occurence_index = next(i for i, x in enumerate(one_ep_transition) if str(x[0]) == state and x[1] == action)
            G = sum(x[2] * (self.gamma ** i) for i, x in enumerate(one_ep_transition[first_occurence_index:])) # 时间复杂度瓶颈在这里, 但是也可以优化
            self.return_sum[sa_pair] += G
            self.return_cnt[sa_pair] += 1 # 因为 set的原因, 在这个地方每一个 epoch只会更新一次 (state, action)
            self.Q_table[state][action] = self.return_sum[sa_pair] / self.return_cnt[sa_pair] # 也可以用增量式更新

class LinearFeatureBaseline(nn.Module):
    def __init__(self, input_size, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()
        self.input_size = input_size
        self._reg_coeff = reg_coeff
        self.weight = nn.Parameter(torch.Tensor(self.feature_size,), requires_grad=False)
        self.weight.data.zero_()
        self._eye = torch.eye(self.feature_size, dtype=torch.float32, device=self.weight.device)
    
    @property
    def feature_size(self): # input_size 就是observation的所有维度乘起来
        return 2 * self.input_size + 4 # dim=2这个维度就是这么多，最后会flatten, 所以只用看的dim=2这里, 两个observations就是这个乘2, 剩下的就是每个都是1
    
    def _feature(self, episodes): # 手动构造特征
        ones = episodes.mask.unsqueeze(2) # unsqueeze: add a dimension
        observations = episodes.observations
        time_step = torch.arange(len(episodes)).view(-1, 1, 1) * ones / 100.0
        # 前面先生成一个id(从0~episodes-1), 然后乘上ones，有数字的地方就有id, 除以100.0是为了防止过大
        return torch.cat([observations, observations ** 2, time_step, time_step ** 2, time_step ** 3, ones], dim=2)
    
    def fit(self, episodes):
        # 用feature去拟合returns, 也就是得到了observation去找returns
        # 运行fit会修改self.weight
        featmat = self._feature(episodes).view(-1, self.feature_size) # flatten
        returns = episodes.returns.view(-1, 1)
        flat_mask = episodes.mask.flatten() # (a*b,)
        # 去掉是0的部分
        flat_mask_idx = torch.nonzero(flat_mask)
        featmat = featmat[flat_mask_idx].view(-1, self.feature_size)
        returns = returns[flat_mask_idx].view(-1, 1)
        # 计算
        reg_coeff = self._reg_coeff
        XT_y = torch.matmul(featmat.t(), returns)
        XT_X = torch.matmul(featmat.t(), featmat)
        for i in range(5):
            try:
                coeffs= torch.linalg.lstsq(XT_y, XT_X + reg_coeff * self._eye, driver='gelsy')
                if torch.isnan(coeffs.solution).any() or torch.isinf(coeffs.solution).any():
                    raise RuntimeError
                break
            except RuntimeError:
                reg_coeff *= 10
        else:
            raise RuntimeError('Unable to compute baseline beacause of singular matrix')
        self.weight.copy_(coeffs.solution.flatten())


    def forward(self, episodes):
        # 输入episodes, 输出每个observation的value
        features = self._feature(episodes) # (100, 20, 38)
        values = torch.mv(features.view(-1, self.feature_size), self.weight) # (2000, 38) * (38,) = (2000,)
        return values.view(features.shape[:2]) # (100, 20)

def weighted_mean(tensor, lengths=None):
    """
    输入tensor的shape是(max_length, batch_size, obs_dim)
    输出的shape是(batch_size, obs_dim)
    返回的是每个episode的平均值
    """
    if lengths is None: # 如果没有传入lengths，就直接求平均值
        return torch.mean(tensor) # 直接返回这个单个episode的平均值
    if tensor.dim() < 2:
        raise ValueError('error at weighted_mean, tensor must be at least 2D')
    for i, lengths in enumerate(lengths):
        tensor[lengths:, i].fill_(0.)
    extra_dims =  (1,) * (tensor.dim() - 2)
    lengths = torch.as_tensor(lengths, dtype=torch.float32)
    out = torch.sum(tensor, dim=0) # 在dim=0上求和，也就是把一个episode的所有值加起来
    out.div_(lengths.view(-1, *extra_dims)) # 然后每个episode都出来除以这个episode的长度
    # 这里lenghths变成了(batch_size, 1, 1, 1, ...)的形式，然后和out做除法，就相当于每个episode都除以这个episode的长度
    return out

def weighted_normalize(tensor, lengths=None, epsilon=1e-8):
    mean = weighted_mean(tensor, lengths)
    out = tensor - mean.mean() # 现在out的均值为0
    for i, length in enumerate(lengths):
        out[length:, i].fill_(0.)
    std = torch.sqrt(weighted_mean(out ** 2, lengths).mean()) # $$ \sigma = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2} $$
    out.div_(std + epsilon)
    return out

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (tuple, list)):
        return np.stack([to_numpy(t) for t in tensor], axis=0) # 如果使用np.array, 可能会有广播操作，或者数据类型的转换
    else:
        raise NotImplementedError('to_numpy not implemented for type')

def detach_distribution(pi):
    # detach可以把一部分计算图固定住,只更新另一部分
    if isinstance(pi, Independent): # Independent是多个分布组成的, 要分别detach
        return Independent(detach_distribution(pi.base_dist), pi.reinterpreted_batch_ndims)
    elif isinstance(pi, Normal):
        return Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
    elif isinstance(pi, Categorical):
        return Categorical(probs=pi.probs.detach())
    else:
        raise NotImplementedError('detach_distribution not implemented for type')

def vector_to_parameters(vector, parameters):
    param_device = None
    pointer = 0
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        num_param = param.numel() # 返回参数的元素个数
        param.data.copy_(vector[pointer:pointer + num_param].view_as(param).data)
        pointer += num_param

class BatchEpisodes(object):
    def __init__(self, batch_size, gamma=0.95, device='cpu'):
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        # 保存数据的列表
        self._observations_list = [[] for _ in range(batch_size)] # 创建一个batch_size大小的列表，每个元素是一个空列表 (batch, max_length, obs_dim)
        self._actions_list = [[] for _ in range(batch_size)]
        self._rewards_list = [[] for _ in range(batch_size)] # (batch, max_length)
        # 定义私有变量 (一般以_开头)
        self._observation_shape = None
        self._action_shape = None
        # maxlength 是每个episode的最大长度
        self._observations = None # (max_length, batch_size, obs_dim)
        self._actions = None      # (max_length, batch_size, action_dim)
        self._rewards = None      # (max_length, batch_size)
        self._returns = None      # (max_length, batch_size)
        self._advantages = None
        self._mask = None         # (max_length, batch_size)
        self._lengths = None
        self._logs = {}
    
    @property # 用于将方法转换为属性调用
    def observation_shape(self):
        if self._observation_shape == None:
            self._observation_shape = self.observations.shape[2:] # (batch_size, max_length, obs_dim), max_length是每个episode的最大长度
        return self._observation_shape
    
    @property
    def action_shape(self):
        if self._action_shape == None:
            self._action_shape = self.actions.shape[2:]
        return self._action_shape
    
    @property
    def observations(self):
        if self._observations == None:
            observation_shape = self._observations_list[0][0].shape # (max_length, obs_dim)
            observations = np.zeros((len(self), self.batch_size) + observation_shape, dtype=np.float32)
            # len(self)是episode的数量，在有面有定义, observations的shape是(max_length, batch_size, obs_dim)
            # 注意下observations的第一个维度不是batch_size，而是max_length
            for i in range(self.batch_size):
                length = self.lengths[i]
                np.stack(self._observations_list[i], axis=0, out=observations[:length, i])
                # stack只stack一个元素，axis=0，就相当于把前面这个变量复制到后面这个observations[:length, i]里面
            self._observations = torch.as_tensor(observations, device=self.device) # as_tensor是把numpy转换成tensor, 共享内存
            del self._observations_list # 删除这个列表，节省内存
        return self._observations
    
    @property
    def actions(self):
        if self._actions == None:
            action_shape = self._actions_list[0][0].shape
            actions = np.zeros((len(self), self.batch_size) + action_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = self.lengths[i]
                np.stack(self._actions_list[i], axis=0, out=actions[:length, i])
            self._actions = torch.as_tensor(actions, device=self.device)
            del self._actions_list
        return self._actions
    
    @property
    def rewards(self):
        if self._rewards == None:
            rewards = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = self.lengths[i]
                np.stack(self._rewards_list[i], axis=0, out=rewards[:length, i])
            self._rewards = torch.as_tensor(rewards, device=self.device)
            del self._rewards_list
        return self._rewards
    
    @property
    def returns(self):
        if self._returns == None:
            self._returns = torch.zeros_like(self.rewards) # zeros_like是创建一个和self.rewards一样shape的tensor，但是值都是0
            return_ = torch.zeros((self.batch_size,), dtype=torch.float32) # (batch_size,)
            for i in range(len(self)-1, -1, -1): # 从len(self)-1到0，步长为1 (倒着来)
                return_ = self.gamma * return_ + self.rewards[i] * self.mask[i] # mask是一个0-1的矩阵
                self._returns[i] = return_
        return self._returns
    
    @property
    def mask(self):
        #  有reward的地方就是1,没有的地方就是0,因为rewards的shape是(max_length, batch_size)，但是不是所有的episode都是max_length
        if self._mask == None:
            self._mask = torch.zeros((len(self), self.batch_size), dtype=torch.float32, device=self.device)
            for i in range(self.batch_size):
                lenghth = self.lengths[i]
                self._mask[:lenghth, i].fill_(1.0)
        return self._mask

    @property
    def advantages(self):
        if self._advantages == None:
            raise ValueError('advantages is not computed yet')
        return self._advantages
    
    @property
    def logs(self):
        return self._logs
    
    def log(self, key, value):
        self.logs_[key] = value

    def append(self, observations, actions, rewards, batch_ids):
        for observation, action, reward, batch_ids in zip(observations, actions, rewards, batch_ids):
            if batch_ids == None:
                continue
            self._observations_list[batch_ids].append(observation.astype(np.float32))
            self._actions_list[batch_ids].append(action.astype(np.float32))
            self._rewards_list[batch_ids].append(reward.astype(np.float32))

    def compute_advantages(self, baseline, gae_lambda=1.0, normalize=True):
        """
        advantage是一个用于评估某个动作相对于平均水平的指标
        """
        values = baseline(self).detach()  # values : (max_length, batch_size)
        values = F.pad(values * self.mask, (0, 0, 0, 1)) # 相当于在最后一行后面加了一排0
        
        deltas = self.rewards + self.gamma * values[1:] - values[:-1]
        # deltas 每个元素都是当前时间步的advantage, 计算方式是将当前时间步的reward加上下一个时间步的value，再减去当前时间步的value
        # 这个算法就是GAE里面的算法
        self._advantages = torch.zeros_like(self.rewards)
        gae = torch.zeros((self.batch_size,), dtype=torch.float32)
        for i in range(len(self)-1, -1, -1):
            gae = gae * self.gamma * gae_lambda + deltas[i] 
            self._advantages[i] = gae
        
        if normalize:
            self._advantages =  weighted_normalize(self._advantages, lengths=self.lengths)
        del self._returns
        del self._mask
        return self._advantages

    @property
    def lengths(self): # lengths 是一个列表，每个元素是一个episode的长度
        if self._lengths == None:
            self._lengths = [len(rewards) for rewards in self._rewards_list] #TODO 有个问题就是这个lengths怎么更新呢？
        return self._lengths

    def __len__(self):
        return max(self.lengths)
    
    def __iter__(self):
        return iter(self)

class ClickableTextItem(QGraphicsTextItem):
    """
    文字的类, 点击文字之后会发出clicked信号, 并且在鼠标进入时修改文字颜色, 鼠标离开时恢复文字颜色
    """
    clicked = pyqtSignal(bool)
    
    def __init__(self, text):
        super().__init__(text)
        self.setAcceptHoverEvents(True)
    
    def mousePressEvent(self, event):
        self.clicked.emit(True)
        super().mousePressEvent(event)
        
    def hoverEnterEvent(self, event):
        self.setDefaultTextColor(QColor(255, 0, 0))  # 鼠标进入时修改文字颜色
        self.setCursor(Qt.PointingHandCursor) # 鼠标进入时修改鼠标样式
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        self.setDefaultTextColor(QColor(0, 0, 0))  # 鼠标离开时恢复文字颜色
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(event)

class SignalProxy(QObject): 
    # 信号代理, 因为有些类没有clicked信号, 所以需要自己定义一个
    clicked = pyqtSignal(bool)

class LineItem(QGraphicsItem):
    """
    线条的类, 可以设置线条的颜色, 线条的起始点和终止点, 流动的动画效果
    """
    clicked = pyqtSignal(bool)

    def __init__(self, x1, y1, x2, y2, state):
        super().__init__()
        self.setAcceptHoverEvents(True)
        self.signal_proxy = SignalProxy()

        if state == 'enable':
            brush = QBrush(QColor(169, 208, 142))
        elif state == 'disable':
            brush = QBrush(QColor(252, 142, 142))
        elif state == 'warn':
            brush = QBrush(QColor(255,242,204))
        else:
            brush = QBrush(QColor(191, 191, 191))

        self.lineColor = Qt.green
        if state == 'standby' or state == 'enable':
            self.backgroundColor = QColor(191, 191, 191)
        elif state == 'warn':
            self.backgroundColor = QColor(255,242,204)
        elif state == 'disable':
            self.backgroundColor = QColor(252, 142, 142)
        # self.backgroundColor = (Qt.gray if state != 'disable' else (Qt.yellow if state == 'warn' else QColor(255, 128, 128)))
        self.lineWidth = 5
        self.bgLineWidth = 5
        self.capStyle = Qt.RoundCap
        self.startPos = QPointF(x1, y1)
        self.endPos = QPointF(x2, y2)
        self.m_offset = 0

        self.animated = (state == 'enable')

        if self.animated: 
            # 如果是流动的线条, 就启动定时器, 定时刷新界面
            self.timer = QTimer()
            self.timer.timeout.connect(self.updateValue)
            self.timer.start(25)
    
    def mousePressEvent(self, event):
        self.signal_proxy.clicked.emit(True)
        super().mousePressEvent(event)
    
    def hoverEnterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(event)

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(self.backgroundColor)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setStyle(Qt.SolidLine)
        pen.setWidth(self.bgLineWidth)
        pen.setCapStyle(self.capStyle)
        painter.setPen(pen)
        painter.drawLine(self.startPos, self.endPos)

        if self.animated: # 这里是流动的线条
            pen1 = QPen(self.lineColor)
            pen1.setWidth(self.lineWidth)
            pen1.setJoinStyle(Qt.RoundJoin)
            pen1.setCapStyle(self.capStyle)
            pen1.setDashPattern([3, 2, 3, 2])
            pen1.setDashOffset(self.m_offset)
            painter.setPen(pen1)
            painter.drawLine(self.startPos, self.endPos)

    def boundingRect(self):
        return self.shape().boundingRect()

    def shape(self):
        path = QPainterPath()
        path.addRect(self.startPos.x(), self.startPos.y(), self.endPos.x() - self.startPos.x(), self.endPos.y() - self.startPos.y())
        return path

    def updateValue(self):
        if self.animated:
            self.m_offset -= 0.5 # 减就是向右边流动, 加就是向左边流动
            self.update()

class dbItem(QGraphicsItem):
    """
    画出右下角的那个跳动的那个图
    """
    def __init__(self, x1, y1, x2, y2): # 左上右下
        super().__init__()

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        self.Pos2 = QPointF(x1, y1)
        self.Pos4 = QPointF(x2, y2)
        self.m_offset = 0
        # 生成跳动的数据（可能没生成好，如果有更好的数据可以改这个地方）
        self.init_data()

        self.colorMin = QColor(Qt.green)
        self.colorMax = QColor(Qt.red)

        self.timer = QTimer()
        self.timer.timeout.connect(self.updateValue)
        self.timer.start(25)
    
    def init_data(self):
        x = np.linspace(0, np.pi*6, 40)
        self.data = np.sin(x+np.random.random()*np.pi/4)
        for i in range(40):
            self.data[i] = max(self.data[i], 0) / 2.5 + 0.5 + random.random()*0.1

    def paint(self, painter, option, widget):
        value = self.data[int(self.m_offset) % len(self.data)]
        painter.setPen(Qt.NoPen)  # 设置画笔为无画笔
        # 绿条
        point1 = QPointF(self.x1, (self.y2-self.y1)*0.40+self.y1)
        point2 = QPointF(self.x2, self.y2)
        rect = QRectF(point1, point2)
        painter.setBrush(QBrush(QColor(169, 208, 142)))
        painter.drawRect(rect)
        if value > 0.625:
            # 黄条
            point1 = QPointF(self.x1, (self.y2-self.y1)*0.15+self.y1)
            point2 = QPointF(self.x2, (self.y2-self.y1)*0.4+self.y1)
            rect = QRectF(point1, point2)
            painter.setBrush(QBrush(QColor(255, 255, 0)))
            painter.drawRect(rect)
            # 红条
            point1 = QPointF(self.x1, (self.y2-self.y1)*0.15+self.y1 - (self.y2-self.y1)*0.15*(value-0.625)/0.375) 
            point2 = QPointF(self.x2, (self.y2-self.y1)*0.15+self.y1)
            rect = QRectF(point1, point2)
            painter.setBrush(QBrush(QColor(252, 142, 142)))
            painter.drawRect(rect)
        else:
            # 只有黄条
            point1 = QPointF(self.x1, (self.y2-self.y1)*0.4+self.y1 - (self.y2-self.y1)*0.25*value/0.625)
            point2 = QPointF(self.x2, (self.y2-self.y1)*0.4+self.y1)
            rect = QRectF(point1, point2)
            painter.setBrush(QBrush(QColor(255, 255, 0)))
            painter.drawRect(rect)

    def boundingRect(self):
        return QRectF(self.Pos2, self.Pos4)

    def shape(self):
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path

    def updateValue(self):
        self.m_offset += 1
        if self.m_offset == 39:
            # 如果到了最后一个数据，就重新生成数据，因为弹窗的时候sigleview要停止刷新
            # 所以如果要看起来更真就让他重新生成数据，这里如果用其他的也行
            self.init_data()
            self.m_offset = 0
        self.update()

class RectangleItem(QGraphicsPathItem):
    """
    画一个矩形, 可以设置矩形的颜色, 矩形的起始点和终止点, 设置成类主要是为了方便设置鼠标事件
    鼠标进入时修改鼠标样式, 鼠标离开时恢复鼠标样式
    """
    def __init__(self, x1, y1, x2, y2, state):
        super().__init__()
        self.setAcceptHoverEvents(True)
        self.signal_proxy = SignalProxy()

        path = QPainterPath()
        path.addRoundedRect(QRectF(x1, y1, x2 - x1, y2 - y1), 20, 20)
        self.setPath(path)

        if state == 'enable':
            brush = QBrush(QColor(169, 208, 142))
        elif state == 'disable':
            brush = QBrush(QColor(252, 142, 142))
        elif state == 'warn':
            brush = QBrush(QColor(255,242,204))
        else:
            brush = QBrush(QColor(191, 191, 191))
        self.setBrush(brush)

        self.setAcceptHoverEvents(True)
        self.setZValue(0)

    def mousePressEvent(self, event):
        self.signal_proxy.clicked.emit(True)
        super().mousePressEvent(event)

    def hoverEnterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(event)

class SingleView(QGraphicsView):
    """
    单个视图的类, 用于显示一个场景 (主场景)
    输入: id (int) 用于区分不同的场景
    """
    def __init__(self, id):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.scene.setSceneRect(0, 0, 1600, 1100) # 设置场景大小
        # 定义定时器, 用于刷新界面
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateScene) # 连接刷新界面的函数
        self.timer.start(1000)  # 设置刷新时间间隔 1000ms
        self.id = id
        self.cnt = 0
        # 防抖，因为有的时候点击的时候会触发很多次，所以设置一个时间间隔
        # 一个时间间隔里面最多判定一次点击
        self.last_click_time = datetime.now()
        # 设置弹窗是否已经显示，显示的时候就不刷新界面
        # 因为如果刷新了，刷新的时候会删除已经画出来的所有的类
        # 当然也包括了那个弹窗，但是弹窗已经弹出来了，需要手动关，如果刷新
        # 后台删除了这个弹窗，但是前台还在显示，就接受不到传来的信息，会报错
        self.popup_shown = False 
        # 读取数据
        self.init_data()
        self.save_data()
        # 判断是不是修改了数据，如果修改了就先保存数据，然后再读取数据
        # 这样就可以保证数据是最新的（用于多个SigleView之间的同步）
        self.update_flag = False
        # 定义 reset按钮
        self.reset_button = QPushButton('复位此间', self)
        self.reset_button.setGeometry(10, 50, 90, 30) 
        
        self.reset_button.clicked.connect(lambda : self.reset_EdgeAndNode())

        self.reset_button2 = QPushButton('复位所有', self)
        self.reset_button2.setGeometry(130, 50, 90, 30)

        def give_reset_sig():
            for i in range(len(reset_sig)):
                reset_sig[i] = True
        # 先给出复位信号，然后再复位
        # 因为这里需要给的是所有的SigleView，所以需要在main里面定义一个reset_sig全局变量
        self.reset_button2.clicked.connect(lambda : give_reset_sig()) 

    def reset_all(self):
        for i in range(len(self.edge)):
            self.edge[i] = 0
            self.edge_warning[i] = 0
        for i in range(len(self.node)):
            self.node[i] = 0
            # node_warning没有用了（第二次意见的时候），但是加在这里问题不大
            # 后面没有用到node_warning了，所以这里也不用管，没删了
            self.node_warning[i] = 0
        reset_sig[self.id] = False
        self.update_flag = True
    
    def reset_EdgeAndNode(self):
        for i in range(len(self.edge)):
            self.edge[i] = 0
            self.edge_warning[i] = 0
            self.update_flag = True
        # 单独把node 012 拿出来，因为012是自己的，3~后面是公共的
        self.node[0] = 0
        self.node[1] = 0
        self.node[2] = 0
        self.node_warning[0] = 0
        self.node_warning[1] = 0
        self.node_warning[2] = 0

    def read_data(self):
        data = np.load('data/data'+str(self.id)+'.npz', allow_pickle=True)
        self.edge = data['edge']
        tempnode = data['node']
        self.node = np.zeros((9,))
        self.node[0] = tempnode[0]
        self.node[1] = tempnode[1]
        self.node[2] = tempnode[2]
        temp = read_global_node()
        for i in range(6):
            self.node[i+3] = temp[i]
        # 警告数据部分
        self.edge_warning = data['edge_warning']
        tempnode = data['node_warning']
        self.node_warning = np.zeros((9,))
        self.node_warning[0] = tempnode[0]
        self.node_warning[1] = tempnode[1]
        self.node_warning[2] = tempnode[2]
        temp = read_global_node_warning()
        for i in range(6):
            self.node_warning[i+3] = temp[i]
    
    def save_data(self):
        self.update_flag = False
        save_global_node(self.node)
        save_global_node_warning(self.node_warning)
        # 判断文件夹是否存在，不存在就创建
        if not os.path.exists('data'):
            os.mkdir('data')
        np.savez(
            'data/data'+str(self.id)+'.npz',
            edge=self.edge,
            node=np.array([self.node[0], self.node[1], self.node[2]]),
            edge_warning=self.edge_warning,
            node_warning=np.array([self.node_warning[0], self.node_warning[1], self.node_warning[2]])
        )

    def init_data(self):
        if os.path.exists('data/data'+str(self.id)+'.npz'):
            data = np.load('data/data'+str(self.id)+'.npz', allow_pickle=True)
            self.edge = data['edge']
            tempnode = data['node']
            self.node = np.zeros((9,))
            self.node[0] = tempnode[0]
            self.node[1] = tempnode[1]
            self.node[2] = tempnode[2]
            temp = read_global_node()
            for i in range(6):
                self.node[i+3] = temp[i]
            # 警告数据部分
            self.edge_warning = data['edge_warning']
            tempnode = data['node_warning']
            self.node_warning = np.zeros((9,))
            self.node_warning[0] = tempnode[0]
            self.node_warning[1] = tempnode[1]
            self.node_warning[2] = tempnode[2]
            temp = read_global_node_warning()
            for i in range(6):
                self.node_warning[i+3] = temp[i]
        else:
            self.edge = np.zeros((28,))
            self.node = np.zeros((9,))
            temp = read_global_node()
            for i in range(6):
                self.node[i+3] = temp[i]
            self.edge_warning = np.zeros((28,))
            self.node_warning = np.zeros((9,))
            temp = read_global_node_warning()
            for i in range(6):
                self.node_warning[i+3] = temp[i]

    def drawRect(self, x1, y1, x2, y2, Str, state, idx):
        """
        state:
            enable : 正在使用这个 (绿色)
            disable : 这个用不了 (红色)
            standby : 待机 (灰色)
            warn : 警告 (黄色)
        """
        if self.node_warning[idx] == 1:
            state = 'warn'
        # 画出矩形
        path_item = RectangleItem(x1, y1, x2, y2, state)
        self.scene.addItem(path_item)
        path_item.signal_proxy.clicked.connect(lambda clicked: self.handleClick(clicked, 'node', idx))
        # 画出文字
        text = ClickableTextItem(Str)
        text.setDefaultTextColor(QColor(0, 0, 0))
        font = text.font()
        font.setPointSize(12)
        text.setFont(font)
        text.setPos((x1+x2)/2 - text.boundingRect().width()/2, (y1+y2)/2 - text.boundingRect().height()/2)
        self.scene.addItem(text)
        # 画出图形
        if state == 'warn':
            pixmap = QPixmap('warn.png')
            w = pixmap.width()
            h = pixmap.height()
            scaled_pixmap = pixmap.scaled(w/40, h/40, Qt.AspectRatioMode.KeepAspectRatio)
            item = self.scene.addPixmap(scaled_pixmap)
            text_x = (x1+x2)/2 - text.boundingRect().width()/2
            text_y = (y1+y2)/2 - text.boundingRect().height()/2
            item.setPos(text_x, text_y-40)
        # 为了避免弹框出现两次，所以注释掉了下面这一行
        # text.clicked.connect(lambda clicked: self.handleClick(clicked, 'node', idx))
    
    def drawLine(self, x1, y1, x2, y2, state, Str, idx):
        """
        state:
            enable : 正在使用这个 (绿色流动 + 灰底)
            disable : 这个用不了 (红色)
            standby : 待机 (灰色)
        """
        if self.edge_warning[idx] == 1:
            state = 'warn'
        Line = LineItem(x1, y1, x2, y2, state)
        self.scene.addItem(Line)
        Line.signal_proxy.clicked.connect(lambda clicked: self.handleClick(clicked, 'edge', idx))
        if Str == None:
            return
        text = ClickableTextItem(Str)
        text.setDefaultTextColor(QColor(0, 0, 0))
        font = text.font()
        font.setPointSize(12)
        text.setFont(font)
        text.setPos((x1+x2)/2 - text.boundingRect().width()/2, (y1+y2)/2 - text.boundingRect().height()/2 - 23)
        self.scene.addItem(text)
        text.clicked.connect(lambda clicked: self.handleClick(clicked, 'edge', idx))
        # 画出图形
        if state == 'warn':
            pixmap = QPixmap('warn.png') # png是透明格式的
            w = pixmap.width()
            h = pixmap.height()
            scaled_pixmap = pixmap.scaled(w/40, h/40, Qt.AspectRatioMode.KeepAspectRatio)
            item = self.scene.addPixmap(scaled_pixmap)
            text_x = (x1+x2)/2 - text.boundingRect().width()/2
            text_y = (y1+y2)/2 - text.boundingRect().height()/2
            item.setPos(text_x-60, text_y - 10)
        
    def drawdefault(self):
        # 直播室和融合矩阵
        self.drawRect(65, 100, 220, 490, '{}号直播室'.format(self.id+1), self.state_node[0], 0)
        self.drawRect(400, 100, 650, 490, 'AOIP/STUDER融合矩阵', self.state_node[3], 3)
        names = ['MADI', 'Dante主', 'Dante备', 'AES', '垫乐主', '垫乐备']
        for i in range(6):
            self.drawLine(225, 120 + i*50, 395, 120 + i*50, self.state_edge[i], names[i], i)
        # 音频处理器和融合矩阵
        self.drawRect(830, 100, 1060, 230, '音频处理器(网络)', self.state_node[1], 1)
        self.drawRect(830, 250, 1060, 380, '音频处理器(数字)', self.state_node[2], 2)
        names = ['网络', 'AES', '模拟']
        for i in range(3):
            self.drawLine(655, 120 + i*46, 825, 120 + i*46, self.state_edge[14+i], names[i], 14+i)
        names = ['AES', '模拟']
        for i in range(2):
            self.drawLine(655, 290 + i*46, 825, 290 + i*46, self.state_edge[17+i], names[i], 17+i)
        # 音频处理器和节传
        names = ['网络', 'AES', '模拟']
        for i in range(3):
            self.drawLine(1065, 120 + i*46, 1195, 120 + i*46, self.state_edge[22+i], names[i], 22+i)
        names = ['AES', '模拟']
        for i in range(2):
            self.drawLine(1065, 290 + i*46, 1195, 290 + i*46, self.state_edge[25+i], names[i], 25+i)
        # 融合矩阵和节传
        self.drawLine(655, 420, 1195, 420, self.state_edge[19], 'AES直连1', 19)
        self.drawLine(655, 470, 1195, 470, self.state_edge[20], 'AES直连2', 20)
        # 应急矩阵和直播室
        self.drawRect(400, 550, 650, 640, '应急矩阵', self.state_node[4], 4)
        self.drawLine(225, 120 + 7*50, 320, 120 + 7*50, self.state_edge[7], None, 7)
        self.drawLine(320, 120 + 7*50, 320, 600, self.state_edge[7], None, 7)
        self.drawLine(320, 600, 395, 600, self.state_edge[7], 'AES', 7)
        # 应急矩阵和节传
        self.drawLine(655, 600, 1195, 600, self.state_edge[21], None, 21)
        # 外传信号到节传
        self.drawRect(65, 670, 220, 910, '外接信号', self.state_node[5], 5)
        names = ['CNR2', '有线电视', '网络推流', '微波']
        for i in range(4):
            self.drawLine(225, 740 + i*50, 1195, 740 + i*50, self.state_edge[i+8], names[i], i+8)
        # 外界信号到融合矩阵
        self.drawLine(225, 690, 270, 690, self.state_edge[6], None, 6)
        self.drawLine(270, 690, 270, 420, self.state_edge[6], None, 6)
        self.drawLine(270, 420, 395, 420, self.state_edge[6], 'CNR1', 6)
        # 本地信号和节传
        self.drawRect(65, 950, 220, 1050, '本地信号', self.state_node[6], 6)
        self.drawLine(225, 970, 1195, 970, self.state_edge[12], '本地卫星CNR', 12)
        self.drawLine(225, 970+46, 1195, 970+46, self.state_edge[13], '本地垫乐', 13)
        self.drawRect(1200, 100, 1320, 1050, '节传', self.state_node[7], 7)
        # 节传和发射
        self.drawRect(1430, 540, 1520, 610, '发射', self.state_node[8], 8)
        self.drawLine(1325, 575, 1425, 575, self.state_edge[27], None, 27)
        # 发射塔图片
        pixmap = QPixmap('Launch Tower.png')
        w = pixmap.width()
        h = pixmap.height()
        scaled_pixmap = pixmap.scaled(w/4, h/4, Qt.AspectRatioMode.KeepAspectRatio)
        item = self.scene.addPixmap(scaled_pixmap)
        item.setPos(1437, 420)
        # db
        if self.state_node[8] == 'enable':
            Item = dbItem(1435, 660, 1520, 1050)
            self.scene.addItem(Item)
    
    def handleClick(self, clicked, type, idx):
        # 就是所有的Click信号都连接到这个函数
        # type区分了到底是node还是edge
        # idx是对应的编号
        if clicked and datetime.now() - self.last_click_time > timedelta(seconds=0.2): # 防抖，0.2秒最多点击一次
            self.update_flag = True
            self.last_click_time = datetime.now()
            if type == 'node':
                self.popup_shown = True # 表示弹窗已经显示了，不要刷新界面
                if self.node[idx] == 1 or self.node_warning[idx] == 1:
                    msg_box = QMessageBox()
                    msg_box.setWindowTitle("选项")
                    msg_box.setText("请选择设备是否恢复正常")
                    msg_box.setIcon(QMessageBox.Question)
                    msg_box.addButton("是", QMessageBox.YesRole)
                    msg_box.addButton("否", QMessageBox.NoRole)
                    clicked_button = msg_box.exec_()  # 监听选项框的按钮点击事件
                    if clicked_button == 0:
                        self.node[idx] = 0
                        self.node_warning[idx] = 0
                    self.popup_shown = False # 最后设置为False，表示弹窗已经关闭了，可以刷新界面了
                else:
                    msg_box = QMessageBox()
                    msg_box.setWindowTitle("选项")
                    msg_box.setText("请选择设备是否断开")
                    msg_box.setIcon(QMessageBox.Question)
                    msg_box.addButton("是", QMessageBox.YesRole)
                    msg_box.addButton("否", QMessageBox.NoRole)
                    clicked_button = msg_box.exec_()  # 监听选项框的按钮点击事件
                    if clicked_button == 0:
                        self.node[idx] = 1
                    if clicked_button == 1:
                        self.node[idx] = 0
                    self.popup_shown = False
            else:
                self.popup_shown = True
                if self.edge[idx] == 1 or self.edge_warning[idx] == 1:
                    msg_box = QMessageBox()
                    msg_box.setWindowTitle("选项")
                    msg_box.setText("请选择链路是否恢复正常")
                    msg_box.setIcon(QMessageBox.Question)
                    msg_box.addButton("是", QMessageBox.YesRole)
                    msg_box.addButton("否", QMessageBox.NoRole)
                    clicked_button = msg_box.exec_()  # 监听选项框的按钮点击事件
                    if clicked_button == 0:
                        self.edge[idx] = 0
                        self.edge_warning[idx] = 0
                    self.popup_shown = False
                else:
                    msg_box = QMessageBox()
                    msg_box.setWindowTitle("选项")
                    msg_box.setText("请选择链路异常状态")
                    msg_box.setIcon(QMessageBox.Question)
                    msg_box.addButton("信号差", QMessageBox.YesRole)
                    msg_box.addButton("已断开", QMessageBox.NoRole)
                    clicked_button = msg_box.exec_()  # 监听选项框的按钮点击事件
                    if clicked_button == 0:
                        self.edge_warning[idx] = 1
                        self.edge[idx] = 1
                    if clicked_button == 1:
                        self.edge[idx] = 1
                    self.popup_shown = False

    def judge1(self): # 判断信号能不能从直播室出去 (不走7那条线)
        if self.node[0] == 1:
            return False
        if (self.edge[0]+self.edge[1]+self.edge[2]+self.edge[3]+self.edge[4]+self.edge[5]) == 6:
            return False
        if self.node[3] == 1:
            return False
        if self.edge[19] + self.edge[20] == 2:
            if self.edge[14]+self.edge[15]+self.edge[16]==3 or self.edge[22]+self.edge[23]+self.edge[24]==3 or self.node[1]==1:
                if self.edge[17]+self.edge[18]==2 or self.edge[25]+self.edge[26]==2 or self.node[2]==1:
                    return False
        return True
    
    def judge2(self): # 判断信号能不能从直播室出去 (包括7的那条线)
        if self.node[0] == 1:
            return False
        if self.edge[7] == 1 or self.edge[21] == 1 or self.node[4] == 1:
            if self.node[0] == 1:
                return False
            if (self.edge[0]+self.edge[1]+self.edge[2]+self.edge[3]+self.edge[4]+self.edge[5]) == 6:
                return False
            if self.node[3] == 1:
                return False
            if self.edge[19] + self.edge[20] == 2:
                if self.edge[14]+self.edge[15]+self.edge[16]==3 or self.edge[22]+self.edge[23]+self.edge[24]==3 or self.node[1]==1:
                    if self.edge[17]+self.edge[18]==2 or self.edge[25]+self.edge[26]==2 or self.node[2]==1:
                        return False
        return True
    
    def judge3(self): # 判断信号能不能从6出去
        if self.node[3] == 1:
            return False
        if self.edge[19] + self.edge[20] == 2:
            if self.edge[14]+self.edge[15]+self.edge[16]==3 or self.edge[22]+self.edge[23]+self.edge[24]==3 or self.node[1]==1:
                if self.edge[17]+self.edge[18]==2 or self.edge[25]+self.edge[26]==2 or self.node[2]==1:
                    return False
        return True

    def logic_calc(self):
        self.state_edge = ['standby' for _ in range(28)]
        self.state_node = ['standby' for _ in range(9)]
        for i in range(9):
            if self.node[i] == 1:
                self.state_node[i] = 'disable'
        for i in range(28):
            if self.edge[i] == 1:
                self.state_edge[i] = 'disable'

        if self.judge2():
            self.state_node[0] = 'enable'
            if self.judge1():
                # 到融合矩阵
                for i in range(8):
                    if i == 6 or self.edge[i] == 1:
                        continue
                    self.state_edge[i] = 'enable'
                    break
                if self.state_edge[7] == 'enable':
                    self.state_node[4] = 'enable'
                    self.state_edge[21] = 'enable'
                else:
                    self.state_node[3] = 'enable'
                    # 到音频处理器
                    if self.node[1] == 0 and self.edge[22]+self.edge[23]+self.edge[24]!=3 and self.edge[14]+self.edge[15]+self.edge[16]!=3:
                        self.state_node[1] = 'enable'
                        for i in range(14, 17):
                            if self.edge[i] == 0:
                                self.state_edge[i] = 'enable'
                                break
                        for i in range(22, 25):
                            if self.edge[i] == 0:
                                self.state_edge[i] = 'enable'
                                break
                    elif self.node[2]==0 and self.edge[17]+self.edge[18]!=2 and self.edge[25]+self.edge[26]!=2:
                        self.state_node[2] = 'enable'
                        for i in range(17, 19):
                            if self.edge[i] == 0:
                                self.state_edge[i] = 'enable'
                                break
                        for i in range(25, 27):
                            if self.edge[i] == 0:
                                self.state_edge[i] = 'enable'
                                break
                    else:
                        if self.edge[19] == 0:
                            self.state_edge[19] = 'enable'
                        else:
                            self.state_edge[20] = 'enable'
            else:
                self.state_edge[7] = 'enable'
                self.state_edge[21] = 'enable'
                self.state_node[4] = 'enable'

        elif self.node[5] == 0 and self.edge[8]+self.edge[9]+self.edge[10]+self.edge[11]!=4: # 外接信号
            self.state_node[5] = 'enable'
            if self.judge3() and self.edge[6]==0: # 走上面
                self.state_edge[6] = 'enable'
                self.state_node[3] = 'enable'
                # ----------------- 3到音频处理器 (和上面重复的) --------------
                if self.node[1] == 0 and self.edge[22]+self.edge[23]+self.edge[24]!=3 and self.edge[14]+self.edge[15]+self.edge[16]!=3:
                    self.state_node[1] = 'enable'
                    for i in range(14, 17):
                        if self.edge[i] == 0:
                            self.state_edge[i] = 'enable'
                            break
                    for i in range(22, 25):
                        if self.edge[i] == 0:
                            self.state_edge[i] = 'enable'
                            break
                elif self.node[2]==0 and self.edge[17]+self.edge[18]!=2 and self.edge[25]+self.edge[26]!=2:
                    self.state_node[2] = 'enable'
                    for i in range(17, 19):
                        if self.edge[i] == 0:
                            self.state_edge[i] = 'enable'
                            break
                    for i in range(25, 27):
                        if self.edge[i] == 0:
                            self.state_edge[i] = 'enable'
                            break
                else:
                    if self.edge[19] == 0:
                        self.state_edge[19] = 'enable'
                    else:
                        self.state_edge[20] = 'enable'
                # ----------------- 3到音频处理器 (和上面重复的) --------------
            else:
                for i in range(8, 12):
                    if self.edge[i] == 0:
                        self.state_edge[i] = 'enable'
                        break
        elif self.node[6] == 0 and self.edge[12]+self.edge[13]!=2:
            self.state_node[6] = 'enable'
            if self.edge[12] == 0:
                self.state_edge[12] = 'enable'
            else:
                self.state_edge[13] = 'enable'
        else:
            return
        # 不通的情况
        if self.node[7] == 1 or self.node[8] == 1:
            for i in range(28):
                if self.state_edge[i] == 'enable':
                    self.state_edge[i] = 'standby'
            for i in range(9):
                if self.state_node[i] == 'enable':
                    self.state_node[i] = 'standby'
            return
        self.state_node[7] = 'enable'
        self.state_node[8] = 'enable'
        self.state_edge[27] = 'enable'

    def updateTextColors(self):
        # 为了解决刷新时文字颜色恢复的问题
        # 因为刷新时会重新画一遍，所以文字颜色也会恢复，鼠标在上面也没用
        for item in self.scene.items():
            if isinstance(item, ClickableTextItem):
                if item.isUnderMouse():
                    item.setDefaultTextColor(QColor(255, 0, 0))  # 鼠标悬停时修改文字颜色
                else:
                    item.setDefaultTextColor(QColor(0, 0, 0))  # 恢复默认文字颜色

    def updateScene(self):
        if self.popup_shown: # 如果弹窗正在显示，就不要刷新界面了
            return
        self.scene.clear() # 清除掉所有的图形，重画
        if reset_sig[self.id]:
            self.reset_all()
        if self.update_flag:
            self.save_data()
        self.read_data() # 最后读数据保证了如果修改了global的数据，其他的SingleView也能更新到
        self.logic_calc()
        self.drawdefault()
        self.updateTextColors()
    
    # 下面是一个调试用的小函数
    # 这个函数取消注释之后，鼠标点击某一个位置，会在界面上显示一个红色的小方块
    # 然后在终端会显示出鼠标点击的坐标
    # 但是我这里全屏显示了，所以鼠标点击的坐标和实际的坐标不一样（貌似.....）
    # 所以要用的话首先关掉全屏，这个函数在画图的时候很有用，因为可以知道画在哪里, 超级方便

    # def mousePressEvent(self, event):
    #     pos = event.pos() # 获取鼠标点击坐标
    #     x = pos.x()
    #     y = pos.y()
    #     print(f"Mouse clicked at: ({x}, {y})") # 输出鼠标点击坐标
    #     # 在界面上显示点击位置的可视化
    #     rect = self.scene.addRect(x - 5, y - 5, 10, 10, QPen(Qt.NoPen), QBrush(Qt.red)) 
    #     rect.setOpacity(0.5)
    #     super().mousePressEvent(event)

def read_global_node():
    if os.path.exists('data/global_node.npz'):
        return np.load('data/global_node.npz', allow_pickle=True)['value']
    else:
        return np.zeros((6,))

def read_global_node_warning():
    if os.path.exists('data/global_node_warning.npz'):
        return np.load('data/global_node_warning.npz', allow_pickle=True)['value']
    else:
        return np.zeros((6,))

def save_global_node_warning(array):
    temp = []
    for i in range(6):
        temp.append(array[i+3])
    temp = np.array(temp)
    if not os.path.exists('data'):
        os.mkdir('data')
    np.savez('data/global_node_warning.npz', value=temp)

def save_global_node(array):
    temp = []
    for i in range(6):
        temp.append(array[i+3])
    temp = np.array(temp)
    if not os.path.exists('data'):
        os.mkdir('data')
    np.savez('data/global_node.npz', value=temp)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 设置导播室的个数
    view_num = 3
    # 设置清零信号
    reset_sig = [False for _ in range(view_num)]
    # 创建视图对象
    views = [SingleView(i) for i in range(view_num)]
    current_view_index = [0]
    # 设置视图的尺寸
    for view in views:
        view.resize(1600, 1100)
        # 移动到屏幕中央 (但是后面都全屏显示了，所以这个也没什么用了)
        center_point = QDesktopWidget().availableGeometry().center()
        x = center_point.x()
        y = center_point.y()
        view.move(x - view.width() // 2, y - view.height() // 2)
        view.hide()  # 隐藏所有视图
        view.setWindowTitle('直播室链路选择')
    # 添加按钮
    for i in range(view_num):
        for j in range(view_num):
            button = QPushButton("{}号直播室".format(j+1), views[i])
            if i == j:
                # 就是设置当前视图的按钮颜色和其他不一样就行
                button.setStyleSheet("background-color: rgb(173, 216, 230);")
            button.move(10 + 120 * j, 10)
            def create_lambda_func(j):
                return lambda: switch_view(j)
            button.clicked.connect(create_lambda_func(j))
    def switch_view(index):
        views[current_view_index[0]].hide() # 隐藏当前视图
        current_view_index[0] = index       # 更新当前视图索引
        views[current_view_index[0]].showMaximized()  # 这个是全屏显示的意思，如果想不全屏就用下面这行代码，注释掉这行，这里需要改，往下三行也需要改
        # views[current_view_index[0]].show()
    # 显示第一个视图 (默认的)
    views[current_view_index[0]].showMaximized() # 这个是全屏显示的意思，如果想不全屏就用下面这行代码，注释掉这行，这里需要改，往上三行也需要改
    # views[current_view_index[0]].show()
    sys.exit(app.exec_())

"""
整个代码的思路就是：
创建视图对象 SingleView(封装的很好)
然后动画效果通过定时器, 时间到了就会发出timeout信号, 告诉程序需要刷新界面了
每次刷新界面都全部重新画一次, 清除掉之前的所有图形
然后因为要重新画嘛, 所以逻辑也需要重新计算一次
关于多个直播室之间的信息共享, 就是通过读写文件来实现的
只要做到了如果修改了就标记一下, 然后保存, 如果没有修改就不管
修改在前面, 读数据在后面, 就能做到顺序不出错的读写了
如果要实现多个电脑之间的同步, 修改savez函数和load的位置就行了(多个电脑应该会有一个共同的位置把, 这个不太懂)

# 用到的库不多, 应该只有numpy和PyQt5
# 打包应用程序还需要pyinstaller, 要pip安装了才能打包
打包直接运行"打包.bat"就行
"""