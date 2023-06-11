'''
Created: 2023-05-31
Revised: 2023-06-06
'''



import gym
import torch
from transformers import AutoTokenizer
import random
import numpy as np
import torch.nn.functional as F
from collections import deque
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge 



class MyEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, args):
        super().__init__()
        self.df = df
        # self.q = self.df['问题']
        self.a = self.df['答案']
        # self.index_to_sample = self.df.index
        self.sampled_index = []
        self.state, reward, self.done = self.reset()
        self.maxlen_of_deque = args.maxlen_of_deque
        self.reward_threshold = args.reward_threshold
        self.reward_deque = deque(maxlen=self.maxlen_of_deque)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        

    def step(self, action):
        step_reward = self._get_reward(action)
        self.state = self._get_state()
        self.reward_deque.append(step_reward)
        
        if len(self.reward_deque) == self.maxlen_of_deque and np.mean(self.reward_deque) > self.reward_threshold:
            self.done = True
        info = {}

        return self.state, step_reward, self.done, info

    def reset(self):
        self.index_to_sample = list(self.df.index)
        self.sampled_index = []
        self.state = self._get_state()
        reward = 0
        self.done = False
        return self.state, reward, self.done

    def _get_state(self):
        self.index_to_sample = [item for item in self.index_to_sample if item not in self.sampled_index]
        self.state = random.sample(self.index_to_sample,1)
        self.sampled_index.append(self.state)
        return self.state[0]

    def _get_reward(self,action):
        # input_ids = self.tokenizer.encode(self.a[self.state], return_tensors="pt")
        # answer_decode = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        # reward = sentence_bleu(action, self.a[self.state])
        '''
        rough score: 
        
        rewards = Rouge.get_scores(action, answer_decode)
        reward = rewards[0]['rouge-l']['f']
        '''
        
        # answer_encode_ = torch.tensor(answer_encode)
        # action = action.float()
        
        '''
        目前是两个str格式的中文语句
        '''
        reward = float(F.cosine_similarity(action, self.a[self.state]))
        return reward