'''
Created: 2023-05-31
Revised: 2023-06-06
'''



import gym
import torch
import transformers
import numpy as np
from tqdm import tqdm



class agent:
    def __init__(self, df, args, device):
        self.learning_rate = args.learning_rate
        # self.df = df
        self.q = df['问题']
        self.a = df['答案']
        self.device = device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-chinese")
        # self.model = transformers.AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad").to(self.device)
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained("bert-base-chinese")

        # Define the loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def take_action(self,state):
        question = self.q[state]
        '''
        answer_encode = self.tokenizer.encode(self.a[state], return_tensors="pt")
        input_ids = self.tokenizer.encode(question, return_tensors="pt")
        # output = self.model.generate(input_ids, max_length=100, pad_token_id=self.tokenizer.eos_token_id, 
        #                         do_sample=True, temperature=0.7).detach().cpu().numpy()[0]
        # action = self.tokenizer.decode(output, skip_special_tokens=True)
        answer_len = len(answer_encode[0])
        action = self.model.generate(input_ids, max_length=answer_len,
                                     pad_token_id=self.tokenizer.eos_token_id, do_sample=True, temperature=0.7)
        # action = torch.tensor(action).to(self.device)
        '''
        
        
        
        inputs = self.tokenizer(question, return_tensors="pt")
        start_scores = self.model(**inputs)[0]
        end_scores = self.model(**inputs)[1]
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores) + 1
        # action = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
        action = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
        return action
    
    def update(self, question, reward, done):
        if not done:
            input_ids = self.tokenizer.encode(question, return_tensors="pt").to(self.device)
            question_ = self.model(input_ids, 
                                   labels=torch.tensor([reward] * input_ids.shape[1]).to(self.device))[1]
            self.loss = self.criterion(question_, torch.tensor([reward] * question.shape[0]).to(self.device))
            self.loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()



def train_on_policy_agent(env, agent, args):
    return_list = []
    num_episodes = args.num_episodes
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, reward, done = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(question,reward,done)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    trained_agent = agent
    return return_list, trained_agent