'''
Created: 2023-05-31
Revised: 2023-06-06
'''



import os
import errno
import json
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agent import agent, train_on_policy_agent
from env import MyEnv



def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise



def load_data(args):
    qa = pd.read_excel(os.path.join(os.getcwd(), args.file_name))
    q = list(qa['问题'])
    a = list(qa['答案'])
    return qa, q, a



class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(str(msg) + '\n')
        self.log_file.flush()
        print(msg)



def parse_args():
    parser = argparse.ArgumentParser()
    # Environment parameters
    parser.add_argument('--maxlen_of_deque', type=int, default=10, help='max length of reward deque')
    parser.add_argument('--reward_threshold', type=float, default=0.99, help='the threshold value of reward')
    parser.add_argument('--file_name', type=str, default='qa_disease.xlsx', help='data file whole name')
    parser.add_argument('--ckpt_root', type=str, default='../checkpoints')
    
    # Agent parameters
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_episodes', type=int, default=500)
    
    # GPT parameters
    parser.add_argument('--engine', type=str, default='text-davinci-002', choices=['text-davinci-002', 'ada'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    args = parser.parse_args()

    # print and save the args
    args.ckpt_path = os.path.join(args.ckpt_root)
    create_dir(args.ckpt_path)
    _logger = Logger(args.ckpt_path + '/args.txt')

    print('====Input Arguments====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args



def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))



if __name__ == '__main__':
    args = parse_args()
    qa, q, a = load_data(args)
    df = qa
    env = MyEnv(df,args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rlagent = agent(df, args, device)
    return_list, trained_agent = train_on_policy_agent(env, rlagent, args)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('results')
    plt.tight_layout()
    plt.savefig("results")
    
    mv_return = moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('smooth results')
    plt.tight_layout()
    plt.savefig("smooth results")