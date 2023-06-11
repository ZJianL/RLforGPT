'''
Created: 2023-04-06
Revised: 2023-05-26
'''



import os
import random
import argparse
import errno
import json
import math
import time
import numpy as np
import pandas as pd
import openai
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import lru_cache
from transformers import AutoTokenizer, AutoModelForTokenClassification



openai.api_key = os.getenv("***")



def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise



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



class policy_network(nn.Module):

    def __init__(self,
                 model_config="bert-base-chinese",
                 add_linear=False,
                 embedding_size=128,
                 freeze_encoder=True) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_config)
        print("model_config:", model_config)
        self.model = AutoModelForTokenClassification.from_pretrained(model_config)

        # Freeze transformer encoder and only train the linear layer
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        if add_linear:
            # Add an additional small, adjustable linear layer on top of BERT tuned through RL
            self.embedding_size = embedding_size
            self.linear = nn.Linear(self.model.config.hidden_size,
                                    embedding_size)  # 768 for bert-base-uncased, distilbert-base-uncased
        else:
            self.linear = None

    def forward(self, input_list):
        input = self.tokenizer(input_list, truncation=True, padding=True,return_tensors="pt").to(self.model.device)
        output = self.model(**input, output_hidden_states=True)
        last_hidden_states = output.hidden_states[-1]
        sentence_embedding = last_hidden_states[:, 0, :]  # len(input_list) x hidden_size

        if self.linear:
            sentence_embedding = self.linear(sentence_embedding)  # len(input_list) x embedding_size

        return sentence_embedding
    

    
def load_data(args):
    qa = pd.read_excel(os.path.join(os.getcwd(), args.file_name))
    q = list(qa['问题'])
    a = list(qa['答案'])
    return qa, q, a



def get_gpt3_output(question, args):
    return call_gpt3(args.engine, question, args.temperature, args.max_tokens, args.top_p, args.frequency_penalty,
                     args.presence_penalty)



@lru_cache(maxsize=10000)
def call_gpt3(engine, question, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    patience = 100
    while True:
        try:
            response = openai.Completion.create(engine=engine,
                                                question=question,
                                                temperature=temperature,
                                                max_tokens=max_tokens,
                                                top_p=top_p,
                                                frequency_penalty=frequency_penalty,
                                                presence_penalty=presence_penalty,
                                                stop=["\n"])
            output = response["choices"][0]["text"].strip()
            break
        except Exception as e:
            patience -= 1
            if not patience:
                print("Running out of patience waiting for OpenAI")
            else:
                time.sleep(0.1)
    return output



def get_batch_reward_loss(policy_model, train_batch, q, a, args):
    batch_loss = 0
    batch_reward = 0
    
    ## loop over the training examples
    for i in train_batch:
        question = q[i]
        output = get_gpt3_output(question, args)
        embedding_gpt = policy_model(output)
        embedding_a = policy_model(a[i])
        '''
        目前是embedding处理后的torch.tensor
        具体见上面policy_network类的实现
        '''
        reward = F.cosine_similarity(embedding_gpt, embedding_a)
        
        batch_reward += _reward
        batch_loss -= _reward

    return batch_reward, batch_loss



def policy_gradient_train(policy_model, qa, q, a, args):

    
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.lr)
    
    # train_samples = random.sample(index_to_sample = q.index,1000)
    
    train_samples = list(qa.index)
    
    num_batch = math.ceil(len(train_samples) / args.batch_size)

    reward_history = []
    loss_history = []

    total_reward_history = []  # epoch based
    total_loss_history = []  # epoch based

    STOP_FLAG = False

    for epoch in range(args.epochs):
        logger.write(f"Epoch: {epoch}")
        total_train_reward = 0
        total_train_loss = 0
        for batch_i in range(num_batch):
            logger.write(f"Batch: {batch_i}")
            train_batch = train_samples[batch_i * args.batch_size:(batch_i + 1) * args.batch_size]
            
            reward, loss = get_batch_reward_loss(policy_model, train_batch, q, a, args)
            
            logger.write(f"### reward for the batch: {reward}")
            logger.write(f"### loss for the batch: {loss}\n")
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # for each iteration/batch
            total_train_reward += reward
            total_train_loss += loss.item()
            
            reward_history.append(reward)
            loss_history.append(loss.item())
            
            if np.isnan(loss.item()):
                STOP_FLAG = True
                break
                
        # for each epoch
        total_reward_history.append(total_train_reward)
        total_loss_history.append(total_train_loss)

        best_reward = max(total_reward_history)
        best_loss = min(total_loss_history)

        best_reward_epoch = total_reward_history.index(best_reward)
        best_loss_epoch = total_loss_history.index(best_loss)

        logger.write("============================================")
        logger.write(f"### Epoch: {epoch} / {args.epochs}")
        logger.write(f"### Total reward: {total_train_reward}, " + f"Total loss: {round(total_train_loss,5)}, " +
                     f"Best reward: {best_reward} at epoch {best_reward_epoch}, " +
                     f"Best loss: {round(best_loss, 5)} at epoch {best_loss_epoch}\n")

        # save every epoch
        ckpt_file = os.path.join(args.ckpt_path, f"ckpt_{epoch}.pt")
        torch.save(policy_model.linear.state_dict(), ckpt_file)
        logger.write(f"saved the ckpt to {ckpt_file}")

        # save best epoch
        if epoch == best_reward_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_reward.pt")
            torch.save(policy_model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best reward ckpt to {ckpt_file}")

        if epoch == best_loss_epoch:
            ckpt_file = os.path.join(args.ckpt_path, "ckpt_best_loss.pt")
            torch.save(policy_model.linear.state_dict(), ckpt_file)
            logger.write(f"saved the best loss ckpt to {ckpt_file}")

        # save reward and loss history
        history = {
            "reward_history": reward_history,
            "loss_history": loss_history,
            "total_reward_history": total_reward_history,
            "total_loss_history": total_loss_history,
        }
        history_file = os.path.join(args.ckpt_path, "history.json")
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, separators=(',', ': '))

        # print cache info
        logger.write(call_gpt3.cache_info())
        logger.write("============================================\n")

        if STOP_FLAG:
            break

    # save in the end
    ckpt_file = os.path.join(args.ckpt_path, "ckpt_final.pt")
    torch.save(policy_model.linear.state_dict(), ckpt_file)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='qa_disease.xlsx')

    # User options
    parser.add_argument('--seed', type=int, default=19990406, help='random seed')

    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='text-davinci-002', choices=['text-davinci-002', 'ada'])
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)

    # Policy gradient settings
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_config',
                        type=str,
                        default='bert-base-chinese',
                        choices=['distilbert-base-uncased', 'bert-base-uncased', 'bert-base-chinese'])
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of policy network.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--embedding_size', type=int, default=128, help='Policy network final layer hidden state size.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=20,
                        help='Policy network training batch size.')
    parser.add_argument('--ckpt_root', type=str, default='../checkpoints')

    args = parser.parse_args()

    # print and save the args
    args.ckpt_path = os.path.join(args.ckpt_root)
    create_dir(args.ckpt_path)
    _logger = Logger(args.ckpt_path + '/args.txt')

    print('====Input Arguments====')
    _logger.write(json.dumps(vars(args), indent=2, sort_keys=False))

    return args



if __name__ == '__main__':

    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)  # CPU random seed
    torch.cuda.manual_seed(args.seed)  # GPU random seed
    torch.backends.cudnn.benchmark = True
    
    qa, q, a = load_data(args)

    ## policy network
    policy_model = policy_network(model_config=args.model_config,
                                  add_linear=True,
                                  embedding_size=args.embedding_size,
                                  freeze_encoder=True)

    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")  # one GPU
    policy_model = policy_model.to(device)
    
    ## TRAINING
    logger = Logger(os.path.join(args.ckpt_path, 'log.txt'))
    policy_gradient_train(policy_model, qa, q, a, args)