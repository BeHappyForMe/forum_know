import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchtext
from torchtext.vocab import Vectors

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import AdamW,get_linear_schedule_with_warmup

import argparse
from collections import Counter
from tqdm import trange,tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy
from scipy import stats,spatial
import copy
from tqdm import tqdm,trange
import os

from model import RNNModel

def setseed():
    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2020)

def preprocess_data(args):
    TEXT = torchtext.data.Field(lower=True)
    train,dev,test = torchtext.datasets.LanguageModelingDataset.splits(
        path='./data',
        train='text8.train.txt',
        validation='text8.dev.txt',
        test='text8.test.txt',
        text_field = TEXT
    )
    cache = 'mycache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    vectors = Vectors(name='/Users/zhoup/wordEmbedding/glove/glove.6B.300d.txt',cache=cache)
    # vectors.unk_init = nn.init.xavier_uniform_

    TEXT.build_vocab(train,vectors=vectors)
    VOCAB_SIZE = len(TEXT.vocab)
    train_iter,dev_iter,test_iter = torchtext.data.BPTTIterator.splits(
        (train,dev,test),
        batch_size=args.batch_size,
        device=args.device,
        bptt_len = 100,
        repeat = False,
        shuffle = True,
    )
    return VOCAB_SIZE,train_iter,dev_iter,test_iter,TEXT.vocab.vectors

def train(args,model,train_iter,val_iter,loss_fn,VOCAB_SIZE):
    LOG_FILE = "language_model_GRU.log"
    tb_writer = SummaryWriter('./runs')

    t_total = args.num_epoch * len(train_iter)
    optimizer = AdamW(filter(lambda p : p.requires_grad,model.parameters()), lr=args.learnning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    tr_loss = 0.
    logg_loss = 0.
    global_step = 0
    train_iterator = trange(args.num_epoch,desc='epoch')
    val_losses = []
    for epoch in train_iterator:
        model.train()  # 训练模式
        hidden = model.init_hidden(args.batch_size)
        # 得到hidden初始化后的维度
        epoch_iteration = tqdm(train_iter,desc='iteration')
        for i, batch in enumerate(epoch_iteration):
            data, target = batch.text, batch.target
            # 取出训练集的输入的数据和输出的数据，相当于特征和标签
            data, target = data.to(args.device), target.to(args.device)
            hidden = repackage_hidden(hidden)
            # 语言模型每个batch的隐藏层的输出值是要继续作为下一个batch的隐藏层的输入的
            # 因为batch数量很多，如果一直往后传，会造成整个计算图很庞大，反向传播会内存崩溃。
            # 所有每次一个batch的计算图迭代完成后，需要把计算图截断，只保留隐藏层的输出值。
            # 不过只有语言模型才这么干，其他比如翻译模型不需要这么做。
            # repackage_hidden自定义函数用来截断计算图的。
            model.zero_grad()  # 梯度归零，不然每次迭代梯度会累加
            output, hidden = model(data, hidden)
            # output = (50,32,50002)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            # output.view(-1, VOCAB_SIZE) = (1600,50002)
            # target.view(-1) =(1600),关于pytorch中交叉熵的计算公式请看下面链接。
            # https://blog.csdn.net/geter_CS/article/details/84857220
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step +=1
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.GRAD_CLIP)
            # 防止梯度爆炸，设定阈值，当梯度大于阈值时，更新的梯度为阈值

            tr_loss += loss.item()
            if (i+1) % 200 == 0:
                loss_scalar = (tr_loss-logg_loss) / 200
                logg_loss = tr_loss
                with open(LOG_FILE, "a") as fout:
                    fout.write("epoch: {}, iter: {}, loss: {},learn_rate: {}\n".format(epoch, i, loss_scalar,scheduler.get_lr()[0]))
                print("epoch: {}, iter: {}, loss: {}, learning_rate: {}".format(epoch, i, loss_scalar,scheduler.get_lr()[0]))
                tb_writer.add_scalar("learning_rate",scheduler.get_lr()[0],global_step)
                tb_writer.add_scalar("loss",loss_scalar,global_step)

            if (i+1) % 1000 == 0:
                val_loss = evaluate(args,model, val_iter,loss_fn,VOCAB_SIZE)
                with open(LOG_FILE, "a") as fout:
                    print("epoch: {}, iteration: {}, val_loss: {}\n".format(
                                epoch, i, val_loss))
                    fout.write("epoch: {}, iteration: {}, val_loss: {}\n".format(
                                epoch, i, val_loss))

                if len(val_losses) == 0 or val_loss < min(val_losses):
                    # 如果比之前的loss要小，就保存模型
                    print("best model, val loss: ", val_loss)
                    torch.save(model.state_dict(), "lm-best-GRU.th")
                val_losses.append(val_loss)  # 保存每1000次迭代后的验证集损失损失
                tb_writer.add_scalar("preplexity", np.exp(val_loss), global_step)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        # 这个是GRU的截断，因为只有一个隐藏层
        # 判断h是不是torch.Tensor
        return h.detach() # 截断计算图，h是全的计算图的开始，只是保留了h的值
    else: # 这个是LSTM的截断，有两个隐藏层，格式是元组
        return tuple(repackage_hidden(v) for v in h)

def evaluate(args,model, data,loss_fn,VOCAB_SIZE):
    model.eval()  # 预测模式
    total_loss = 0.
    it = iter(data)
    total_count = 0.
    with torch.no_grad():
        hidden = model.init_hidden(args.batch_size, requires_grad=False)
        # 这里不管是训练模式还是预测模式，h层的输入都是初始化为0，hidden的输入不是model的参数
        # 这里model里的model.parameters()已经是训练过的参数。
        for i, batch in enumerate(it):
            data, target = batch.text, batch.target
            # # 取出验证集的输入的数据和输出的数据，相当于特征和标签
            data, target = data.to(args.device), target.to(args.device)
            hidden = repackage_hidden(hidden)  # 截断计算图
            output, hidden = model(data, hidden)
            # 调用model的forward方法进行一次前向传播，得到return输出值
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            # 计算交叉熵损失

            total_count += np.multiply(*data.size())
            # 上面计算交叉熵的损失是平均过的，这里需要计算下总的损失
            # total_count先计算验证集样本的单词总数，一个样本有50个单词，一个batch32个样本
            # np.multiply(*data.size()) =50*32=1600
            total_loss += loss.item() * np.multiply(*data.size())
            # 每次batch平均后的损失乘以每次batch的样本的总的单词数 = 一次batch总的损失

    loss = total_loss / total_count  # 整个验证集总的损失除以总的单词数
    model.train()  # 训练模式
    return loss


def main():
    parse = argparse.ArgumentParser()

    parse.add_argument("--batch_size", default=16, type=int)
    parse.add_argument("--do_train",default=True, action="store_true", help="Whether to run training.")
    parse.add_argument("--do_eval", default=True, action="store_true", help="Whether to run training.")
    parse.add_argument("--learnning_rate", default=1e-4, type=float)
    parse.add_argument("--num_epoch", default=5, type=int)
    parse.add_argument("--max_vocab_size",default=50000,type=int)
    parse.add_argument("--embed_size",default=300,type=int)
    parse.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parse.add_argument("--hidden_size",default=1000,type=int)
    parse.add_argument("--num_layers",default=2,type=int)
    parse.add_argument("--GRAD_CLIP", default=1, type=float)
    args = parse.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    setseed()

    VOCAB_SIZE, train_iter, dev_iter, test_iter,weight_matrix = preprocess_data(args)

    model = RNNModel(weight_matrix,'GRU',VOCAB_SIZE,args.embed_size,args.hidden_size,args.num_layers)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失
    if args.do_train:
        train(args,model,train_iter,dev_iter,loss_fn,VOCAB_SIZE)

    if args.do_eval:
        model.load_state_dict(torch.load('lm-best-GRU.th'))
        model.to(device)

        test_loss = evaluate(args,model,test_iter,loss_fn,VOCAB_SIZE)
        LOG_FILE = "language_model_GRU.log"
        with open(LOG_FILE,'a') as fout:
            fout.write("test perplexity: {} ".format(np.exp(test_loss)))
        print("perplexity: ", np.exp(test_loss))

if __name__ == '__main__':
    main()