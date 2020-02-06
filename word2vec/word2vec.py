import random
import numpy as np
import torch
from torch.utils.data import DataLoader

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

from model import wordEmbeddingDataset,word2VecModel

def prepare_data(args):
    with open(args.data_dir,'r') as fin:
        text = fin.read()
    text = [w for w in text.lower().split()]
    vocab = dict(Counter(text).most_common(args.max_vocab_size-1))
    vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
    idx_to_word = [word for word in vocab.keys()]
    word_to_idx = {word:idx for idx,word in enumerate(idx_to_word)}

    # negsample的采样概率分布
    word_counts = np.asarray([value for value in vocab.values()],dtype=np.float32)
    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs ** (3./4.)
    word_freqs = word_freqs / np.sum(word_freqs)
    vocab_size = len(idx_to_word)

    return text,idx_to_word,word_to_idx,word_freqs,vocab_size


def setseed():
    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2020)

def train(args,model,dataloader,word_to_idx,idx_to_word):
    LOG_FILE = "word-embedding.log"
    tb_writer = SummaryWriter('./runs')
    model.train()
    t_total = args.num_epoch * len(dataloader)
    optimizer = AdamW(model.parameters(),lr=args.learnning_rate,eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,num_warmup_steps=args.warmup_steps,num_training_steps=t_total)
    train_iterator = trange(args.num_epoch,desc="epoch")
    tr_loss = 0.
    logg_loss = 0.
    global_step = 0
    for k in train_iterator:
        print("the {} epoch beginning!".format(k))
        epoch_iteration = tqdm(dataloader,desc="iteration")
        for step,batch in enumerate(epoch_iteration):
            batch = tuple(t.to(args.device) for t in batch)
            input = {"center_words":batch[0],"pos_words":batch[1],"neg_words":batch[2]}
            loss = model(**input)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step +=1

            tr_loss += loss.item()
            if (step+1) % 100 == 0:
                loss_scalar = (tr_loss - logg_loss) / 100
                logg_loss = tr_loss
                with open(LOG_FILE, "a") as fout:
                    fout.write("epoch: {}, iter: {}, loss: {},learn_rate: {}\n".format(k, step, loss_scalar,scheduler.get_lr()[0]))
                    print("epoch: {}, iter: {}, loss: {}, learning_rate: {}".format(k, step, loss_scalar,scheduler.get_lr()[0]))
                    tb_writer.add_scalar("learning_rate",scheduler.get_lr()[0],global_step)
                    tb_writer.add_scalar("loss",loss_scalar,global_step)

            if (step+1) % 2000 == 0:
                embedding_weights = model.input_embeddings()
                sim_simlex = evaluate("./worddata/simlex-999.txt", embedding_weights,word_to_idx)
                sim_men = evaluate("./worddata/men.txt", embedding_weights,word_to_idx)
                sim_353 = evaluate("./worddata/wordsim353.csv", embedding_weights,word_to_idx)
                with open(LOG_FILE, "a") as fout:
                    print("epoch: {}, iteration: {}, simlex-999: {}, men:{}, sim353:{}, nearest to monster: {}\n".format(
                                k, step, sim_simlex,sim_men,sim_353, find_nearest("monster",embedding_weights,word_to_idx,idx_to_word)))
                    fout.write("epoch: {}, iteration: {}, simlex-999: {}, men: {}, sim353: {}, nearest to monster: {}\n".format(
                                k, step, sim_simlex, sim_men, sim_353, find_nearest("monster",embedding_weights,word_to_idx,idx_to_word)))

    embedding_weights = model.input_embeddings()
    np.save("embedding-{}".format(args.embed_size), embedding_weights)
    torch.save(model.state_dict(), "embedding-{}.th".format(args.embed_size))


def evaluate(filename,embedding_weights,word_to_idx):
    if filename.endswith(".csv"):
        data = pd.read_csv(filename,sep=',')
    else:
        data = pd.read_csv(filename,sep='\t')
    human_similarity = []
    model_similarity = []
    for i in data.iloc[:,0:2].index:
        word1 ,word2 = data.iloc[i,0],data.iloc[i,1]
        if word1 not in word_to_idx or word2 not in word_to_idx:
            continue
        else:
            word1_idx,word2_idx = word_to_idx[word1],word_to_idx[word2]
            word1_embed,word2_embed = embedding_weights[[word1_idx]],embedding_weights[[word2_idx]]
            model_similarity.append(float(cosine_similarity(word1_embed,word2_embed)))
            human_similarity.append(float(data.iloc[i,2]))

    return stats.spearmanr(human_similarity,model_similarity)

def find_nearest(word,embedding_weights,word_to_idx,idx_to_word):
    word_idx = word_to_idx[word]
    embedding = embedding_weights[word_idx]
    embedding = embedding.reshape((1,-1))
    scores = cosine_similarity(embedding,embedding_weights)[0].argsort()[::-1]
    # cos_dis = np.array([spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in scores[:10]]


def main():
    parse = argparse.ArgumentParser()

    parse.add_argument("--data_dir",default="./worddata/text8.train.txt",type=str,required=False)
    parse.add_argument("--batch_size", default=16, type=int)
    parse.add_argument("--do_train",default=True, action="store_true", help="Whether to run training.")
    parse.add_argument("--learnning_rate", default=5e-4, type=float)
    parse.add_argument("--num_epoch", default=2, type=int)
    parse.add_argument("--max_vocab_size",default=30000,type=int)
    parse.add_argument("--embed_size",default=200,type=int)
    parse.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parse.add_argument("--C",default=5,type=int)
    parse.add_argument("--K",default=100,type=int)
    args = parse.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    setseed()

    text, idx_to_word, word_to_idx, word_freqs,vocab_size = prepare_data(args)
    model = word2VecModel(vocab_size,args.embed_size)
    dataset = wordEmbeddingDataset(text,word_to_idx,idx_to_word,word_freqs,args.C,args.K)
    dataloader = DataLoader(dataset,batch_size = args.batch_size,shuffle=True,num_workers=4)
    model.to(device)

    if args.do_train:
        train(args,model,dataloader,word_to_idx,idx_to_word)

    LOG_FILE = "word-embedding.log"
    model.load_state_dict(torch.load("embedding-{}.th".format(args.embed_size)))
    embedding_weights = model.input_embeddings()

    with open(LOG_FILE,'a') as fout:
        fout.write("simlex-999: {}, men: {}, sim353: {}\n".format(
            evaluate("./worddata/simlex-999.txt", embedding_weights,word_to_idx),
            evaluate("./worddata/men.txt", embedding_weights,word_to_idx),
            evaluate("./worddata/wordsim353.csv", embedding_weights,word_to_idx)))

        for word in ["good", "fresh", "monster", "green", "like", "america", "chicago", "work", "computer", "language"]:
            nearest = find_nearest(word,embedding_weights,word_to_idx,idx_to_word)
            print(word, nearest)
            fout.write("word:{}, nearest:{}\n".format(word,nearest))

    man_idx = word_to_idx["man"]
    king_idx = word_to_idx["king"]
    woman_idx = word_to_idx["woman"]
    embedding = embedding_weights[woman_idx] - embedding_weights[man_idx] + embedding_weights[king_idx]
    embedding = embedding.reshape((1,-1))
    scores = cosine_similarity(embedding,embedding_weights)[0].argsort()[::-1]
    # cos_dis = np.array([spatial.distance.cosine(e, embedding) for e in embedding_weights])
    with open(LOG_FILE,'a') as fout:
        for i in scores[:20]:
            print(idx_to_word[i])
            fout.write("the nearest to <women-man+king> : {}\n".format(idx_to_word[i]))


if __name__ == '__main__':
    main()