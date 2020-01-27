import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,dataloader


class word2VecModel(nn.Module):
    def __init__(self,vocab_size,emb_size):
        super(word2VecModel,self).__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size

        initrange = 0.5/self.emb_size
        self.in_embed = nn.Embedding(vocab_size,emb_size)
        self.in_embed.weight.data.uniform_(-initrange,initrange)

        self.out_embed = nn.Embedding(vocab_size,emb_size)
        self.out_embed.weight.data.uniform_(-initrange,initrange)

    def forward(self,center_words,pos_words,neg_words):

        batch_size = center_words.size(0)

        input_embedding = self.in_embed(center_words) #[batch,emb]
        pos_embedding = self.out_embed(pos_words) #[batch,2c,emb]
        neg_embedding = self.out_embed(neg_words)  #[batch,2c*k,emb]

        log_pos = torch.matmul(pos_embedding,input_embedding.unsqueeze(2)).squeeze() # [batch,2c]
        log_nes = torch.matmul(neg_embedding,-input_embedding.unsqueeze(2)).squeeze() #[batch,2c*k]

        log_pos_los = F.logsigmoid(log_pos).sum(1)
        log_neg_los = F.logsigmoid(log_nes).sum(1)
        loss = log_neg_los+log_pos_los

        return -loss.mean()

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()


class wordEmbeddingDataset(Dataset):
    def __init__(self,text, word_to_idx, idx_to_word, word_freqs,C,K):
        '''
        :param text: 语料
        :param word_to_idx:
        :param idx_to_word:
        :param word_freqs: 词频的3/4，negatiesample
        :param C: skip_gram的周围词个数
        :param K: negative sample的个数
        '''
        super(wordEmbeddingDataset, self).__init__()
        self.vocab_size = len(word_to_idx)
        self.text_encoded = [word_to_idx.get(t,self.vocab_size-1) for t in text]
        self.text_encoded = torch.tensor(self.text_encoded).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.tensor(word_freqs)
        self.C = C
        self.K = K

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx-self.C,idx)) + list(range(idx+1,idx+self.C+1))
        # 前后超范围从后前取
        pos_indices = [i%len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(self.word_freqs,self.K*pos_words.shape[0],replacement=True)

        return center_word,pos_words,neg_words

