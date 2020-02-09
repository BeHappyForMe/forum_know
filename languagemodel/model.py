import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self,rnn_type,vocab_size,embed_size,hidden_size,num_layers,dropout=0.5):
        super(RNNModel,self).__init__()
        '''
            模型包含以下几层:
            - 词嵌入层
            - 循环网络层
            - 线性层，从hidden state映射到输出单词表
            - dropout层
        '''
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size,embed_size)

        if rnn_type in ['LSTM','GRU']:
            self.rnn = getattr(nn,rnn_type)(embed_size,hidden_size,num_layers,
                                            dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(embed_size,hidden_size,num_layers,nonlinearity=nonlinearity,
                              dropout=dropout)
        # 解码成单词输出
        self.decoder = nn.Linear(hidden_size,vocab_size)
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange,initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange,initrange)

    def forward(self,x,hidden):
        # 【seq,batch,embed】
        emb = self.dropout(self.embedding(x))

        # output: 【seq,batch,bidirectional * hidden】
        # hidden: 【num_layers * bidirectional, batch, hidden】
        output,hidden = self.rnn(emb,hidden)

        output = self.dropout(output)
        # 做线性映射前需要将维度拉长
        decoded = self.decoder(output.view(output.shape[0]*output.shape[1],-1))

        # 恢复维度
        return decoded.view(output.shape[0],output.shape[1],decoded.shape[1]),hidden

    def init_hidden(self,bsz, requires_grad=True):
        weight = next(self.parameters())
        # 所有参数self.parameters()，是个生成器，LSTM所有参数维度种类如下：
        # print(list(iter(self.parameters())))

        if self.rnn_type == 'LSTM':
            return (weight.new_zeros((self.num_layers, bsz, self.hidden_size), requires_grad=requires_grad),
                    weight.new_zeros((self.num_layers, bsz, self.hidden_size), requires_grad=requires_grad))

        else:
            return weight.new_zeros((self.num_layers, bsz, self.hidden_size), requires_grad=requires_grad)
            # GRU神经网络把h层和c层合并了，所以这里只有一层。

