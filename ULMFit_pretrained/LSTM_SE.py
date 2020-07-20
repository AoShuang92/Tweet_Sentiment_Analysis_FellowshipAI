import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class channel_attention(nn.Module):
    def __init__(self, num_channels=None):
        super(channel_attention, self).__init__()
        self.fc1 = nn.Linear(num_channels, num_channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_tensor):
        x = self.fc1(input_tensor)
        x = self.sigmoid(x)
        return torch.mul(input_tensor, x)

class channel_attention_v2(nn.Module):
    def __init__(self, num_channels=None, reduction=8):
        super(channel_attention_v2, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.fc1 = nn.Linear(num_channels, num_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(num_channels // reduction, num_channels)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_tensor):
        x = self.relu(self.fc1(input_tensor))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return torch.mul(input_tensor, x)


# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


class LSTM_SE(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_length, hidden_size, batch_size,  weights):
        super(LSTM_SE, self).__init__()
        
        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 3 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        weights : Pre-trained GloVe word_embeddings which we will use to create our word_embedding look-up table 
        
        """
        
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)# Initializing the look-up table.
        self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False) # Assigning the look-up table to the pre-trained GloVe word embedding.
        self.lstm = nn.LSTM(embedding_length, hidden_size, num_layers=2, bidirectional=True)
        self.attn_fc1 = channel_attention_v2(num_channels=hidden_size)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.attn_out = channel_attention_v2(num_channels=32)
        self.label = nn.Linear(32, output_size)

        #self.attn_se = SELayer(channel=25)
        
    def forward(self, input_sentence, batch_size=None):

        """ 
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
        
        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)
        
        """
        
        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
        input = self.word_embeddings(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        #input = self.attn_se(input[:,:,:,None]).squeeze()
        #print('input',input.size())
        input = input.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
        if batch_size is None:
            h_0 = Variable(torch.zeros(4, self.batch_size, self.hidden_size).cuda()) # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(4, self.batch_size, self.hidden_size).cuda()) # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
        
        
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        hidden_state = final_hidden_state[-1]
        hidden_state = self.attn_fc1(hidden_state)
        out = self.fc1(hidden_state)
        out = self.attn_out(out)
        logits = self.label(out) 
        return logits
