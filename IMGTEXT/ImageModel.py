from unicodedata import bidirectional
import torch
from torch import nn
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageModel(nn.Module):
    def __init__ (self):
        super(ImageModel, self).__init__()
        self.encoder = nn.LSTM(4096, 4096, batch_first = True, bidirectional = True)  #batch_first -> input and output tensors are provided as (batch, seq, feature)
        self.decoder = nn.LSTM(1024 *2 + 1024, 1024 , batch_first = True )
        self.linear1 = nn.Linear(60000 , 4096)

    def forward(self, X1,X2 , Y):
        X1 = X1.view(X1.size(0), -1)  
        X2 = X2.view(X2.size(0), -1)
        X = torch.cat((X1,X2) , dim = 1)
        X = self.linear1(X)
        X = torch.unsqueeze(X, dim=1)
        enc_outputs, _ = self.encoder(X) #LSTM input (batch_size, seq_length (panjang text max atau panjang LSTMnya atau detik video , input size)
        #for t in range(1, trg_len):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            #output, hidden, cell = self.decoder(input, hidden, cell)
        return enc_outputs


