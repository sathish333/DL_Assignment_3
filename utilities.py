import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from types import SimpleNamespace
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import random
import torch.nn.functional as F

data_path='data/aksharantar_sampled/'
lang='tel'
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(path,lang):
    ''' reading data from path based on the lang '''
    train=pd.read_csv(path+lang+'/'+lang+'_train.csv',header=None)
    valid=pd.read_csv(path+lang+'/'+lang+'_valid.csv',header=None)
    test=pd.read_csv(path+lang+'/'+lang+'_test.csv',header=None)
    return train,valid,test

def add_start_end(df):
    ''' adding end charcter to the source, start and end character to the target'''
    df[0]=df[0].apply(lambda x: x+'$')
    df[1]=df[1].apply(lambda x: '^'+x+'$')


def get_unique_chars(df):
    ''' return source and target unique chars'''
    src_chars=set()
    target_chars=set()

    _=df[0].apply(lambda each: src_chars.update(each))
    _=df[1].apply(lambda each: target_chars.update(each))
    src_chars.add('^')
    target_chars.update(['^','$'])
    
    return src_chars,target_chars

def get_char_map(chars):
    ''' map chars to index and return the dict'''
    src_char_idx=dict([(char,idx) for idx,char in enumerate(chars,start=1)])
    src_idx_char=dict([(idx, char) for char, idx in src_char_idx.items()])
    return src_char_idx,src_idx_char

def vectorize(df,src_char_idx,target_char_idx,max_seq_length):
    ''' Convert words to vectors based on the char mapping'''
    src_int=[]
    for word in df[0]:
        word_int=[]
        for char in word:
            word_int.append(src_char_idx[char])
        src_int.append(torch.Tensor(word_int))
    
    target_int=[]
    for word in df[1]:
        word_int=[]
        for char in word:
            word_int.append(target_char_idx.get(char,target_char_idx.get('*')))
        target_int.append(torch.Tensor(word_int))
      
    src_int.insert(0,torch.zeros(max_seq_length))   
    return pad_sequence(src_int)[:,1:].to(device).T,pad_sequence(target_int).to(device).T

def get_batch(src,target,batch_size):
    ''' return batch of data. will reset after epoch is over'''
    num_batches=len(src)//batch_size
    for i in range(num_batches):
        start=i*batch_size
        end=start+batch_size
        yield src[start:end],target[start:end] 

def cal_acc(pred,actual):
    '''  calculate accuracy based on prediction and actual. Pred points to  probabilties'''
    pred=pred.clone()
    pred=torch.argmax(pred,2)+1
    pred=pred[1:]
    pred=torch.transpose(pred,0,1)
    
    actual=actual[1:] 
    actual=torch.transpose(actual,0,1)
    acc=torch.logical_or(pred==actual,actual==0).all(axis=1).sum().item()/len(actual)
    
    return acc

def decode_src(src_int,src_idx_char):
    ''' map back source vector to string representation  '''
    s=""
    for each in src_int:
        if each==0:
            return s
        s+=src_idx_char[each]
    return s

def decode_target(target_int,target_idx_char,target_end_index):
    ''' map back target vector to string representation  '''
    s=""
    for each in target_int[1:]:
        if each==target_end_index:
            return s
        s+=target_idx_char[each]
    return s


        
        
class Encoder(nn.Module):
    ''' 
     class structure for Encoder.
     Based on config:dict provided it will instantinate the fields.
     Note that in forward, when called with GRU/RNN  we are returning None for cell so as match same shapes at every return.
     
    '''
    def __init__(self,config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.encoder_num_layers
        self.dropout = nn.Dropout(config.dropout)
        self.cell_type=config.cell_type
        self.embedding = nn.Embedding(config.encoder_vocab_size, config.embedding_size)
        if self.num_layers==1:
            dropout=0
        else:
            dropout=config.dropout
            
        if config.bidirectional=='Yes':
            self.bidirectional=True
        else:
            self.bidirectional=False
    
        if self.cell_type=='LSTM':
            self.cell = nn.LSTM(config.embedding_size, config.hidden_size, self.num_layers,                     dropout=dropout,bidirectional=self.bidirectional)
        elif self.cell_type=='GRU':
            self.cell = nn.GRU(config.embedding_size, config.hidden_size, self.num_layers, dropout=dropout,bidirectional=self.bidirectional)
        elif self.cell_type=='RNN':
            self.cell=nn.RNN(config.embedding_size, config.hidden_size, self.num_layers, dropout=dropout,bidirectional=self.bidirectional)
            

    def forward(self, x):
        embedding =self.dropout(self.embedding(x))
        if self.cell_type=='LSTM':
            outputs, (hidden, cell) = self.cell(embedding)
            return hidden,cell,outputs
        else:
            outputs, hidden = self.cell(embedding)
            return hidden,None,outputs
        

class Decoder(nn.Module):
    ''' 
     class structure for Decoder.
     Based on config:dict provided it will instantinate the fields.
     Note that in forward, when called with GRU/RNN  we are returning None for cell so as match same shapes at every return.
     
    '''
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.decoder_num_layers
        self.dropout = nn.Dropout(config.dropout)
        self.cell_type=config.cell_type
        self.embedding = nn.Embedding(config.decoder_vocab_size, config.embedding_size)
        self.fc = nn.Linear(config.hidden_size, config.decoder_vocab_size)
        if self.num_layers==1:
            dropout=0
        else:
            dropout=config.dropout
        
        if self.cell_type=='LSTM':
            self.cell = nn.LSTM(config.embedding_size, config.hidden_size, self.num_layers, dropout=dropout)
        elif self.cell_type=='GRU':
            self.cell = nn.GRU(config.embedding_size, config.hidden_size, self.num_layers, dropout=dropout)
        elif self.cell_type=='RNN':
            self.cell=nn.RNN(config.embedding_size, config.hidden_size, self.num_layers, dropout=dropout)
            

    def forward(self, x, hidden, cell=None):
        x = x.unsqueeze(0)
        embedding =self.dropout(self.embedding(x))
        if(self.cell_type=='LSTM'):
            outputs, (hidden, cell) = self.cell(embedding, (hidden, cell))
        else:
            outputs, hidden = self.cell(embedding, hidden)
            
        preds = self.fc(outputs).squeeze(0)
        
        return preds, hidden, cell
    
class Seq2Seq(nn.Module):
    ''' 
     Combines Encoder and Decoder classes so as to construct Sequence to sequence model.
     
    '''
    def __init__(self,config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.batch_size=config.batch_size
        self.config=config

    def forward(self,source,target,prediction=False):
         # if Prediction is True, then teacher forcing will be disbaled.
      
        outputs = torch.zeros(target.shape[0], self.batch_size,self.config.decoder_vocab_size).to(device)
        
        cell=None 
        hidden, cell,_ = self.encoder(source)
        if self.encoder.bidirectional: # take mean of the last states if it's bidirectional
                hidden=hidden[[self.encoder.num_layers-1,-1]]
                hidden=hidden.mean(axis=0).unsqueeze(0)
                
                if self.config.cell_type=='LSTM':
                    cell=cell[[self.encoder.num_layers-1,-1]]
                    cell=cell.mean(axis=0).unsqueeze(0)
        
        else:
            hidden=hidden[-1,:,:]
            hidden=hidden.unsqueeze(0)
            
            if self.config.cell_type=='LSTM':
                cell=cell[-1,:,:]
                cell=cell.unsqueeze(0)
        hidden=hidden.repeat(self.config.decoder_num_layers,1,1) # repeat it to match decoder num layers
        
        if self.config.cell_type=='LSTM':
            cell=cell.repeat(self.config.decoder_num_layers,1,1)
    
        x = target[0]
        
        for t in range(1, target.shape[0]):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            max_prob = output.argmax(1)
            
            if not prediction:
                x = target[t] if random.random() < 0.5 else max_prob
            else:
                x=max_prob

        return outputs,None
  


class AttentionDecoder(nn.Module):
    '''
    Decoder class when working with Attention.
    '''
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = config.decoder_num_layers
        self.dropout = nn.Dropout(config.dropout)
        self.cell_type=config.cell_type
        self.embedding = nn.Embedding(config.decoder_vocab_size, config.embedding_size)
        self.fc = nn.Linear(config.hidden_size, config.decoder_vocab_size)
        
        if self.num_layers==1:
            dropout=0 # if layers is 1, make dropout 0.
        else:
            dropout=config.dropout
        
        if self.cell_type=='LSTM':
            self.cell = nn.LSTM(config.hidden_size, config.hidden_size, self.num_layers, dropout=dropout)
        elif self.cell_type=='GRU':
            self.cell = nn.GRU(config.hidden_size, config.hidden_size, self.num_layers, dropout=dropout)
        elif self.cell_type=='RNN':
            self.cell=nn.RNN(config.hidden_size, config.hidden_size, self.num_layers, dropout=dropout)
         
        self.attn = nn.Linear(self.hidden_size +config.embedding_size, config.max_seq_length)
        
        if config.bidirectional=='No':
            self.attn_combined = nn.Linear(self.hidden_size +config.embedding_size, self.hidden_size)
        else:
            self.attn_combined = nn.Linear(self.hidden_size*2 +config.embedding_size, self.hidden_size)
            


    def forward(self, x, hidden, cell=None,encoder_outputs=None):
       
        x = x.unsqueeze(0)
        embedded =self.embedding(x)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 encoder_outputs.permute(1,0,2)).squeeze(1)
        output = torch.cat((embedded[0], attn_applied), 1)
        output = self.attn_combined(output)
        output = F.relu(output.unsqueeze(0))
        
        
        if(self.cell_type=='LSTM'):
            output, (hidden, cell) = self.cell(output, (hidden, cell))
        else:
            output, hidden = self.cell(output, hidden)

        preds = self.fc(output)
        

        preds = preds.squeeze(0)
        
            
        return preds, hidden, cell,attn_weights


class AttentionSeq2Seq(nn.Module):
    '''
    Combines Encoder and AttentionDecoder classes so as to make single model for Seq2seq.
    '''
    def __init__(self,config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = AttentionDecoder(config)
        self.batch_size=config.batch_size
        self.config=config

        

    def forward(self,source,target,prediction=False):
        # source  (Sequnce_lenghth,batch_size)
        # target: (max_traget_length,batch_size)

         # if Prediction is True, then teacher forcing will be disbaled.
        
        outputs = torch.zeros(target.shape[0], self.batch_size,self.config.decoder_vocab_size).to(device)
        attention_scores = torch.zeros(target.shape[0], self.batch_size,self.config.max_seq_length).to(device)
        
        cell=None 
        hidden, cell,encoder_outputs = self.encoder(source)
        if self.encoder.bidirectional:
                hidden=hidden[[self.encoder.num_layers-1,-1]]
                hidden=hidden.mean(axis=0).unsqueeze(0)
                
                if self.config.cell_type=='LSTM':
                    cell=cell[[self.encoder.num_layers-1,-1]]
                    cell=cell.mean(axis=0).unsqueeze(0)
        
        else:
            hidden=hidden[-1,:,:]
            hidden=hidden.unsqueeze(0)
            
            if self.config.cell_type=='LSTM':
                cell=cell[-1,:,:]
                cell=cell.unsqueeze(0)

        hidden=hidden.repeat(self.config.decoder_num_layers,1,1) # repeat it to match decoder num layers
        
        if self.config.cell_type=='LSTM':
            cell=cell.repeat(self.config.decoder_num_layers,1,1)
        
        x = target[0]
        
        for t in range(1, target.shape[0]):
            output, hidden, cell,attention_scores[t] = self.decoder(x, hidden, cell,encoder_outputs)
            outputs[t] = output
            max_prob = output.argmax(1)
            
            if not prediction:
                x = target[t] if random.random() < 0.5 else max_prob
            else:
                x=max_prob
        return outputs,attention_scores
 
