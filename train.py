import wandb
wandb.login(key='24434976526d9265fdbe2b2150787f46522f5da4')

import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from types import SimpleNamespace
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import random
from utilities import *
import argparse

#loading data
train,valid,test=load_data(data_path,lang)

add_start_end(train) #adding start and end characters
add_start_end(valid)
add_start_end(test)

train_src_chars,train_target_chars=get_unique_chars(train) # obtain unique charcaters
valid_src_chars,valid_target_chars=get_unique_chars(valid)
test_src_chars,test_target_chars=get_unique_chars(test)
train_target_chars.add('*') # extra char to handle unknowns in valid and test data.
    
src_char_idx,src_idx_char=get_char_map(train_src_chars) # create map for each unique charcter to -> integer
target_char_idx,target_idx_char=get_char_map(train_target_chars)

encoder_vocab_size=len(src_char_idx)+1 # one extra for padding
decoder_vocab_size=len(target_char_idx)+1 # one extra for padding

max_seq_length=train[0].apply(lambda x:len(x)).max() # maximum sequence length of latin word
max_target_length=train[1].apply(lambda x:len(x)).max() # maximum word length of target


# vectorize train,valid,test based on characters mapping
train_src_int,train_target_int=vectorize(train,src_char_idx,target_char_idx,max_seq_length)
valid_src_int,valid_target_int=vectorize(valid,src_char_idx,target_char_idx,max_seq_length)
test_src_int,test_target_int=vectorize(test,src_char_idx,target_char_idx,max_seq_length)


upto=1000
train_src_int=train_src_int[:upto]
train_target_int=train_target_int[:upto]

valid_src_int=valid_src_int[:upto]
valid_target_int=valid_target_int[:upto]


def predict_test(model):
    ''' Used at the end to calculate test accuracy'''
    with torch.no_grad():
        target_end_index=target_char_idx['$']
        batch_no=0
        test_acc=0
        for data in get_batch(test_src_int,test_target_int,config.batch_size):
            batch_no+=1
            x=data[0]
            y=data[1]
            x=x.to(torch.int64).T
            y=y.to(torch.int64).T
            target=y.detach().cpu().numpy()
            src=x.detach().cpu().numpy()
            outputs,_=model.forward(x,y,prediction=True) # prediction=True,disables teacher forcing.
            batch_acc=cal_acc(outputs,y)
            test_acc+=batch_acc
        test_acc/=batch_no
        return test_acc
    
            
    

def main():
    config=wandb.config
    config.encoder_vocab_size=encoder_vocab_size
    config.decoder_vocab_size=decoder_vocab_size
    config.max_seq_length=max_seq_length
    
    if config.attention=='Yes':
        model=AttentionSeq2Seq(config).to(device) # if attention is enabled
    else:
        model=Seq2Seq(config).to(device) # if attention is not enabled.
        
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(config.epochs):
        train_loss=0
        train_acc=0
        model.train()
        batch_no=0
        for data in get_batch(train_src_int,train_target_int,config.batch_size):
            batch_no+=1
            x=data[0]
            y=data[1]
            x=x.to(torch.int64).T
            y=y.to(torch.int64).T
            outputs,attention_scores=model.forward(x,y)
            output=outputs.reshape(-1,outputs.shape[2])
            target=y.reshape(-1)
            optimizer.zero_grad() # zeroing grads before backpropagtaing gradients
            target=target-1
            target[target<0]=0
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # gradient clipping
            optimizer.step()# update parameters
            train_loss+=loss.item()*config.batch_size

            batch_acc=cal_acc(outputs,y)
            train_acc+=batch_acc
        train_loss/=len(train_src_int)
        train_acc/=batch_no # avg train accuracy across all batches
        model.eval()

        valid_loss=0
        valid_acc=0
        batch_no=0
        with torch.no_grad():
            for data in get_batch(valid_src_int,valid_target_int,config.batch_size):
                batch_no+=1
                x=data[0]
                y=data[1]
                x=x.to(torch.int64).T
                y=y.to(torch.int64).T
                outputs,attention_scores=model.forward(x,y,prediction=True)# prediction set to True to disable teacher forcing
                output=outputs.reshape(-1,outputs.shape[2])
                target=y.reshape(-1)
                target=target-1
                target[target<0]=0
                loss = criterion(output, target)
                valid_loss+=loss.item()*config.batch_size
                valid_acc+=cal_acc(outputs,y)
            valid_loss/=len(valid_src_int)
            valid_acc/=batch_no
        print(f'Epoch: {epoch+1} Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f} Train Acc: {train_acc:.4f}  Valid Acc: {valid_acc:.4f}')
        wandb.log({'train accuracy':train_acc,'train loss':train_loss,'valid accuracy':valid_acc,'valid loss':valid_loss})
    return model

if __name__=="__main__": 
    
    #adding support for command line arguments
    parser = argparse.ArgumentParser(description = 'Set Hyper Parameters')
    parser.add_argument('-wp'   , '--wandb_project'  , type = str  , default='CS22M080',metavar = '', help = 'WandB Project Name (Non-Empty String)')
    parser.add_argument('-we'   , '--wandb_entity'   , type = str  , default='CS22M080',metavar = '', help = 'WandB Entity Name (Non-Empty String)')
    parser.add_argument('-e'    , '--epochs'         , type = int  , default=15,metavar = '', help = 'Number of Epochs (Positive Integer)')
    parser.add_argument('-b'    , '--batch_size'     , type = int  , default=256,metavar = '', help = 'Batch Size (Positive Integer)')
    parser.add_argument('-el'  , '--encoder_num_layers'     , type = int  , default=3,metavar = '', help = '')
    parser.add_argument('-dl'  , '--decoder_num_layers'     , type = int  , default=2,metavar = '', help = '')
    parser.add_argument('-hs'   , '--hidden_size'  , type = int  , default=512,metavar = '', help = 'Cell hidden size')
    parser.add_argument('-es'   , '--embedding_size'  , type = int  , default=256,metavar = '', help = 'embedding size')
    parser.add_argument('-do'   , '--dropout'  , type = float  , default=0.2,metavar = '', help = 'Dropout in cell')
    parser.add_argument('-bi'   , '--bidirectional'  , type = str  , default="Yes",choices=['Yes','No'])
    parser.add_argument('-ct'   , '--cell_type'  , type = str  , default="GRU",choices=['LSTM','GRU','RNN'])
    parser.add_argument('-at'   , '--attention'  , type = str  , default="Yes",choices=['Yes','No'])
    
    
    params = vars(parser.parse_args())
    wandb.init(project=params['wandb_project'],config=params)
    config=wandb.config
    run_name=f'Attention {config.attention} - Cell-{config.cell_type} Hidden-{config.hidden_size} Embedding-{config.embedding_size} Bidir-{config.bidirectional} Dropout -{config.dropout} EL-{config.encoder_num_layers} DL-{config.decoder_num_layers}'
    wandb.run.name=run_name
    model=main()
    print("training completed")
    test_accuracy=predict_test(model)
    print("*"*50)
    print(f"Final Test accuracy: {test_accuracy:.2f}")
    print("*"*50)
    wandb.log({'test accuracy':test_accuracy})
    wandb.finish()
    
    
