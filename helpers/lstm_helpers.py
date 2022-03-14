import re
import pickle
import numpy as np
from nltk.stem.isri import ISRIStemmer
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from tqdm import tqdm



    
class DialectRNN(nn.Module):
    """
    The RNN model that will be used to perform Dialect Classification
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, 
                 n_layers, seq_length, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(DialectRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        
        # define all layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.fc1 = linear(hidden_dim, 64, batch_norm=False)
#        self.fc2 = linear(256, 256)
        self.fc2 = linear(64, output_size, batch_norm=False)

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
#         out = self.dropout(lstm_out)
        out = self.fc1(lstm_out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
#        out = F.relu(out)
#        out = self.dropout(out)
#        out = self.fc3(out)

        out = out.view(batch_size, self.seq_length, self.output_size)
        # we only care about the last stage
        out = out[:, -1, :]
        
        # return last output and hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size, train_on_gpu):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                     weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                     weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden
        
        

def features_split(X, y, split_fraction=0.2):
    '''
    Split features (X), and labels (y) to train, validation, test sets.
    Size of train set is (1-split_fraction) of all dataset
    Each of validation, test sets have size (split_fraction/2) of all dataset

    Parameters
    X: numpy array
    Numpy array contains the features of the dataset

    y: numpy array
    Numpy array contains the labels of the dataset

    split_fraction: float, optional
    Size of the validation and test sets, default(0.2)

    Returns
    datasets: dictionary
    Dictionary with keys are 'train', 'val', and 'test', 
    and values are tuples: (X_train, y_train), (X_val, y_val), and (X_test, y_test)
    '''

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_fraction, shuffle=True, 
                                                      random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, shuffle=True, 
                                                    random_state=42, stratify=y_val)

    datasets = {'train': (X_train, y_train), 'val': (X_val, y_val), 
                'test':(X_test, y_test)} 

    return datasets


def move_to(variables, to):
    '''
    Move variables to cpu to gpu and vice versa

    Parameters
    varaibles: list 
    List of variables

    to: string
    "to" is the distination of the variables, expected values: 'gpu', or 'cpu'
    
    Returns
    variables: list
    List of the given variables after moving them.
    '''        
    if to == 'gpu':
        for i in range(len(variables)):
            variables[i] = variables[i].cuda()

    if to == 'cpu':
        for i in range(len(variables)):
            variables[i] = variables[i].cpu()

    return variables
                                    

def linear(in_features, out_features, batch_norm=True):
    '''
    Create Linear layer with batch normalization

    Paremeters
    in_features: int
    Number of input features

    out_features: int
    Number of ouput features

    batch_norm: boolean
    if true(default), add Batch Normalization layer after the Linear layer

    Returns
    sequential container of specefied layers
    '''
    layers = []
    # if batch_norm is true, then we don't need the biases in Linear layer
    bias = not batch_norm
    layers.append(nn.Linear(in_features, out_features, bias=bias))
    
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_features))
    
    return nn.Sequential(*layers)



def train_batch_lstm(model, inputs, labels, h, optimizer, criterion, clip, train_on_gpu):
    model.train()
    
    if train_on_gpu:
        inputs, labels = move_to([inputs, labels], to='gpu')
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    # zero accumulated gradients
    model.zero_grad()

    # get the output from the model
    output, h = model(inputs, h)
    # calculate the loss and perform backprop
    loss = criterion(output.squeeze(), labels)
    loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    return output, loss, h



def train_epoch_lstm(model, train_loader, optimizer, criterion, clip, train_on_gpu):
    all_labels = []
    all_preds = []
    # number of complete batches in train loader
    train_n_batches = len(train_loader.dataset) // train_loader.batch_size
    train_loss = 0
    # total number of elements used to train the model until now
    train_total = 0
    # initialize hidden state
    h_train = model.init_hidden(train_loader.batch_size, train_on_gpu)
    batch_number = 1
    for inputs, labels in tqdm(train_loader):
        if batch_number > train_n_batches:
            break
        batch_number += 1

        output, loss, h_train = train_batch_lstm(model, inputs, labels, h_train, optimizer, criterion, clip, train_on_gpu)
        
        train_loss += loss.item() * len(labels)
        train_total += len(labels)
        
        pred = output.data.max(1, keepdim=True)[1]
        
        labels = labels.cpu()
        pred = pred.cpu()
        all_labels.extend(list(labels.numpy()))
        all_preds.extend(list(pred.numpy()))
        
    all_preds = np.array([x.item() for x in all_preds])
    all_labels = np.array(all_labels)
    f1_score_macro = f1_score(all_labels, all_preds, average='macro')

    return f1_score_macro, train_loss / train_total


def validate_batch_lstm(model, inputs, labels, h, criterion, train_on_gpu):
    model.eval()

    if train_on_gpu:
        inputs, labels = move_to([inputs, labels], to='gpu')
    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    output, h = model(inputs, h)
    loss = criterion(output.squeeze(), labels)
    return output, loss, h



def validate_lstm(model, val_loader, criterion, train_on_gpu):
    all_labels = []
    all_preds = []
    # number of complete batches in val loader
    val_n_batches = len(val_loader.dataset) // val_loader.batch_size
    val_loss = 0
    # total number of elements used to valiate the model until now
    val_total = 0
    # initialize hidden state
    h_val = model.init_hidden(val_loader.batch_size, train_on_gpu)
    batch_number = 1
    for inputs, labels in tqdm(val_loader):
        if batch_number > val_n_batches:
            break
        batch_number +=1

        output, loss, h_val = validate_batch_lstm(model, inputs, labels, h_val, criterion, train_on_gpu)
        val_loss += loss.item() * len(labels)
        val_total += len(labels)
        
        pred = output.data.max(1, keepdim=True)[1]
        
        labels = labels.cpu()
        pred = pred.cpu()
        all_labels.extend(list(labels.numpy()))
        all_preds.extend(list(pred.numpy()))
        
    all_preds = np.array([x.item() for x in all_preds])
    all_labels = np.array(all_labels)
    f1_score_macro = f1_score(all_labels, all_preds, average='macro')
    return f1_score_macro, val_loss / val_total

def train_lstm(model, n_epochs, optimizer, criterion, scheduler, loaders, train_on_gpu, save_path, criteria, print_every=1, clip=5):
    print('Start Training on "{}" for {} epochs...'.format('GPU' if train_on_gpu else 'CPU', n_epochs))
    train_loader = loaders['train']
    val_loader = loaders['val']

    if train_on_gpu:
        model = move_to([model], 'gpu')[0]
    
    val_loss_min = np.Inf
    val_score_max = 0
    train_losses_list = []
    val_losses_list = []
    train_scores_list = []
    val_scores_list = []
    for e in range(n_epochs): 
        print('\nEpoch:', e+1,'train...')
        train_score, train_loss = train_epoch_lstm(model, train_loader, optimizer, criterion, clip, train_on_gpu)
        train_losses_list.append(train_loss)
        train_scores_list.append(train_score)

        print('validation...')
        val_score, val_loss = validate_lstm(model, val_loader, criterion, train_on_gpu)
        val_losses_list.append(val_loss)
        val_scores_list.append(val_score)
        
        # save the model weights of this epoch
        torch.save(model.state_dict(), save_path+'/model_{}.pt'.format(e))

        # print the stats of this epoch
        if e % print_every == 0:
            if criteria == 'loss':
                print("Epoch: {}/{}...".format(e+1, n_epochs),
                      "Train Loss: {:.6f}...".format(train_loss),
                      "Val Loss: {:.6f}".format(val_loss))
                
            elif criteria == 'score':
                    print("Epoch: {}/{}...".format(e+1, n_epochs),
                      "Train Score: {:.6f}...".format(train_score),
                      "Val Score: {:.6f}".format(val_score))
        
        if criteria == 'loss':
            # save model weights of this epoch as the best model if val_loss < val_loss_min 
            if val_loss < val_loss_min:
                print('Validation loss decreased from: {:.6f}, to: {:.6f}\tSAVING MODEL... \
                    in Epoch: {}'.format(val_loss_min, val_loss, e+1))

                # save the model weights
                torch.save(model.state_dict(), save_path+'/best_model.pt')

                # update minimum val loss
                val_loss_min = val_loss
                    
        elif criteria == 'score':
            # save model weights of this epoch as the best model if val_score < val_score_max 
            if val_score > val_score_max:
                print('F1 Score Macro increased form: {:.6f}, to: {:.6f}\tSAVING MODEL... \
                in Epoch: {}'.format(val_score_max, val_score, e+1))

                # save the model weights
                torch.save(model.state_dict(), save_path+'/best_model.pt')

                # update max val score
                val_score_max = val_score
                
        print('===============================================================')         
         
        scheduler.step()
        
    return (train_losses_list, val_losses_list) if criteria == 'loss' else (train_scores_list, val_scores_list)
        

def test(model, loader, criterion, train_on_gpu):
    model.eval()
    # Get test data loss and accuracy
    test_loss = 0
    test_total = 0
    num_correct = 0
    all_labels = []
    all_preds = []
    # init hidden state
    h = model.init_hidden(loader.batch_size, train_on_gpu)
    
    # iterate over test data
    n_batches = len(loader.dataset)//loader.batch_size
    batch_number = 1
    for inputs, labels in loader:
        if batch_number > n_batches:
            break
        batch_number += 1
        
        output, loss, h = validate_batch_lstm(model, inputs, labels, h, criterion, train_on_gpu)
        
        test_loss += loss.item() * len(labels)
        test_total += len(labels)
        
        pred = output.data.max(1, keepdim=True)[1]

        labels = labels.cpu()
        pred = pred.cpu()
        all_labels.extend(list(labels.numpy()))
        all_preds.extend(list(pred.numpy()))
        
        # compare predictions to true labels
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)


    # avg test loss
    test_loss = test_loss / test_total

    # accuracy over all test data
    test_acc = num_correct/len(loader.dataset)

    all_preds = np.array([x.item() for x in all_preds])
    all_labels = np.array(all_labels)
    f1_score_macro = f1_score(all_labels, all_preds, average='macro')

    return f1_score_macro, test_loss, test_acc


# def predict(model, inputs, use_cuda):
    
#     output, h = model(inputs, h)