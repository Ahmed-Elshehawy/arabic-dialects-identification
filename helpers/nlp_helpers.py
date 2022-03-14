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


def clean_hashtags(doc):
    '''
    Clean hashtags by removing "#" and, then replace underscores "_" 
    with spaces.
    
    Example: "#الحرب_العالمية" becomes "الحرب العالمية"

    Paremeters:
    doc: string
    string in any language

    Returns:
    doc: string
    same input string without "#" and "_"
    '''
    # 1- remove "#"
    doc = re.sub('#', '', doc)
    # 2- replace "_" with space
    doc = re.sub('_', ' ', doc)
    return doc


def normalize_text(doc, alef=True, hamzah=True, taa_marbotah=True, alef_layennah=True):
    '''
    Replace letters that may cause confusion with one letter to standarize
    all the text.

    Parameters:
    doc: string
    text that contains arabic alphabet
    
    alef: boolean, optional
    if true (default): replace any letter in "إأٱآ" with "ا"
    
    hamzah: boolean, optional
    if true (default): replace any letter in "ؤئ" with ء
    
    taa_marbotah: boolean, optional
    if true (default): replace any "ة" with "ه"
    
    alef_layennah: boolean, optional
    if true (default): replace any "ى" with "ي"
    
    Returns:
    doc: string
    normalized text
    '''

    alef_group = 'إأٱآ'
    hamzah_group = 'ؤئ'
    taa_marbotah_group = 'ة'
    alef_layennah_group = 'ى'

    if alef:
        # normalize alef
        doc = re.sub('[{}]'.format(alef_group), 'ا', doc)
    if hamzah:
        # normalize hamzah
        doc = re.sub('[{}]'.format(hamzah_group), 'ء', doc)
    if taa_marbotah:
        # normalize taa marbotah
        doc = re.sub('{}'.format(taa_marbotah_group), 'ه', doc)
    if alef_layennah:
        # normalize alef layennah
        doc = re.sub('{}'.format(alef_layennah_group), 'ي', doc)

    return doc


def remove_tashkeel(doc):
    '''
    Remove taskeel from document

    Parametes:
    doc: string
    text that contains arabic alphabet

    Returns:
    doc: string
    same text after removing all tashkeel from it
    '''
    # all tashkeel in arabic
    tashkeel = ['ُ', 'ّ', 'َ', 'ً', 'ِ', 'ٍ', 'ٌ', 'ْ', '~']
    tashkeel = ''.join(tashkeel) # make them one string

    doc = re.sub('[{}]'.format(tashkeel), '', doc)

    return doc



def clean_doc_arabic(doc):
    '''
    Keep only arabic letters in a document, this functions assumes that 
    this doc is (completly)normalized

    Parameters:
    doc: string
    text that contains arabic alphabet, other alphabets, and other symbols.

    Returns:
    doc: string
    text that has only arabic letters
    '''
    arabic_alphabet = 'ابتثجحخدذرزسشصضطظعغفقكلمنهويء'

    # keep only arabic letters and white spaces
    doc = re.sub('[^{}\s]'.format(arabic_alphabet), '', doc)
    return doc



def remove_curr(doc, curr_path):
    '''
    Remove currencies symbols from the text, this functions assumes
    this doc is (completly)normalized and has no tashkeel

    Parameters:
    doc: string
    text that may contian currency symbols

    curr_path: string
    path to txt file that contains symbols to be removed

    doc: string
    same text after removing all currency symbols
    '''

    with open(curr_path) as f:
        # read all data, then split it
        curr_list = f.read().split('\n')

    
    # some symbols are written in arabic so we want to remove tashkeel and normalize them
    # convert to set to remove any dupicated symbols after normalization and removing tashkeel
    curr_set = set(map(remove_tashkeel, curr_list)) 
    curr_set = set(map(normalize_text, curr_list))
    
    # now we keep only arabic symbols
    curr_string = ' '.join(curr_list)
    arabic_curr = clean_doc_arabic(curr_string)
    
    # creat the pattern
    curr_symbols = '|'.join(arabic_curr.split())

    # remove currency symbols
    doc = re.sub('{}'.format(curr_symbols), '', doc)

    return doc    



def remove_repeated(doc, n=2):
    '''
    Keep only one letter of any arabic letter that is repeated "n" or more times.

    Parameters:
    doc: string
    text that may contain repeated arabic letters

    n: int, optional
    threshold to remove letters, default(2)

    Returns:
    doc: string
    same text after removing repeated letters
    '''
    arabic_alphabet = 'ابتثجحخدذرزسشصضطظعغفقكلمنهويء'
    # pattern for any arabic letter that is repeated n or more times
    n = str(int(n))
    # link = '{%s,}|'%n
    # end = '{%s,}'%n
    # pattern = link.join(list(arabic_alphabet))+end

    # remove repeated letters
    # doc = re.sub('{}'.format(pattern), '', doc)
    
    frequency = '{%s,}'%n
    for l in arabic_alphabet:
        doc = re.sub('{}{}'.format(l, frequency), l, doc)
    
    return doc


def tokenize(doc):
    '''
    Convert a document to list of words

    Parameters:
    doc: string
    text document

    Returns:
    words_list: list
    list of all words in the input document
    '''
    words_list = doc.split()
    return words_list



def remove_stopwords(doc, stopwords):
    '''
    Remove stop words for a document

    Parametes
    doc: list
    list of words (string)

    sropwords: list
    list of words to be removed from doc

    Returns:
    doc: list
    list of words without stop words
    '''

    # len(word) > 1, will remove any remaining unwanted characters
    doc = [word for word in doc if (len(word) > 1) and (word not in stopwords)]
    return doc



def get_one_string(doc):
    '''
    Convert any sequence of words to one string

    Paremters:
    doc: list
    list of words (string)

    Returns:
    one_string: str
    string that contains all words in doc sperated by space
    '''
    one_string = ' '.join(doc)
    return one_string


def get_corpus_words(series):
    '''
    Get all words in a corpus as a list

    Paremters:
    series: pandas.core.series.Series
    pandas series that contians all docs, each doc is a list of words

    Returns:
    words: list
    list of all words in this corpus
    '''

    # all_docs: is an array of docs, each doc is on string of words separated by space
    all_docs = series.apply(get_one_string).values
    # all_doc_words: is one string contains all words of all docs separated by space
    all_doc_words = get_one_string(all_docs)

    words = all_doc_words.split() # list of all words in our corpus
    return words


def get_vocab(words):
    '''
    Get all unique words in a list of words

    Parametes
    words: list 
    list of words(string)

    Returns
    vocab: list
    list of unique words(string), ordered form most frequent to least frequent
    '''
    # count the occurence of unique words
    counter = Counter(words)
    vocab = sorted(counter, key=counter.get, reverse=True)

    return vocab


def get_mappings(seq, i=0):
    '''
    Get dictionary that maps seq elements to integers
    and dictionary that maps integers to seq elements

    Parametes
    seq: list

    i: int,  optional
    first number in the mappings, default (0)

    Returns:
    elem_to_int: dictionay
    dictionary where keys are the seq elements and values are integers 
    starting from i

    int_to_elem: dictionary
    dictionary wehre keys are integers starting from 1 and values are
    the seq elements
    '''

    # dictionary that maps seq elements to integers
    i = int(i)
    elem_to_int = {elem: i for i, elem in enumerate(seq, i)}

    # dictionary that maps integers to seq elements
    int_to_elem = {i: elem for elem, i in elem_to_int.items()}

    return elem_to_int, int_to_elem



def encode(doc, vocab_to_int):
    '''
    Tokenize the given document

    Parameters
    doc: list
    list of words

    vocab_to_int: dictionary
    dictionary contains mappings from words to integers

    Returns
    tokenized_doc: list
    list of integers corresponding to each word in doc
    '''
    tokenized_doc = [vocab_to_int.get(word, 0) for word in doc]
    return tokenized_doc



def get_zero_length_docs(docs):
    '''
    Get indexes of zero length docs in a list of docs, 
    each doc is a list

    Parameters
    docs: list
    list of documents, each document is a list

    Returns
    inds: list
    list contains teh indexes of zero length docs
    '''
    # list of indexes
    inds = []
    for i, doc in enumerate(docs):
        if len(doc) == 0:
            inds.append(i)

    return inds


def pad_docs(docs, seq_length):
    '''
    Pad docs that are shorters than seq_length with zeros
    and cut the end of docs longer than seq_length

    Paramters
    docs: list
    list of documents

    seq_length: int
    the threshold
    '''
    features = np.zeros((len(docs), seq_length), dtype=np.int64)
    
    for i, doc in enumerate(docs):
        features[i, -len(doc):] = doc[:seq_length]
    
    return features


def stem(doc):
    '''
    Stem all words in a given doc using the ISRI stemme

    Parameteres
    doc: list
    list of words

    Returns
    stem_doc: list
    list of doc stemmed words
    '''

    stemmer = ISRIStemmer()
    # initialize empty list to have all stemmed words
    stem_doc = []
    for word in doc:
        stem_doc.append(stemmer.stem(word))

    return stem_doc


def save_pickle_file(obj, save_path):
    '''
    Save object as pickle file

    Parameters
    obj: Python object
    the object you want to save

    save_path: string
    this is where the object is saved

    Returns
    None
    '''

    with open(save_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle_file(path):
    '''
    Load an object from the disk

    Parameters
    path: string
    the path to the pickle file

    Returns
    obj: Python object
    the loaded object
    '''

    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


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

    return loss, h



def train_epoch_lstm(model, train_loader, optimizer, criterion, clip, train_on_gpu):
    # number of complete batches in train loader
    train_n_batches = len(train_loader.dataset) // train_loader.batch_size
    train_loss = 0
    # total number of elements used to train the model until now
    train_total = 0
    # initialize hidden state
    h_train = model.init_hidden(train_loader.batch_size)
    batch_number = 1
    for inputs, labels in tqdm(train_loader):
        if batch_number > train_n_batches:
            break
        batch_number += 1

        loss, h_train = train_batch_lstm(model, inputs, labels, h_train, optimizer, criterion, clip, train_on_gpu)
        
        train_loss += loss.item() * len(labels)
        train_total += len(labels)

    return train_loss / train_total


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
    # number of complete batches in val loader
    val_n_batches = len(val_loader.dataset) // val_loader.batch_size
    val_loss = 0
    # total number of elements used to valiate the model until now
    val_total = 0
    # initialize hidden state
    h_val = model.init_hidden(val_loader.batch_size)
    batch_number = 1
    for inputs, labels in tqdm(val_loader):
        if batch_number > val_n_batches:
            break
        batch_number +=1

        _, loss, h_val = validate_batch_lstm(model, inputs, labels, h_val, criterion, train_on_gpu)

        val_loss += loss.item() * len(labels)
        val_total += labels

    return val_loss / val_total

def train_lstm(model, n_epochs, optimizer, criterion, scheduler, loaders, train_on_gpu, save_path, print_every=1, clip=5):
    print('Start Training on "{}" for {} epochs...'.format('GPU' if train_on_gpu else 'CPU', n_epochs))
    train_loader = loaders['train']
    val_loader = loaders['val']

    if train_on_gpu:
        model = move_to([model], 'gpu')[0]
    
    val_loss_min = np.Inf

    train_losses_list = []
    val_losses_list = []
    for e in range(n_epochs): 
        print('epoch:', e+1,'train...')
        train_loss = train_epoch_lstm(model, train_loader, optimizer, criterion, clip, train_on_gpu)
        train_losses_list.append(train_loss)

        print('validation...')
        val_loss = validate_lstm(model, val_loader, criterion, train_on_gpu)
        val_losses_list.append(val_loss)

        # save the model weights of this epoch
        torch.save(model.state_dict(), save_path+'/model_{}.pt'.format(e))

        # save model weights of this epoch as the best model if val_loss < val_loss_min 
        if val_loss < val_loss_min:
            print('Validation loss decreased from: {:.6f}, to: {:.6f}\tSAVING MODEL... \
                in Epoch: {}\n'.format(val_loss_min, val_loss, e+1))
            
            # save the model weights
            torch.save(model.state_dict(), save_path+'/best_model.pt')
            
            # update minimum val loss
            val_loss_min = val_loss

        
        # print the stats of this epoch
        if e % print_every == 0:
            print("Epoch: {}/{}...".format(e+1, n_epochs),
                  "Train Loss: {:.6f}...".format(train_loss),
                  "Val Loss: {:.6f}".format(val_loss))
         
    scheduler.step()
    return train_losses_list, val_losses_list
        

def test(model, loader, criterion, train_on_gpu):
    model.eval()
    # Get test data loss and accuracy
    test_loss = 0
    test_total = 0
    num_correct = 0
    all_labels = []
    all_preds = []
    # init hidden state
    h = model.init_hidden(loader.batch_size)
    
    # iterate over test data
    n_batches = len(loader.dataset)//loader.batch_size
    batch_number = 1
    for inputs, labels in loader:
        if batch_number > n_batches:
            break
        i += 1
        
        output, loss, h = validate_batch_lstm(model, inputs, labels, h, criterion, train_on_gpu)
        
        test_loss += loss.item() * len(labels)
        test_total += labels
        
        pred = output.data.max(1, keepdim=True)[1]

        all_labels.extend(list(labels.cpu().numpy()))
        all_preds.extend(list(pred.cpu().numpy()))
        
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