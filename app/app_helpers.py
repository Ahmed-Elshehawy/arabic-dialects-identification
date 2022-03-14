import daar.lstm_helpers as lh
import daar.nlp_helpers as nh
import nltk
from nltk.corpus import stopwords
import numpy as np
import torch



nltk.download('stopwords')
arabic_stopwords = stopwords.words('arabic')
arabic_stopwords = set(map(nh.remove_tashkeel, arabic_stopwords))
arabic_stopwords = set(map(nh.normalize_text, arabic_stopwords))

#def load_csv(file_path):
#    data = pd.read_csv(file_path, lineterminator='\n')
    
#    return data
    

def preprocess_data_no_stem(data_df, n):
    data = data_df.copy()
    data['text'] = data['text'].apply(nh.clean_hashtags)
    data['text'] = data['text'].apply(nh.normalize_text)
    data['text'] = data['text'].apply(nh.clean_doc_arabic)

    data['text'] = data['text'].apply(nh.remove_repeated, args=(n,))
    data['text'] = data['text'].apply(nh.tokenize)
    
    data['text'] = data['text'].apply(nh.remove_stopwords, args=(arabic_stopwords,))
    data['tweet_length'] = [len(tweet) for tweet in data['text']]
    # remove zero-length tweets
    data = data[data['tweet_length'] > 0]
    data.reset_index(inplace=True, drop=True)
    
    return data
    

def prepare_data_ml(data):
    data_string = data['text'].apply(nh.get_one_string)
    X = data_string.values
    
    return X
    

def predict_ml(model, data):
    y_hat = model.predict(data)    
    return y_hat
    

    
    
def prepare_data_lstm(data, vocab_to_int, seq_length=20):    
    data['text'] = data['text'].apply(nh.encode, args=(vocab_to_int,))
    features = nh.pad_docs(data['text'], seq_length).astype(np.int64)
    
    features_tensor = torch.from_numpy(features)
    return features_tensor
    
    

def predict_lstm(model, data, use_cuda):
    if use_cuda:
        model, data = lh.move_to([model, data], to='gpu')
    else:
        model, data = lh.move_to([model, data], to='cpu') 
        
    
    
    all_preds = []
    for inp in data:
        inp = torch.unsqueeze(inp, 0)
        batch_size = inp.size(0)
        h = model.init_hidden(batch_size, use_cuda)   
        output, h = model(inp, h)
        pred = output.data.max(1, keepdim=True)[1]
        pred = pred.cpu().numpy()

        all_preds.append(pred.item())
    
    return np.array(all_preds)


def load_lstm_model(weights_path, vocab_size, output_size, embedding_dim, hidden_dim, 
                 n_layers, seq_length, drop_prob, use_cuda):
 
    model = lh.DialectRNN(vocab_size, output_size, embedding_dim, hidden_dim, 
                 n_layers, seq_length, drop_prob=drop_prob)
                 
    if use_cuda:
        model = model.cuda()
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('gpu')))
        
    else:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        
    return model  
    

def load_ml_model(path):
    #path = 'no_stem/models/pipe_rf_20.obj'
    model_ml = nh.load_pickle_file(path)
    return model_ml
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
          
    
    
    
