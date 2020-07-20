import numpy as np
import torch
import re
from torchtext import data
import pandas as pd
import string
import os
import nltk
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from torchtext.vocab import Vectors, GloVe

def remove_unnecessary(text):
    #remove_URL
    url = re.compile(r'https?://\S+|www\.\S+')
    text = url.sub('', text)

    #remove_html
    html = re.compile(r'<.*?>')
    text = html.sub('', text)



    #remove @
    text = re.sub('@[^\s]+','',text)

    #remove_emoji
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)


    #Removes integers 
    text = ''.join([i for i in text if not i.isdigit()])         
    
    #remove_punct
    table = str.maketrans('', '', string.punctuation)
    text = text.translate(table)

    #Replaces contractions from a string to their equivalents 
    contraction_patterns = [(r'won\'t', 'will not'), (r'can\'t', 'cannot'), (r'i\'m', 'i am'), 
                            (r'ain\'t', 'is not'), (r'(\w+)\'ll', '\g<1> will'), (r'(\w+)n\'t', '\g<1> not'),
                            (r'(\w+)\'ve', '\g<1> have'), (r'(\w+)\'s', '\g<1> is'), (r'(\w+)\'re', '\g<1> are'),
                            (r'(\w+)\'d', '\g<1> would'), (r'&', 'and'), (r'dammit', 'damn it'), 
                            (r'dont', 'do not'), (r'wont', 'will not')]
    
    patterns = [(re.compile(regex), repl) for (regex, repl) in contraction_patterns]
    for (pattern, repl) in patterns:
        text, _= re.subn(pattern, repl, text)



    #lemmatize_sentence
    sentence_words = text.split(' ')
    new_sentence_words = list()
    
    for sentence_word in sentence_words:
        sentence_word = sentence_word.replace('#', '')
        new_sentence_word = WordNetLemmatizer().lemmatize(sentence_word.lower(), wordnet.VERB)
        new_sentence_words.append(new_sentence_word)
        
    new_sentence = ' '.join(new_sentence_words)
    new_sentence = new_sentence.strip()

    return new_sentence


def prepare_csv(df_train, seed=27, val_ratio=0.2):
    idx = np.arange(df_train.shape[0])    
    np.random.shuffle(idx)
    
    val_size = int(len(idx) * val_ratio)
    if not os.path.exists('cache'): # cache is tem memory file 
        os.makedirs('cache')
    
    df_train.iloc[idx[val_size:], :][['id', 'target', 'text']].to_csv(
        'cache/dataset_train.csv', index=False
    )
    
    df_train.iloc[idx[:val_size], :][['id', 'target', 'text']].to_csv(
        'cache/dataset_val.csv', index=False
    )
    
    
    
def get_iterator(dataset, batch_size, train=True,
                 shuffle=True, repeat=False, device=None): 
    dataset_iter = data.Iterator(
        dataset, batch_size=batch_size, device=device,
        train=train, shuffle=shuffle, repeat=repeat,
        sort=False)  
    return dataset_iter

def get_dataset(fix_length=25, lower=False, vectors=None,train_dir = 'train.csv', batch_size=1, device=None): 
    
    train = pd.read_csv(train_dir,error_bad_lines=False)
    train['text'] = train['text'].apply(lambda x: remove_unnecessary(x))
    
    if vectors is not None:
        lower=True

    prepare_csv(train)
    
    TEXT = data.Field(sequential=True, 

                      lower=True, 
                      include_lengths=True, 
                      batch_first=True, 
                      fix_length=fix_length)
    LABEL = data.Field(use_vocab=True,
                       sequential=False,
                       dtype=torch.float16)
    ID = data.Field(use_vocab=False,
                    sequential=False,
                    dtype=torch.float16)   
    train_temp, val_temp = data.TabularDataset.splits(
        path='cache/', format='csv', skip_header=True,
        train='dataset_train.csv', validation='dataset_val.csv',
        fields=[
            ('id', ID),
            ('target', LABEL),
            ('text', TEXT)])  
  
    
  
    TEXT.build_vocab(
        train_temp, val_temp,
        max_size=20000,
        min_freq=10,
        vectors=GloVe(name='6B', dim=300)  
    )
    LABEL.build_vocab(
        train_temp
    )
    ID.build_vocab(
        train_temp, val_temp, 
    )
    
    word_embeddings = TEXT.vocab.vectors
    vocab_size = len(TEXT.vocab)
    
    train_iter = get_iterator(train_temp, batch_size=batch_size, 
                              train=True, shuffle=True,
                              repeat=False,device=device)
    val_iter = get_iterator(val_temp, batch_size=batch_size, 
                            train=True, shuffle=True,
                            repeat=False, device=device)

    print('Train samples:%d'%(len(train_temp)), 'Valid samples:%d'%(len(val_temp)),'Train minibatch nb:%d'%(len(train_iter)),
          'Valid minibatch nb:%d'%(len(val_iter)))
    return vocab_size, word_embeddings, train_iter, val_iter

