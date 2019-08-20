'''
Created on Mar 25, 2018

This module trains/predicts NER using pre-trained word embeddings.
There are two optional datasets for training/testing. One is the CoNLL sample text given with this project.
Another is the GMB dataset ( split  into 80 training, 20 test). This can be downloaded from:
    http://gmb.let.rug.nl/data.php
The version used is 2.2.0 of the dataset. Because the downloaded dataset is large (803 MB) we have
a pre-processing script (Proprocess_GMB_Data.py) which loads only the tokens we are interested in
and stores them in a pickle file (gmb_dataset.joblib). We will NOT include the original dataset
with the submission as we include the pickle file. The script (Preprocess_GMB_Data) does NOT need
to be run.

For the pre-trained word vectors, the Google News Negative 300 are used. 

Please note that this module requires many python libraries:
- keras
- tensor flow
- h5py (save model)
- scikit learn
- scipy
- numpy
As well as their dependencies.

'''
import sys
import warnings
from optparse import OptionParser
from timeit import default_timer as timer

from gensim.models.keyedvectors import KeyedVectors
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.models import Sequential, load_model as keras_load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import numpy as np
from sklearn.metrics.classification import precision_score, recall_score

# RUNTIME PARAMS
hidden_size = 64
batch_size = 32
num_epochs = 5

warnings.filterwarnings("ignore")

# These are the standard labels we are dealing with. The GMB dataset uses different labels but we convert to these.
labels =  ['B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'O']
label2ind = {label: (index + 1) for index, label in enumerate(labels)}
ind2label = {(index + 1): label for index, label in enumerate(labels)}
max_label = max(label2ind.values()) + 1

def load_word2vec():
    '''
    Loads the word2vec pretrained model we are using.
    '''
    start = timer()
    print('Loading the word2vec pre-trained word embeddings. This may take a minute...')
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print('It took %3.2f s to load the word2vec embeddings' % (timer() - start))

    # Prepare the word map and embedding matrix.
    # We add a zero vector for 'unknown' since the gemsim model does not have one.
    word_map = {}
    for i, w in enumerate(model.index2word):
        word_map[w] = i+1 # We leave 0 to the UNKNOWN words
    max_features = len(word_map) + 1
    embedding_matrix = np.zeros((max_features, model.vector_size), dtype = 'float32')
    embedding_matrix[1:, :] = model.vectors 
    
    return word_map, embedding_matrix

# init global vars
word_map = None
embedding_matrix = None

def read_data(use_gmb):    
    print('Reading training data from file')
    if use_gmb:
        return read_gmb()
    else:
        return read_conll()
    
def read_gmb():
    dmp = joblib.load('gmb_dataset.joblib')
    return dmp

def read_conll():
    def read_conll_file(file_name):
        raw = open(file_name, 'r').readlines()
        all_x = []
        point = []
        for line in raw:
            stripped_line = line.strip().split(' ')
            point.append(stripped_line)
            if line == '\n':
                all_x.append(point[:-1])
                point = []
        all_x = all_x[:-1]
    #     short_x = [x for x in all_x if len(x) < 300]
        
        X = [[c[0] for c in x] for x in all_x]
        y = [[c[3] for c in y] for y in all_x]
        return X,y
    X_train, y_train = read_conll_file('train.txt')
    X_test, y_test = read_conll_file('test.txt')
    return X_train, X_test, y_train, y_test

def encode_label(y, n):
    result = np.zeros(n)
    result[y] = 1
    return result

def decode_label(y):
    return np.argmax(y, axis=1)
    
    
def word_index(w):
    if w in word_map:
        return word_map[w]
    else:
        return 0
    
def encode_sentence(sentence):
    if type(sentence) == str: 
        sentence = sentence.split()
    return [word_index(w) for w in sentence]
    
def train_model(X_enc, y_enc, verbose = False, show_confusion = False):
    '''
    The training data should be encoded already.
    '''
    out_size = len(label2ind) + 1
    
    # Create the Keras Bi-LSTM model
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], mask_zero=True, weights = [embedding_matrix], trainable = False))
    # We have to tell LSTM layer to 'return sequences'.
    model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
    model.add(TimeDistributed(Dense(out_size)))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X_enc, y_enc, batch_size=batch_size, epochs=num_epochs, validation_split=0.1, verbose=verbose)
    show_metrics(model, X_enc, y_enc, show_confusion)
    return model

    
def encode_xy(X,y):
    maxlen = max([len(x) for x in X])
    X_enc = [[word_index(c) for c in x] for x in X]
    y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
    y_enc = [[encode_label(c, max_label) for c in ey] for ey in y_enc]
    
    X_enc = pad_sequences(X_enc, maxlen=maxlen)
    y_enc = pad_sequences(y_enc, maxlen=maxlen)
    return X_enc, y_enc

def decode_results(yh, pr):
    '''
    This methods takes the encoded ground truth and prediction and converts
    them to a one dim array of label indexes for use by the metrics methods.
    '''
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

def show_metrics(model, X_enc,y_enc, show_confusion = False):
    pr = model.predict_classes(X_enc)
    yh = y_enc.argmax(2)
    fyh, fpr = decode_results(yh, pr)
    print ('Accuracy:', accuracy_score(fyh, fpr))
    print('F1:', f1_score(fyh, fpr, average='weighted'))
    print('Precision (per class: %s)' % labels)
    print(precision_score(fyh, fpr, average = None))
    print('Recall (per class: %s)' % labels)
    print(recall_score(fyh, fpr, average = None))

    if show_confusion:
        print ('Confusion matrix:')
        print (confusion_matrix(fyh, fpr))

def get_model_file_name(use_gmb):
    if use_gmb:
        filename = 'ner_gmb.model.z'
    else:
        filename = 'ner.model.z'
    return filename

def save_model(model, use_gmb):
#     d = {'model' : model}
#     joblib.dump(d, get_model_file_name(use_gmb))
    model.save(get_model_file_name(use_gmb))

def load_model(use_gmb):
    return keras_load_model(get_model_file_name(use_gmb))
#     try:
#         d = joblib.load(get_model_file_name(use_gmb))
#     except:
#         print('Could not load model from file', get_model_file_name(use_gmb))
#         sys.exit(-1)
#     return d['model']

if __name__ == '__main__':
    usage = """
    Usage: Tagger can be used to do one of three things: train, test or interactive. It uses the CoNLL by default,
    but using the -g option will tell it to use the GMB dataset. 
    Usage scenarios:
    1) Tagger train [-g] [-c]: This command trains our bidirectional lsmt model against the Conll (or GMB if -g option is used)
        and saves the resulting model in ner.model (or ner_gmb.model if -g option used.) It will also test the model and show
        the accuracy and F1. If -c option is used this will also show the confusion matrix.
    2) Tagger test [-g] [-c]: This command tests our saved model ner.model (or ner_gmb.model if -g option used) against the
        test data. If CoNLL dataset is used the test.txt file is used. For the GMB, the test dataset is stored in the same 
        pickle file as the training dataset. The command shows the accuracy and F1 scores. Also it shows confusion matrix
        if -c option is used.
    3) Tagger interactive [-g]: This loads the ner.model (or ner_gmb.model if -g option is used) and allows the user to 
        enter sentences. It tags the sentences using the model and shows our predicted tags.
    """
  
    parser = OptionParser(usage)
    parser.add_option("-g", "--gmb", default = False, 
        action = "store_true", dest = "use_gmb", help = 'Use GMB dataset. Default is false')
    parser.add_option("-c", "--confusion", default = False, 
        action = "store_true", dest = "show_confusion", help = 'Show the confusion matrix in the metrics')
    parser.add_option("-v", "--verbose", default = False, 
        action = "store_true", dest = "verbose")
      
    options, args = parser.parse_args()
  
    if len(args) < 1:
        print('At least 1 command line option is expected.')
        print(usage)
        sys.exit(-1)
  
    cmd = args[0].lower().strip()
    
    if cmd not in ['train', 'test', 'interactive']:
        print('Command %s not recognized.' % cmd)
        print(usage)
        sys.exit(-1)
    
    word_map, embedding_matrix = load_word2vec()
    
    if cmd == 'train':
        X_train, X_test, y_train, y_test = read_data(options.use_gmb)
        print('Encoding training data')
        X_train_enc, y_train_enc = encode_xy(X_train, y_train)
        print('\n\nTraining the model. This will take sometime. Grab a coffee')
        start = timer()
        model = train_model(X_train_enc, y_train_enc, options.verbose, options.show_confusion)
        print('It took %3.2f s to train the model' % (timer() - start))

        print('\n\nDone training. Now saving the model to file. This may take a minute or two')
        start = timer()
        save_model(model, options.use_gmb)
        print('It took %3.2f s to save the model' % (timer() - start))
        print('Done saving. Now will validate model')
        X_test_enc, y_test_enc = encode_xy(X_test, y_test)
        show_metrics(model, X_test_enc, y_test_enc, options.show_confusion)
    elif cmd == 'test':
        print('Loading saved model. That may take a couple of minutes')
        start = timer()
        model = load_model(options.use_gmb)        
        print('It took %3.2f s to load the model from file' % (timer() - start))

        print('Loading data from file')
        X_train, X_test, y_train, y_test = read_data(options.use_gmb)
        print('Encoding test data')
        X_test_enc, y_test_enc = encode_xy(X_test, y_test)
        print('Running metrics')
        show_metrics(model, X_test_enc, y_test_enc, options.show_confusion)

    elif cmd == 'interactive':
        print('Loading saved model. That may take a couple of minutes')
        start = timer()
        model = load_model(options.use_gmb)        
        print('It took %3.2f s to load the model from file' % (timer() - start))
        while True:
            sentence = input('Enter a sentence to NER (or quit to exit): ')
            if sentence == 'quit':
                break
            sentence = " '".join(sentence.split("'"))
            p = model.predict(np.array([encode_sentence(sentence)]))
            p = np.argmax(p[0], axis=1)
            p = [ind2label[i] for i in p]
            print(p)

    
