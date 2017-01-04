"""
Takes two inputs and train a lstm on each and then apply a merge layer and then two dense layers and
classifies them
"""
from collections import defaultdict
from keras.layers import merge, Dense, Dropout, Flatten, LSTM, Input, Lambda, Masking
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras import backend as K
import itertools as it
import numpy as np
np.random.seed(1337)  # for reproducibility
import codecs, sys
from sklearn import metrics
from keras.regularizers import l2, activity_l2
import matplotlib.pyplot as plt

unique_chars = []
languages = []
max_word_len = 10
nb_filter = 16
filter_length = 2
nb_epoch = 20
batch_size = 128
tr_threshold = 0.7

def wrd_to_1d(w):
    onehot = []
    for x in w:
        temp = len(unique_chars)*[0]
        if x == "0":
            onehot += temp
        else:
            idx = unique_chars.index(x)
            temp[idx] = 1
            onehot += temp
    return np.array(onehot)

def wrd_to_2d(w):
    w2d = []
    for x in w:
        temp = len(unique_chars)*[0]
        if x == "0":
            w2d.append(temp)
        else:
            idx = unique_chars.index(x)
            temp[idx] = 1
            w2d.append(temp)
    return np.array(w2d).T

def wrd_to_phonetic(w, d):
    w2d = []
    for x in w:
        temp = 16*[0]
        if x == "0":
            w2d.append(temp)
        else:
            temp = d[x]
            w2d.append(temp)
    return np.array(w2d).T

def make_pairs(d):
    tr_labels = []
    tr_pairs = []
    tr_lang_pairs = []
    te_labels = []
    te_pairs = []
    te_langs_pairs = []
    concepts = d.keys()
    n_concepts = len(concepts)
    print "No. of concepts %d" %(n_concepts)
    len_concepts = int(np.ceil(tr_threshold*n_concepts))
    len_concepts = int(sys.argv[1])
    tr_concepts = concepts[:len_concepts]
    #tr_concepts = concepts[:20]
#    len_concepts = int(np.floor(.3*n_concepts))
#    len_concepts = n_concepts-len_concepts
    len_concepts = n_concepts-len_concepts
    te_concepts = concepts[-len_concepts:]
    print "No. of training concepts %d testing concepts %d" %(len(tr_concepts),len(te_concepts))
    for concept in tr_concepts:
        for i1, i2 in it.combinations(d[concept], r=2):
            w1, g1, l1 = i1
            w2, g2, l2 = i2
            if l1 != l2:
                tr_pairs.append((w1,w2))
#                tr_lang_pairs.append((l1, l2))
                if g1 == g2:
                    tr_labels.append(1)
                else:
                    tr_labels.append(0)
    for concept in te_concepts:
        for i1, i2 in it.combinations(d[concept], r=2):
            w1, g1, l1 = i1
            w2, g2, l2 = i2
            if l1 != l2:
                te_pairs.append((w1,w2))
#                te_lang_pairs.append((l1, l2))
                if g1 == g2:
                    te_labels.append(1)
                else:
                    te_labels.append(0)


    return (tr_pairs, tr_labels, te_pairs, te_labels)

data_file = sys.argv[2]

d = defaultdict(list)
#f = codecs.open("/home/taraka/list_length_project/sets/CognateData/output/Mixe-Zoque.tsv","r", encoding="utf-8")
f = codecs.open(data_file,"r", encoding="utf-8")
#f = codecs.open("/home/taraka/list_length_project/sets/IELex/output/IELex-2016.tsv.asjp","r", encoding="utf-8")
#f = codecs.open("/home/taraka/list_length_project/sets/abvd2/output/abvd2-part2.tsv.asjp","r", encoding="utf-8")
f.readline()
for line in f:
    line = line.strip()
    arr = line.split("\t")
    lang = arr[0]
    if lang not in languages:
        languages.append(lang)
    concept = arr[3]
    cogid = arr[6]
    cogid = cogid.replace("-","")
    cogid = cogid.replace("?","")
    asjp_word = arr[5].split(",")[0]
    asjp_word = asjp_word.replace(" ", "")
    #tokenized_word = ipa2tokens(asjp_word)
    #asjp_word = "".join(tokens2class(tokenized_word, 'asjp'))
    asjp_word = asjp_word.replace("%","")
    asjp_word = asjp_word.replace("~","")
    asjp_word = asjp_word.replace("*","")
    asjp_word = asjp_word.replace("$","")
    asjp_word = asjp_word.replace("K","k")
    asjp_word = asjp_word.replace("\"","")
    if len(asjp_word) < 1:
        continue
    for x in asjp_word:
        if x not in unique_chars:
            unique_chars.append(x)
    if len(asjp_word) > max_word_len:
        #print "Exceeded maximum word length %s ",word 
        asjp_word = asjp_word[:max_word_len]
    else:
        asjp_word = asjp_word.center(max_word_len,"0")
    d[concept].append((asjp_word, cogid, lang))
    
f.close()

print unique_chars

phonetic_features = defaultdict()

f = open("my_phonetic_features_without_vowel.txt.csv", "r")
header = f.readline()
for line in f:
    line = line.replace("\n","")
    arr = line.split("\t")
    character, bin_str = arr[0], arr[1:]
    bin_vec = [int(x) for x in arr[1:]]
    phonetic_features[character] = bin_vec
f.close()

n_dim = len(header.split("\t")) - 1
n_dim = 16
train_pairs, train_labels, test_pairs, test_labels = make_pairs(d)

train_1 = []
train_2 = []
test_1 = []
test_2 = []

for p1, p2 in train_pairs:
    onehotp1, onehotp2 = wrd_to_phonetic(p1, phonetic_features), wrd_to_phonetic(p2, phonetic_features)
    train_1.append(onehotp1)
    train_2.append(onehotp2)

for p1, p2 in test_pairs:
    onehotp1, onehotp2 = wrd_to_phonetic(p1, phonetic_features), wrd_to_phonetic(p2, phonetic_features)
    test_1.append(onehotp1)
    test_2.append(onehotp2)

train_1 = np.array(train_1)
train_2 = np.array(train_2)

test_1 = np.array(test_1)
test_2 = np.array(test_2)

in_dim = max_word_len*len(unique_chars)

print train_1.shape, train_2.shape

print "Random labeling training accuracy %f" %(1.0-np.mean(train_labels))
print "Random labeling test accuracy %f" %(1.0-np.mean(test_labels))
print train_1.shape, train_2.shape
print test_1.shape, test_2.shape

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

train_1 = train_1.reshape(train_1.shape[0], 1, n_dim, max_word_len)
train_2 = train_2.reshape(train_2.shape[0], 1, n_dim, max_word_len)
test_1 = test_1.reshape(test_1.shape[0], 1, n_dim, max_word_len)
test_2 = test_2.reshape(test_2.shape[0], 1, n_dim, max_word_len)

train_1 = train_1.astype('float32')
train_2 = train_2.astype('float32')
test_1 = test_1.astype('float32')
test_2 = test_2.astype('float32')

word_1 = Input(shape=(1, n_dim, max_word_len))
word_2 = Input(shape=(1, n_dim, max_word_len))
word_input = Input(shape=(1, n_dim, max_word_len))

#, activation="relu"
x = Convolution2D(10, 2, 3, input_shape = (1, n_dim, max_word_len))(word_input)
x = Convolution2D(10, 2, 3)(x)
x = MaxPooling2D(pool_size=(1, 2))(x)
#x = Dropout(0.25)(x)
out = Flatten()(x)

word_model = Model(word_input, out)

encoded_1 = word_model(word_1)
encoded_2 = word_model(word_2)
print(word_model.get_output_shape_at(0))

merged_vector = merge([encoded_1, encoded_2],  mode=lambda x: abs(x[0]-x[1]), output_shape=lambda x: x[0])
#merged_vector = merge([encoded_1, encoded_2],  mode=lambda x: abs(x[0]-x[1]), output_shape=word_model.get_output_shape_at(0))
predictions = Dense(8, activation='relu')(merged_vector)
predictions = Dropout(.5)(predictions) 
predictions = Dense(1, activation='sigmoid')(predictions) 
model = Model(input=[word_1, word_2], output=predictions)
model.summary()

#sgd = SGD(lr=.01, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(optimizer="adadelta",
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([train_1, train_2], train_labels, nb_epoch=nb_epoch, batch_size=batch_size, validation_data=([test_1, test_2], test_labels), callbacks=[early_stopping])

tr_score = model.predict([train_1, train_2], verbose=1)
te_score = model.predict([test_1, test_2], verbose=1)
print("Average Precision Score %s " %(metrics.average_precision_score(test_labels, te_score, average="micro")))

#plt.plot(te_score, test_labels)
#plt.show()

c = tr_score > 0.5
b = te_score > 0.5
tr_pred = c.astype('int')
te_pred = b.astype('int')

print("Training")
print(metrics.classification_report(train_labels, tr_pred, digits=3))
print("Testing")
print(metrics.classification_report(test_labels, te_pred, digits=3))
print("Testing Accuracy")
print(metrics.accuracy_score(test_labels, te_pred))





