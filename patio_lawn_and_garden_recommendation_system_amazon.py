# import libraries
import json
import numpy as np
from collections import defaultdict
import warnings
from keras.utils.vis_utils import plot_model
import gensim
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import keras
import gzip
from keras import backend as K
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, Concatenate, Multiply, Lambda, Reshape
from keras.layers.core import Dropout
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.callbacks import EarlyStopping, ModelCheckpoint

"""## Display the dataset"""

# Convert data from gz to DataFrame


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


"""## Download The Review and Meta Data"""

# # Download Review Data
# !wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Patio_Lawn_and_Garden.json.gz
#
# # Download Meta Data
# !wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Patio_Lawn_and_Garden.json.gz

# display the meta data as  dataframe
metadata = getDF('Data/Gardan_recom/meta_Patio_Lawn_and_Garden.json.gz')
metadata.head()
print(metadata.head())

# display the review data as  dataframe
reviews = getDF('Data/Gardan_recom/reviews_Patio_Lawn_and_Garden.json.gz')
reviews.head()
print(reviews.head())

# # extract review baby file
# !gunzip -k reviews_Patio_Lawn_and_Garden.json.gz
#
# # extract meta baby file
# !gunzip -k meta_Patio_Lawn_and_Garden.json.gz

"""## Preprocessing"""

# Hyperparameter
dataName = "Patio_Lawn_and_Garden"# name of your product
vector_size = 100  # vector size of reviewtext and item description
epoch_num = 100  # number of vector
reviews_data = []  # review list to insert review data inside
meta_data = []  # meta list to insert meta data inside
num_ui_link = 20  # the number of each user link to items
num_iu_link = 0  # the number of each item link to user

# user -> item  ->  for text
ui_dict = defaultdict(list)
# item -> user - >  for text
iu_dict = defaultdict(list)
# rating training data list
reviews_train_data = []
# rating testing data list
reviews_test_data = []
# max number of item
max_num_item = 1
# max number of users
max_num_user = 1
# item -> user -> for rating
iu_dict2 = defaultdict(list)
# user -> item  ->  for rating
ui_dict2 = defaultdict(list)
len_ui_dict = {}
split_ratio = 4  # ratio to split between test and train
epochs = 20

# Append review.json to  review_data list

with open('Data/Gardan_recom/reviews_' + dataName + '.json') as f:
    for line in f:
        reviews_data.append(json.loads(line))
    f.close()

# reviews_data[:1]

"""```
## Output

[{'reviewerID': 'A28O3NP6WR5517',
  'asin': '0188399313',
  'reviewerName': 'Jennifer gymer',
  'helpful': [0, 0],
  'reviewText': 'They work very well. Easy to clean, we wash them in the dishwasher every day. Our LO loves to hold on to the bottle and the plastic covering makes it easy for her to hold on to.',
  'overall': 5.0,
  'summary': 'These bottles are great!',

'unixReviewTime': 1369612800,
  'reviewTime': '05 27, 2013'}]
```
"""

# Append meta.json to meta_data list
with open('Data/Gardan_recom/meta_' + dataName + '.json') as f:
    for line in f:
        line_dict = json.dumps(eval(line))
        meta_data.append(json.loads(line_dict))
    f.close()

# meta_data[:1]

"""```
[{'asin': '0188399313',
  'categories': [['Baby']],
  'description': 'Wee-Go Glass baby bottles by LifeFactory (Babylife) are designed to grow with your child. The included clear cover can also serve as an easy to hold cup. Twist on the solid cap (sold separately) and use your bottles for storing juice or snacks. Perfect for a lunchbox or traveling. The bright colored silicone sleeve (patent pending) helps to protect the bottle from breakage and provides a great gripping surface and tactile experience during feeding. The bottle and sleeve can be boiled or put in the dishwasher together. They can also go in the freezer, making breast milk storage simple.',
  'title': 'Lifefactory 4oz BPA Free Glass Baby Bottles - 4-pack-raspberry and Lilac',
  'price': 69.99,
  'imUrl': 'http://ecx.images-amazon.com/images/I/41SwthpdD9L._SX300_.jpg',
  'brand': 'Lifefactory',
  'related': {'also_bought': ['B002SG7K7A',
    'B003CJSXW8',
    'B004PW4186',
    'B002O3JH9Q',
    'B002O3NLIO',
    'B004HGSU28'],
   'also_viewed': ['B003CJSXW8',
    'B0052QOL1Q',
    'B004PW4186',
    'B00EN0OLZ8',
    'B00EN0OOQY',
    'B0049YS46K',
    'B00E64CBLM',
    'B00F9YOOS6',
    'B00AH9RPVQ',
    'B00BCU2R7G',
    'B002O3NLIO',
    'B008NZ4X2K',
    'B005NIDFEW',
    'B00DKPJCH4',
    'B00CZNGWWK',
    'B00DAKJIQ4',
    'B005CT55IQ',
    'B0049YRJM0',
    'B0071IEWD0',
    'B00E64CA68',
    'B00IUB3SKK',
    'B00A7AA6XY',
    'B001F50FFE',
    'B002HU9EO4',
    'B007HP11SQ',
    'B009WPUMX4',
    'B002O3JH9Q',
    'B00F2FT3K6',
    'B00I5CR35A',
    'B00BCTY5EK',
    'B002SG7K7A',
    'B00F2FLU2U',
    'B0062ZK0GQ',
    'B002UOFR66',
    'B0055LKQQ2',
    'B00A0FGN8I',
    'B00HMYCG2W',
    'B00DHFLUO0',
    'B0040HMPA2',
    'B00I5CT9XE',
    'B008B5MMNO',
    'B00BQYVNGO',
    'B00925WM28',
    'B00BGKC3EY',
    'B005Q3LSDO',
    'B0038JDVCE',
    'B0045I6IA4'],
   'bought_together': ['B002SG7K7A', 'B003CJSXW8'],
   'buy_after_viewing': ['B003CJSXW8',
    'B0052QOL1Q',
    'B004PW4186',
    'B002SG7K7A']}}]
   ```

"""

# fill user to item and item to user dictionaries
for line_data in reviews_data:
    user_id = line_data["reviewerID"]  # extract review_id from each review in reviews_data list
    item_id = line_data["asin"]  # extract item_id from each review in reviews_data list
    ui_dict[user_id].append(item_id)  # assign review_id  to  item_id in ui_dict
    iu_dict[item_id].append(user_id)  # assign item_id  to review_id   in iu_dict
# print lenght for the two dictionaries
print("len(ui_dict)=", len(ui_dict))
print("len(iu_dict)=", len(iu_dict))

# ui_dict >>  ex >> 'A28O3NP6WR5517': ['0188399313', 'B004I110D8', 'B006ZZSME0']

# iu_dict >>  ex >> '0188399313': ['A28O3NP6WR5517']

# convert review json file to txt file 
ex_data = open( 'Data/Gardan_recom/reviews_' + dataName + '.txt', "w")
for line_dict in reviews_data:
    user_id = line_dict["reviewerID"]  # extract review_id from each review in reviews_data list
    item_id = line_dict["asin"]  # extract item_id from each review in reviews_data list
    ui_num = len(ui_dict[user_id])  # lenght of the users for this item which gave review for
    iu_num = len(iu_dict[item_id])  # lenght of  items for each user
    if ui_num >= num_ui_link and iu_num >= num_iu_link:
        # check if they are greater or eual to numbers which we have mentioned before  in hyperparamter cell
        # so we could get best performance for recommendation system before we insert to text file
        ex_data.writelines(str(line_dict) + "\n")
        iu_dict[item_id].append(user_id)
ex_data.close()

# convert meta json file to txt file 
ex_data = open('Data/Gardan_recom/meta_' + dataName + '.txt', "w")
for line_dict in meta_data:
    item_id = line_dict["asin"]  # extract item_id from each review in reviews_data list
    if item_id in iu_dict.keys():
        iu_num = len(iu_dict[item_id])  # lenght of  items for each user
        if iu_num >= num_iu_link:
            # check if they are greater or eual to numbers which we have mentioned before  in hyperparamter cell
            # so we could get best performance for recommendation system before we insert to text file
            ex_data.writelines(str(line_dict) + "\n")
ex_data.close()

TaggededDocument = gensim.models.doc2vec.TaggedDocument  # import  the pretrained model to turn the text to vector
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


# turn the txt file to vector >>  stage 1  in research paper
class DoctoVec:
    # turn dictionary data to list with counter
    # give each paragraph an id_number (count)  and append in text file
    # so we could identify each paragraph and be ready for vectoriztion
    def get_data(dict_d):
        x_train = []
        count = 0
        for (i, text) in dict_d.items():
            word_list = text.split(' ')
            l = len(word_list)
            word_list[l - 1] = word_list[l - 1].strip()
            document = TaggededDocument(word_list, tags=[i])
            x_train.append(document)
            count += 1
        return x_train, count
   
    # train model to vector
    def train(dataName, x_train, vector_size, epoch_num,kind):
        model_dm = Doc2Vec(x_train, min_count=1, window=3, vector_size=vector_size, sample=1e-3, negative=5, workers=4)
        model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=epoch_num)
        # train model to turn it to vectors using pretrained model
        model_dm.save('Data/Gardan_recom/' + dataName + '_Model_'+ kind+"_"+ str(vector_size))
        # saving model after we turn it to vector so we could use it in future
        return model_dm

    # save vectors
    # saving the vector for any text we will give to our model
    def saveVector(dataName, model_dm, v_size, count,kind):
        out = open('Data/Gardan_recom/' + dataName + '.'+kind, "w", encoding='utf-8')
        for num in range(0, count):
            doc_vec = model_dm.docvecs[num]
            vec_list = str(num) + ","
            for i_doc in doc_vec:
                vec_list = vec_list + str(i_doc) + ","
            out.writelines(vec_list[:-1] + "\n")
        out.close()

# read review txt file


reviews_data = []
with open('Data/Gardan_recom/reviews_' + dataName + '.txt') as f:
    for line in f:
        line_dict = json.dumps(eval(line))
        reviews_data.append(json.loads(line_dict))
    f.close()
    
# give unique number for each product and for each user and append the in two different dictioary
asin2itemNum = {}
reviewerID2userNum = {}
num = 1
for ui in reviews_data:
    if ui["asin"] not in asin2itemNum:
        asin2itemNum[ui["asin"]] = num
        num += 1
num = 1
for uu in reviews_data:
    if uu["reviewerID"] not in reviewerID2userNum:
        reviewerID2userNum[uu["reviewerID"]] = num
        num += 1

# reviewerID2userNum

# loading the dictionary of UserId_ItemID_Num


def loading_metadata():
    data = []
    dict_d = {}
    with open('Data/Gardan_recom/meta_' + dataName + '.txt') as f:
        for line in f:
            line_dict = json.dumps(eval(line))
            data.append(json.loads(line_dict))
        for d_item in data:
            k = asin2itemNum.get(d_item["asin"])
            if "description" in d_item.keys():
                dict_d[k] = d_item["description"].replace("\n", "")
            else:
                dict_d[k] = ""
        return dict_d

# turn item text to a vedcor with size of 100  and save it


dict_i = loading_metadata()
x_train_item, count_item = DoctoVec.get_data(dict_i)
model_dm = DoctoVec.train(dataName, x_train_item, vector_size, epoch_num,"item")
DoctoVec.saveVector(dataName, model_dm, vector_size, count_item,"item")
print("Item Vector Finished")


def loading_reviewdata():
    # loading the dictionary of UserId_ItemID_Num
    data = []
    dict_d = {}
    with open('Data/Gardan_recom/reviews_' + dataName + '.txt') as f:
        for line in f:
            line_dict = json.dumps(eval(line))
            data.append(json.loads(line_dict))
        for d_item in data:
            k = reviewerID2userNum.get(d_item["reviewerID"])
            # print(k)
            if "reviewText" in d_item.keys():
                dict_d[k] = d_item["reviewText"].replace("\n", "")
            else:
                dict_d[k] = ""
            # print(dict_d[k])
        return dict_d


"""
```
#output example
1
I ended up with a variety of different brands of cotton flannel baby wipes.  These are good quality, just not my personal favorite.  However, if the colors really do it for you, go ahead and stock up on these, they're good.  And really, we're talking squares of flannel, it's not critical.  For what it's worth, I like BumGenius wipes best for absorbency - like wiping up spit-up.  They're thick and soft.  I like OsoCozy best for cleaning baby bums, because they're thin, the better to clean crevices with.  ALL the wipes I bought shrank a lot in the first wash (hot), so expect that.I avoided GroVia for being synthetic, but in retrospect that might have been the way to go.  I have all BumGenius 4.0 diapers, which shouldn't be washed with natural fibers.  Now I have to wash the cotton wipes with baby's clothes, rather than with the diapers, which isn't ideal.
```"""

# turn review text to a vedcor with size of 100  and save it
dict_u = loading_reviewdata()
x_train_user, count_user = DoctoVec.get_data(dict_u)
model_dm = DoctoVec.train(dataName, x_train_user, vector_size, epoch_num, "user")
DoctoVec.saveVector(dataName, model_dm, vector_size, count_user, "user")
print("User Vector Finished")

"""## User Rating Input"""

# extract the rating for each review and assign them to the product_id with the review_id
for d in reviews_data:
    user_id = reviewerID2userNum.get(d["reviewerID"])
    item_id = asin2itemNum.get(d["asin"])
    if user_id not in ui_dict2:
        ui_dict2[user_id] = list()
        if int(user_id) > max_num_user: 
            max_num_user = int(user_id)
    ui_dict2[user_id].append(item_id)
    if item_id not in iu_dict2:
        iu_dict2[item_id] = list()
        if int(item_id) > max_num_item:
            max_num_item = int(item_id)
    iu_dict2[item_id].append(user_id)

for l_d in reviews_data:
    u_id = reviewerID2userNum.get(l_d["reviewerID"])
    len_ui_dict[u_id] = len(ui_dict2.get(u_id))
print("max_num_user =  ", max_num_user)
print("max_num_item = ", max_num_item)

# reviews_train_data vs reviews_test_data
# split our dataset to train and test for rating and append them to different lists
num = 0
for l_d in reviews_data:
    if num % (split_ratio + 1) < split_ratio:
        reviews_train_data.append(l_d)
        num += 1
    else:
        reviews_test_data.append(l_d)
        num += 1

# insert Rating in train file
train_list = []
train_data = open('Data/Gardan_recom/' + dataName + ".train.rating", "w")
for l_train in reviews_train_data:
        user_id = reviewerID2userNum.get(l_train["reviewerID"])
        item_id = asin2itemNum.get(l_train["asin"])
        rating_score = int(l_train["overall"])
        str_line = str(user_id) + "  " + str(item_id) + "  " + str(rating_score)
        train_data.writelines(str_line + "\n")
        train_list.append(str_line)
train_data.close()
print("Training Rating Finished")

print(train_list[:1])

# Insert Rating in test file
test_list = []
test_data = open('Data/Gardan_recom/' + dataName + ".test.rating", "w")
for l_test in reviews_test_data:
        user_id = reviewerID2userNum.get(l_test["reviewerID"])
        item_id = asin2itemNum.get(l_test["asin"])
        rating_score = int(l_test["overall"])
        str_line = str(user_id) + "  " + str(item_id) + "  " + str(rating_score)
        test_data.writelines(str_line + "\n")
        test_list.append(str_line)
test_data.close()
print("Test Rating Finished")

print(test_list[:1])
# Read .rating file and Return dok matrix (sparse matrix ).
# The first line of .rating file is: num_users > num_items > rating


def get_max_users_max_items(filename):
    # Get number of users and items
    num_users, num_items = 0, 0
    with open(filename, "r") as f:
            line = f.readline()
            # print(line)
            while line is not None and line != "":
                arr = line.split("  ")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
                # print(line)
                
    return num_users, num_items
# Construct  sparse matrix with size of maxmuim of users and items


def load_rating_file_as_matrix(filename):
        num_users, num_items = get_max_users_max_items(filename)
        # print(num_users , num_items)
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                arr = line.split("  ")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    mat[user, item] = rating
                line = f.readline()
        return mat
# get the  text vector for  user_review and item_description


def load_review_feature(filename):
        dict = {}
        with open(filename, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                fea = line.strip('\n').split(',')
                index = int(fea[0])
                if index not in dict:
                    dict[index] = fea[1:]
                line = f.readline()
        return dict


# get read all files we have saved in previous stage
path = 'Data/Gardan_recom/' + dataName
trainMatrix = load_rating_file_as_matrix(path + ".train.rating")
user_review_fea = load_review_feature(path + ".user")
item_review_fea = load_review_feature(path + ".item")
testRatings = load_rating_file_as_matrix(path + ".test.rating")
num_users, num_items = trainMatrix.shape

print(num_users, num_items)

"""## Build Model"""

# build model
# Intiliaze the model input
user_input = Input(shape=(1,), dtype='int32', name='user_input')
user_sent = Input(shape=(vector_size,), dtype='float32', name='user_sentiment')
item_input = Input(shape=(1,), dtype='int32', name='item_input')
item_cont = Input(shape=(vector_size,), dtype='float32', name='item_content')

# Embedding layer for user
Embedding_User = Embedding(input_dim=num_users, input_length=1, output_dim=vector_size, name='user_embedding')

# Embedding layer for item
Embedding_Item = Embedding(input_dim=num_items, input_length=1,  output_dim=vector_size, name='item_embedding')

# User Sentiment Dense Network


def user_Sentiment(user_latent, user_sent):
    latent_size = user_latent.shape[1]
    inputs = user_sent
    layer = Dense(latent_size, activation='relu', name='user_attention_layer')(inputs)
    sent = Lambda(lambda x: K.softmax(x), name='user_Sentiment_softmax')(layer)
    output = Multiply()([user_latent, sent])
    return output
# Item Content Dense Network


def item_Content(item_latent, item_cont):
    latent_size = item_latent.shape[1]
    inputs = item_cont
    layer = Dense(latent_size, activation='relu', name='item_attention_layer')(inputs)
    cont = Lambda(lambda x: K.softmax(x), name='item_Content_softmax')(layer)
    output = Multiply()([item_latent, cont])
    return output
# Crucial to flatten an embedding vector


user_latent = Reshape((vector_size,))(Flatten()(Embedding_User(user_input)))
item_latent = Reshape((vector_size,))(Flatten()(Embedding_Item(item_input)))
user_latent_atten = user_Sentiment(user_latent, user_sent)
item_latent_atten = item_Content(item_latent, item_cont)

user_latent = Dense(vector_size, activation='relu')(user_latent_atten)
item_latent = Dense(vector_size, activation='relu')(item_latent_atten)

# review-based attention calculation
vec = Multiply()([user_latent, item_latent])
user_item_concat = Concatenate()([user_sent, item_cont, user_latent, item_latent])
att = Dense(vector_size, kernel_initializer='random_uniform', activation='softmax')(user_item_concat)

# Element-wise product of user and item embeddings 
predict_vec = Multiply()([vec, att])
# Final prediction layer
prediction = Dense(vector_size, activation='relu')(predict_vec)
# for overfitting
prediction = Dropout(0.5)(prediction)
prediction = Dense(1, name='prediction')(prediction)

model = Model(inputs=[user_input, user_sent, item_input, item_cont], outputs=prediction)

model.summary()

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# compile our model
model.compile(optimizer="adam", loss="mean_absolute_error", metrics=['mean_squared_error'])

# using callbacks to avoid overfitting and save best model during training
earlystopping = EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint('Data/Gardan_recom/Patio_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Make the data ready fro training 
# convert it to numpy array as we extract the labels


def get_instances(trainMatrix, user_review_fea, item_review_fea):
    user_input, user_fea, item_input, item_fea, labels = [], [], [], [], []
    num_users = trainMatrix.shape[0]
    for (u, i) in trainMatrix.keys():
        if u in user_review_fea.keys() and i in item_review_fea.keys():
            user_input.append(u)
            user_fea.append(user_review_fea[u])
            item_input.append(i)
            item_fea.append(item_review_fea[i])
            label = trainMatrix[u, i]
            labels.append(label)
    return np.array(user_input), np.array(user_fea, dtype='float32'), np.array(item_input),\
           np.array(item_fea, dtype='float32'), np.array(labels)

# user_input = np.save("npy_dat/gardan_npy/user_input",user_input)
# user_fea = np.save("npy_dat/gardan_npy/user_fea", user_fea)
# item_input = np.save("npy_dat/gardan_npy/item_input", item_input)
# item_fea = np.save("npy_dat/gardan_npy/item_fea", item_fea)
# labels = np.save("npy_dat/gardan_npy/labels", labels)

# user_input_test = np.save("npy_dat/gardan_npy/user_input_test",user_input_test)
# user_fea_test = np.save("npy_dat/gardan_npy/user_fea_test", user_fea_test)
# item_input_test = np.save("npy_dat/gardan_npy/item_input_test", item_input_test)
# item_fea_test = np.save("npy_dat/gardan_npy/item_fea_test", item_fea_test)
# test_label = np.save("npy_dat/gardan_npy/test_label", test_label)
# get instance of training dataset for train the moodel


user_input, user_fea, item_input, item_fea, labels = get_instances(trainMatrix, user_review_fea, item_review_fea)
user_input = np.load("npy_dat/gardan_npy/user_input.npy")
user_fea = np.load("npy_dat/gardan_npy/user_fea.npy")
item_input = np.load("npy_dat/gardan_npy/item_input.npy")
item_fea = np.load("npy_dat/gardan_npy/item_fea.npy")
labels = np.load("npy_dat/gardan_npy/labels.npy")

# fit our model
hist = model.fit([user_input, user_fea, item_input, item_fea], labels, batch_size=256, epochs=10, validation_split=0.2,
                 verbose=1, callbacks=[earlystopping, checkpoint])

# model.save("model.h5")

# Draw the progress during the training
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(1, len(loss) + 1)
# loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# get instance for testing data
user_input_test, user_fea_test, item_input_test, item_fea_test, test_label = get_instances(testRatings, user_review_fea, item_review_fea)
user_input_test = np.load("npy_dat/gardan_npy/user_input_test.npy")
user_fea_test = np.load("npy_dat/gardan_npy/user_fea_test.npy")
item_input_test = np.load("npy_dat/gardan_npy/item_input_test.npy")
item_fea_test = np.load("npy_dat/gardan_npy/item_fea_test.npy")
test_label = np.load("npy_dat/gardan_npy/test_label.npy")
# evaluate the test dataset
mean_absolute_error(model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test]), test_label)

mean_squared_error(model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test]),test_label)

model.predict([user_input_test, user_fea_test, item_input_test, item_fea_test])
