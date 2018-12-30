#Gigi Hsueh
#image folder paths may need to be changed to run accordingly
#preprocess_formulas and image files are from zenodo.org

import sys
from skimage import io
import numpy as np
import csv
import re
from scipy.misc import imresize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from preprocess_formulas import tokenize_formula, remove_invisible, normalize_formula

#images folder paths
image_train= "./im2latex_train.lst"
image_path="./formula_images/"
formulas_path="./im2latex_formulas.lst"


class LatexTranslator:

    def __init__(self, learning_rate=0.001, 
            image_size=50, split=0.2):

        #initializing things
        self.learning_rate = learning_rate
        self.image_size = image_size #for resizing and reshaping 
        self.split = split
        self.im_train = {} #puts the training data into a dictionary for searching purposes 
        self.image_data = [] #array an image pixels, used as inputs later
        self.token_list= {} #array of tokens, for tokenizing 
        #self.targets_vector = [] attempted for the prediction
        self.target = [] #targets
        self.labels = [] #an array of image labels, to open the images, and to find its correct formula
        self.index = []
        self.i = 0
        self.maxlen = 10

    def extract_label(self, image_name):
        #spliting the file name and keeping the label of the image
        label_name = image_name.split('/')[2]
        label_name = label_name.split('.')[0]
        
        return label_name

    def image_processing(self, image):
        #takes each image, crop, resize, reshape, and then add to an array
        label = self.extract_label(image) 
        img = io.imread(image, as_gray=True)
        img = img[300:1550, 330:1580]
        img = imresize(img, (self.image_size, self.image_size)) 
        img = img.reshape([self.image_size, self.image_size, 1])           
        self.image_data.append(img) #after reshaping the images, it is stored in image_data to use as inputs later
        self.labels.append(np.array(label)) #keeping labels of each images

    def formula_processing(self, formulas):
        #only taking the formulas that are smaller than length of 150
        #using the tokenizer, it removes invisible, noralize the formula, and then tokenize the formula
        #then for special token, they're saved to a dictionary assigning an integer to each token
        #Also, each formulas are turned into integer values for easier use of target
        token_for = []
        formula = remove_invisible(formulas)
        formula = normalize_formula(formula)
        tokenized = tokenize_formula(formula)
        #for each token in a formula
        for token in tokenized:
            #creates a unique list of tokens, self.token_list
            if token not in self.token_list:
                self.token_list[token] = self.i 
                self.i = self.i + 1
            #changing the formula into integers
            token_for.append(self.token_list[token])

        #for any formulas that has extra spaces at the end, add 0's
        if (len(token_for) != self.maxlen):
            for i in (range(self.maxlen-len(token_for))):
                token_for.append(0)

        return token_for


    def open_formula(self, index):
        #opens the formula file line by line, for each line, process it
        f = open(formulas_path, "r", errors='ignore')
        formula = f.readlines()
        f.close()

        #for any formula with its length less than 200,
        #remember where that formula came from using its index
        if (len(formula[index]) <= 200):
            self.index.append(index)
            #keeping a highest possble max length
            if (len(formula[index]) > self.maxlen):
                self.maxlen = len(formula[index])
            #process the formula
            t = self.formula_processing(formula[index])
            #save into the target array
            self.target.append(t)


    def values_to_targets(self, values):
        #one-hot vector encoder (not used)
        # #integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        #binary encode 
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        self.targets_vector = onehot_encoded


    def predict_image(self, image):
        #this is for after the trained_model, it makes a prediction about the model and output an array of probabilities
        #(not used)
        results = self.model.predict([img])[0]
        most_probable = max(results)
        results = list(results)
        most_probable_index = results.index(most_probable)
        return results

    def load_model(self, model_file):
        #loading the model saved 
        model = self.build_cnn()
        model.load(model_file)
        self.model = model


    def build_cnn(self):
        #CNN
        convnet = input_data(shape=[None, self.image_size, self.image_size, 1], name='input')
        
        convnet = conv_2d(convnet, 32, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 128, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)

        convnet = conv_2d(convnet, 64, 5, activation='relu')
        convnet = max_pool_2d(convnet, 5)
      
        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        #output layer of 200 because maxlen is 200
        convnet = fully_connected(convnet, 200, activation='softmax')
        convnet = regression(convnet, optimizer='adam',
                             learning_rate=self.learning_rate,
                             loss='binary_crossentropy',
                             name='targets')

        model = tflearn.DNN(convnet,
                            tensorboard_dir='log',
                            tensorboard_verbose=0
        )

        return model


    def train_model(self, n_epochs=10, batch_size=2):
        #input is our images
        #output is the latex formulas
        X = self.image_data
        y = self.target


        model = self.build_cnn()

        model.fit(X, y,
                      n_epoch=10,
                      validation_set=0.1,
                      snapshot_step = 500,
                      show_metric=True,
            batch_size=batch_size)

        model.save('network.tflearn')
        


if __name__ == '__main__':

    # n_epochs = sys.argv[1]
    # m_model = sys.argv[2]
    # test_images_list = sys.argv[3]

    lt = LatexTranslator(learning_rate=0.001,
                image_size=50, split=0.2)
    numtrain = 100
    train = {}

    with open(image_train, "r") as data:
            for line in data.readlines()[:numtrain]:
                parts = line.split(' ')
                train[parts[0]] = parts[1] 
                lt.open_formula(int(parts[0]))
                for index in lt.index:
                    if (int(parts[0])==index):
                        lt.image_processing('./formula_images/'+parts[1]+'.png')


    lt.train_model(
        n_epochs=10,
        batch_size=5)

    # lt.load_model(m_model)

    # with open(test_images_list, "r") as data:
    #         for line in data.readlines()[:numtrain]:
    #             parts = line.split(' ')
    #             train[parts[0]] = parts[1] 
    #             lt.open_formula(int(parts[0]))
    #             for index in lt.index:
    #                 if (int(parts[0])==index):
    #                     lt.image_processing('./formula_images/'+parts[1]+'.png')


    # X = lt.image_data
    # y = lt.targets_vector

    # results = lt.model.predict(X)
    # #most_probable = max(results)
    # results = list(results)
    # most_probable_index = results.index(most_probable)
