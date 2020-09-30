import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import csv
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

dir_train = '/home/srikumar/Desktop/ENPM673_Perception_for_autonomous_robots/Project6/train/'
dir_test = '/home/srikumar/Desktop/ENPM673_Perception_for_autonomous_robots/Project6/test1/'
size= 200
alpha = 0.001
epoch = 15
name = 'data-{}-{}-{}.model'.format(alpha,epoch, '2conv-basic')

def label_name(image):              #To split names to dog and  cats 
    name = image.split('.')[0]
    if name =='cat':
        return [1,0]
    elif name=='dog':
        return [0,1]
    
def create_train_data():
    training_data = []
    for im in os.listdir(dir_train):
        label = label_name(im)
        path = os.path.join(dir_train,im)
        new_img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (size,size))
        training_data.append([np.array(new_img),np.array(label)])
    shuffle(training_data)          #To just shuffle data and make it random, and avoid overfitting
    np.save('train_set.npy', training_data)
    return training_data


def process_test_data():
    test_data = []
    for im in os.listdir(dir_test):
        path = os.path.join(dir_test,im)
        num = im.split('.')[0]
        new_img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE), (size,size))
        test_data.append([np.array(new_img),num])
    test_data.sort(key=lambda x:x[1])
    shuffle(test_data)          #To just shuffle data and make it random, and avoid overfitting
    np.save('test_set.npy', test_data)
    return test_data

def validate_train_set(c_net,train_set):
    
    #First layer
    #For ouput filter size - 32
    c_net = conv_2d(c_net, 32, 3, activation='relu', padding='same')
    c_net = max_pool_2d(c_net, 3)
    
    #Second layer
    #For ouput filter size - 64
    c_net = conv_2d(c_net, 64, 3, activation='relu')
    c_net = max_pool_2d(c_net, 3)
    
    #Third layer
    #For ouput filter size - 128
    c_net = conv_2d(c_net, 128, 3, activation='relu')
    c_net = max_pool_2d(c_net, 3)
    
    #Fourth layer
    #For ouput filter size - 128
    c_net = conv_2d(c_net, 64, 3, activation='relu')
    c_net = max_pool_2d(c_net, 3)
    
    
    #Fifth layer
    #For ouput filter size - 128
    c_net = conv_2d(c_net, 128, 3, activation='relu')
    c_net = max_pool_2d(c_net, 3)
        
    #Sixth layer
    #For ouput filter size - 64
    c_net = conv_2d(c_net, 64, 3, activation='relu')
    c_net = max_pool_2d(c_net, 3)
    
    #Seventh layer
    #For ouput filter size - 32
    c_net = conv_2d(c_net, 32, 3, activation='relu')
    c_net = max_pool_2d(c_net, 3)
    
    #Fully connected layer with 'relu' activation
    c_net = fully_connected(c_net, 1024, activation='relu')
    
    #drop_out to avoid over-fitting
    c_net = dropout(c_net, 0.9)
    
    #Fully connected layer with 'softmax' activation
    c_net = fully_connected(c_net, 2, activation='softmax')
    c_net = regression(c_net, optimizer='adam', learning_rate = alpha, loss = 'categorical_crossentropy', name='targets')
    
    model = tflearn.DNN(c_net, tensorboard_dir='log')
    
    if os.path.exists('{}.meta'.format(name)):
        model.load(name)
        print('model loaded!')
        return model
    
    #Creating 2 new list from train_set and labling them as testing and training sub data sets
    train_sub = train_set[:-2500]    #choosing 24000 sets as train dataset 
    test_sub = train_set[-2500:]     #choosing last 1000 as the test dataset
    
    #for fit
    train_x = np.array([i[0] for i in train_sub]).reshape(-1, size, size, 1)
    train_y = [i[1] for i in train_sub]
    
    #for testing accuracy
    test_x = np.array([i[0] for i in test_sub]).reshape(-1, size, size, 1)
    test_y = [i[1] for i in test_sub]
    
    model.fit({'input': train_x}, {'targets': train_y}, n_epoch=epoch, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=500, show_metric=True)
    model.save(name)
    
    return model
    
def run_for_test(model,test_set):
    fig=plt.figure()
    
    print("writing onto files: \n")
    
    with open('submission_file.csv','w') as f:
        f.write('id,label\n')
     
    with open('submission_file.csv','a') as f:
        for data in test_set:
            img_num = data[1]
            img_data = data[0]
            orig = img_data
            data = img_data.reshape(size,size,1)
            model_out = model.predict([data])[0]
            f.write('{},{}\n'.format(img_num,model_out[1]))
        f.close()
    
    #For non rounded file 
    csv1 = 'submission_file.csv'
    file = open(csv1, newline='\n')
    reader = csv.reader(file)
    header = next(reader)
    data = []
    for row in reader:
        img_num = int(row[0])
        d_or_c = float(row[1])
        data.append([img_num, d_or_c])
    
    data.sort(key = lambda x: x[0])
    new_file = 'submission_file_sorted.csv'
    file = open(new_file, 'w')
    writer = csv.writer(file)
    writer.writerow(["id", "label"])
    for d in data:
        writer.writerow([d[0],d[1]])   
    for num,data in enumerate(test_set[:12]):
        img_num = data[1]
        img_data = data[0]
    
        y = fig.add_subplot(3,4,num+1)
        orig = img_data
        data = img_data.reshape(size,size,1)
        model_out = model.predict([data])[0]
    
    # for rounded file
    csv1 = 'submission_file.csv'
    file = open(csv1, newline='\n')
    reader = csv.reader(file)
    header = next(reader)
    data = []
    for row in reader:
        img_num = int(row[0])
        d_or_c = float(row[1])
        if d_or_c > 0.5:
            d_or_c = 1
        else:
            d_or_c = 0
        data.append([img_num, d_or_c])
    
    data.sort(key = lambda x: x[0])
    new_file = 'submission_file_roundedandsorted.csv'
    file = open(new_file, 'w')
    writer = csv.writer(file)
    writer.writerow(["id", "label"])
    for d in data:
        writer.writerow([d[0],d[1]])   
    for num,data in enumerate(test_set[:12]):
        img_num = data[1]
        img_data = data[0]
    
        y = fig.add_subplot(3,4,num+1)
        orig = img_data
        data = img_data.reshape(size,size,1)
        model_out = model.predict([data])[0]
    
        if np.argmax(model_out) == 1: 
            str_label='Dog'
        else: 
            str_label='Cat'
        
        y.imshow(orig,cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()
    plt.savefig('cats_and_dogs_epoch_{}'.format(epoch))
    plt.pause(10)

    
    
def main():
    
    #next four lines are to be executed just once - loading testing and training data sets
    print("Creating training dataset... \n")
    train_set = create_train_data()
    print("Processing testing dataset... \n")
    test_set = process_test_data()
    
    #to reset graph for every run
    tf.reset_default_graph()
    
    #workaround for earlier verison of numpy to use np.load    
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)        # modify the default parameters of np.load
    #loading training data
    train_set = np.load('train_set.npy') #len 25000
    test_set = np.load('test_set.npy')  #len 12500
    #restoring to curret version
    np.load = np_load_old
        
    #creating flattened image and sending as input
    c_net = input_data(shape = [None,size,size,1], name='input')
    
    #training dataset validation and preparation
    model = validate_train_set(c_net,train_set)
    
    #running for test dataset
    run_for_test(model, test_set)
    
if __name__ == '__main__':
    main()