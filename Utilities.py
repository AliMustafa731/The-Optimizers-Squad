import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" #to suppress some unnecessary warnings

import tensorflow as tf, keras, numpy as np, random


class DataLoaderContrastive(tf.keras.utils.Sequence):

    '''
    This is the data loader used to train a siamese network that uses contrastive loss.
    The dataset returns two images of either the same person or different people, along with a label that indicates this fact.
    '''

    def __init__(self, dataset_path, batch_size, input_shape=(250, 250), *args, **kwargs):
        super().__init__(**kwargs)

        self.input_shape = input_shape
        self.batch_size = batch_size

        self.dataset_path = dataset_path
        self.classes_paths = []
        
        for dir in os.scandir(self.dataset_path):
            self.classes_paths.append([])
            
            for file in os.scandir(dir.path):
                if file.is_file():
                    self.classes_paths[-1].append(file.path)
        
        # calculate the total number of batches
        self.batches_num = np.ceil(len(self.classes_paths) / self.batch_size).astype(np.int32)
    
    def load_img(self, path):
        image = keras.preprocessing.image.load_img(path)
        image_arr = keras.preprocessing.image.img_to_array(image)
        return image_arr / 255
        
    # the training algorithm (model.fit()) will call this function to get the n'th batch of the dataset
    def __getitem__(self, n):
        
        paths_batch = self.classes_paths[n*self.batch_size : (n+1)*self.batch_size]

        X1 = np.zeros((self.batch_size, self.input_shape[1], self.input_shape[0], 3))
        X2 = np.zeros((self.batch_size, self.input_shape[1], self.input_shape[0], 3))
        Y = np.zeros((self.batch_size, 1))

        for i in range(self.batch_size):
            sample_class_1 = random.choice(paths_batch)
            sample_class_2 = random.choice(paths_batch)

            X1[i] = self.load_img(random.choice(sample_class_1))
            X2[i] = self.load_img(random.choice(sample_class_2))
            
            if sample_class_1 == sample_class_2:
                Y[i] = 1
            else:
                Y[i] = 0
        
        #Since the model has two inputs, X1 and X2 has to be grouped into a tuple.
        return (X1, X2), Y

    # returns the number of batches in the dataset
    def __len__(self):
        return self.batches_num
    
    # called at the end of the epoch
    def on_epoch_end(self):
        # shuffle the paths of images at the end of each epoch
        np.random.shuffle(self.classes_paths)


def get_dataset_with_prefetching(dataset_root_path, batch_size = 32, image_size = (250, 250)):

    '''
    This is a simple function that turns a data loader into a tensorflow dataset with prefetching enabled 
    for better performance when training.
    '''

    h, w = image_size

    return tf.data.Dataset.from_generator(
        DataLoaderContrastive, 
        args = [dataset_root_path, batch_size, (h, w)], 
        output_signature = ((tf.TensorSpec(shape = (batch_size, h, w, 3), dtype = tf.float32), tf.TensorSpec(shape = (batch_size, h, w, 3), dtype = tf.float32)), tf.TensorSpec(shape = (batch_size, 1), dtype = tf.float32))
    ).prefetch(tf.data.AUTOTUNE)
