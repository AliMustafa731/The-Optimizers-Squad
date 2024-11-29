import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" #to suppress some unnecessary warnings

import tensorflow as tf, keras, numpy as np, random

class DataLoaderContrastive(tf.keras.utils.Sequence):

    def __init__(self, dataset_root_path, batch_size, input_shape=(250, 250), *args, **kwargs):
        super().__init__(**kwargs)
        
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.half_batch_size = batch_size // 2

        self.dataset_root_path = dataset_root_path
        self.multi_image_classes_paths = []      # list containing paths of classes that have two or more images
        self.single_image_classes_paths = []     # list containing paths of classes that have only one images
        
        # scan all classes directories
        for class_dir in os.scandir(dataset_root_path):
            files = os.listdir(class_dir.path)
            
            if len(files) > 1:
                # this path contains more than one file (image) of the same class (person)
                self.multi_image_classes_paths.append([])
                for file in files:
                    self.multi_image_classes_paths[-1].append(os.path.join(class_dir.path, file))
            else:
                # this path conatins only one file (image) of the same class (person)
                self.single_image_classes_paths.append([])
                self.single_image_classes_paths[-1].append(os.path.join(class_dir.path, files[0]))

        self.all_classes_paths = self.single_image_classes_paths + self.multi_image_classes_paths

        # shuffle the paths of images of the dataset
        np.random.shuffle(self.all_classes_paths)
        np.random.shuffle(self.single_image_classes_paths)
        np.random.shuffle(self.multi_image_classes_paths)

        # calculate the total number of batches
        min_length = min(len(self.multi_image_classes_paths), len(self.single_image_classes_paths))
        self.batches_num = np.ceil(min_length / self.half_batch_size).astype(np.int32)
    
    # load an image from a file specified by 'path'
    # returns numpy array
    def load_img(self, path):
        image = keras.preprocessing.image.load_img(path)
        image_arr = keras.preprocessing.image.img_to_array(image)
        return image_arr / 255
    
    # the training algorithm (model.fit()) will call this function to get the n'th batch of the dataset
    def __getitem__(self, n):

        X1 = np.zeros((self.batch_size, self.input_shape[1], self.input_shape[0], 3))
        X2 = np.zeros((self.batch_size, self.input_shape[1], self.input_shape[0], 3))
        Y = np.zeros((self.batch_size, 1))

        # make "half-batch-size" of similar pairs
        multi_image_classes_batch = self.multi_image_classes_paths[n*self.half_batch_size : (n+1)*self.half_batch_size]

        for i in range(self.half_batch_size):
            # make "half-batch-size" of similar pairs

            # remember that each class in "similar_batch_sample" has two images or more!
            # take two images from the 'i'th class (they will be shuffeled at the end of each epoch)
            similar_sub_images = multi_image_classes_batch[i]
            
            X1[i] = self.load_img(similar_sub_images[0])
            X2[i] = self.load_img(similar_sub_images[1])
            Y[i] = 1

            # make another "half-batch-size" of dissimilar pairs

            # remember that we can pair one class from "similar" group with another from "dissimilar" group
            # to make a "dissimilar" pair. in other words, we can take the sample from the whole dataset!
            dissimilar_pair_sample = random.sample(self.all_classes_paths, 2)

            X1[i + self.half_batch_size] = self.load_img(dissimilar_pair_sample[0][0])
            X2[i + self.half_batch_size] = self.load_img(dissimilar_pair_sample[1][0])
            
            # account for the damn small probability of getting two similar classes!
            if dissimilar_pair_sample[0] == dissimilar_pair_sample[1]:
                Y[i + self.half_batch_size] = 1
            else:
                Y[i + self.half_batch_size] = 0
        
        #Since the model has two inputs, X1 and X2 has to be grouped into a tuple.
        return (X1, X2), Y

    # returns the number of batches in the dataset
    def __len__(self):
        return self.batches_num
    
    # called at the end of the epoch
    def on_epoch_end(self):
        # shuffle the paths of images at the end of each epoch
        np.random.shuffle(self.all_classes_paths)
        np.random.shuffle(self.single_image_classes_paths)
        np.random.shuffle(self.multi_image_classes_paths)

        for cls in self.multi_image_classes_paths:
            np.random.shuffle(cls)


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
