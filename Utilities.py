import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" #to suppress some unnecessary warnings

import tensorflow as tf, keras, numpy as np, random

class DataLoaderContrastive(tf.keras.utils.Sequence):
    '''
        DataLoader for Contrastive loss, used to train a Siamese Network
    '''

    '''
        Args:
            'dataset_root_path': path to the root directory of the dataset

            'batch_size': batch size

            'positive_ratio': positive to negative ratio of pairs for each batch, must be in range [0.0, 1.0]

            'img_size': width and height of image
    '''
    def __init__(self, dataset_root_path, batch_size, positive_ratio, img_size=(250, 250), *args, **kwargs):
        super().__init__(**kwargs)
        
        self.img_size = img_size
        self.batch_size = batch_size

        self.dataset_root_path = dataset_root_path
        self.classes_paths = []
        
        # scan all classes directories of the dataset
        for class_dir in os.scandir(dataset_root_path):
            files = os.listdir(class_dir.path)
            self.classes_paths.append([])

            for file in files:
                self.classes_paths[-1].append(os.path.join(class_dir.path, file))
        
        # shuffle the paths of the images of the dataset
        np.random.shuffle(self.classes_paths)

        '''
            calculate the total number of batches that would cover the whole dataset in a single epoch,
            The dataset will be divided into groups of three sub-batches,
            - One sub-batch is used for simialr pairs, each class images is paired with another image from the same class,
            - The other two sub-batches are used to pair with each other to make dissimilar pairs
            The final result is the concatenation of the two sub-batches, with size of (batch_size)
        '''
        self.num_positive_pairs = np.round(self.batch_size * positive_ratio).astype(np.int32)
        self.num_negative_pairs = self.batch_size - self.num_positive_pairs
        self.sub_batches_size = self.num_positive_pairs + self.num_negative_pairs * 2

        self.batches_num = np.floor(len(self.classes_paths) / self.sub_batches_size).astype(np.int32)
    
    # load an image from a file specified by 'path', returns numpy array
    def load_img(self, path):
        image = keras.preprocessing.image.load_img(path)
        image_arr = keras.preprocessing.image.img_to_array(image)
        return image_arr / 255
    
    # the training algorithm (model.fit()) will call this function to get the (n)th batch of the dataset
    def __getitem__(self, n):

        # allocate memory for the batch
        X1 = np.zeros((self.batch_size, self.img_size[1], self.img_size[0], 3))
        X2 = np.zeros((self.batch_size, self.img_size[1], self.img_size[0], 3))
        Y = np.zeros((self.batch_size, 1))

        # get three sub-batches
        i = n * self.sub_batches_size
        num_p = self.num_positive_pairs
        num_n = self.num_negative_pairs

        sub_batch_1 = self.classes_paths[i : i + num_p]
        sub_batch_2 = self.classes_paths[i + num_p : i + num_p + num_n]
        sub_batch_3 = self.classes_paths[i + num_p + num_n : i + num_p + num_n * 2]

        # make sub-batch of similar pairs
        for i in range(num_p):
            person_1 = sub_batch_1[i]

            person_1_images = random.sample(person_1, 2)
            X1[i] = self.load_img(person_1_images[0])
            X2[i] = self.load_img(person_1_images[1])
            Y[i] = 1

        # make another sub-batch of dissimilar pairs
        for i in range(num_n):
            person_2 = sub_batch_2[i]
            person_3 = sub_batch_3[i]

            X1[i + num_p] = self.load_img(random.sample(person_2, 1)[0])
            X2[i + num_p] = self.load_img(random.sample(person_3, 1)[0])
            Y[i + num_p] = 0
        
        # Since the model has two inputs, X1 and X2 has to be grouped into a tuple.
        return (X1, X2), Y

    # returns the number of batches in the dataset
    def __len__(self):
        return self.batches_num
    
    # called at the end of the epoch
    def on_epoch_end(self):
        # shuffle the paths of the images of the dataset
        np.random.shuffle(self.classes_paths)

def get_dataset_with_prefetching(dataset_root_path, batch_size = 32, positive_ratio=0.2, image_size = (250, 250)):
    '''
        This is a simple function that turns a data loader into a tensorflow dataset with prefetching enabled 
        for better performance when training.
    '''
    h, w = image_size

    return tf.data.Dataset.from_generator(
        DataLoaderContrastive, 
        args = [dataset_root_path, batch_size, positive_ratio, (h, w)], 
        output_signature = ((tf.TensorSpec(shape = (batch_size, h, w, 3), dtype = tf.float32), tf.TensorSpec(shape = (batch_size, h, w, 3), dtype = tf.float32)), tf.TensorSpec(shape = (batch_size, 1), dtype = tf.float32))
    ).prefetch(tf.data.AUTOTUNE)
