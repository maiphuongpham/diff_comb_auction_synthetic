import itertools

import os

import numpy as np

class BaseGenerator(object):
    def __init__(self, config, mode="train"):
        self.config = config
        self.mode = mode
        self.num_agents = config.num_agents
        self.num_items = config.num_items
        self.num_bundles = config.num_bundles
        self.num_instances = config[self.mode].num_batches * config[self.mode].batch_size
        self.num_misreports = config[self.mode].num_misreports
        self.batch_size = config[self.mode].batch_size
        print("batch_size: ", self.batch_size, "num_instances: ", self.num_instances, "num_batches: ", config[self.mode].num_batches) 
    def build_generator(self, X=None, ADV=None, C=None):#, I=None):
        if self.mode == "train":
            if self.config.train.data == "fixed":
                self.get_data(X, ADV, C)#, I)
                self.gen_func = self.gen_fixed()
            else:
                self.gen_func = self.gen_online()

        else:
            if self.config[self.mode].data == "fixed" or X is not None:
                self.get_data(X, ADV, C)
                self.gen_func = self.gen_fixed()
            else:
                self.gen_func = self.gen_online()

    def get_data(self, X=None, ADV=None, C=None):#, I=None):
        """Generates data"""
        x_shape = [self.num_instances, self.num_agents, self.num_items] #self.num_bundles]
        adv_shape = [self.num_misreports, self.num_instances, self.num_agents, self.num_items] #self.num_bundles]
        c_shape = [self.num_instances, self.num_agents, self.num_bundles - self.num_items]

        if X is None:
            X = self.generate_random_X(x_shape)
        if ADV is None:
            ADV = self.generate_random_ADV(adv_shape)
        if C is None:
            C = self.generate_random_C(c_shape)

        self.X = X
        self.ADV = ADV
        self.C = C

    def load_data_from_file(self, iter):
        """Loads data from disk"""
        self.X = np.load(os.path.join(self.config.dir_name, "X.npy"))
        self.ADV = np.load(os.path.join(self.config.dir_name, "ADV_" + str(iter) + ".npy"))
        self.C = np.load(os.path.join(self.config.dir_name, "C.npy"))

    def save_data(self, iter):
        """Saved data to disk"""
        if self.config.save_data is None:
            return

        if iter == 0:
            np.save(os.path.join(self.config.dir_name, "X"), self.X)
            np.save(os.path.join(self.config.dir_name, "C"), self.C)
            # np.save(os.path.join(self.config.dir_name, "I"), self.I)
        else:
            np.save(os.path.join(self.config.dir_name, "ADV_" + str(iter)), self.ADV)


    def gen_fixed(self):
        i = 0
        if self.mode == "train":
            perm = np.random.permutation(self.num_instances)
        else:
            perm = np.arange(self.num_instances)

        while True:
            if (i + 1) * self.batch_size > self.num_instances:
                i = 0
                perm = np.random.permutation(self.num_instances) if self.mode == "train" else np.arange(self.num_instances)
                continue

            idx = perm[i * self.batch_size : (i + 1) * self.batch_size]
            yield self.X[idx, :, :], self.ADV[:, idx, :, :], self.C[idx, :], idx
            i += 1
            if i * self.batch_size == self.num_instances:
                i = 0
                if self.mode == "train":
                    perm = np.random.permutation(self.num_instances)
                else:
                    perm = np.arange(self.num_instances)

    def gen_online(self):
        x_batch_shape = [self.batch_size, self.num_agents, self.num_items]
        adv_batch_shape = [self.num_misreports, self.batch_size, self.num_agents, self.num_items]
        c_batch_shape = [self.batch_size, self.num_agents, self.num_bundles - self.num_items]# [self.num_instances, self.num_agents, self.num_bundles - self.num_items]
        while True:
            X = self.generate_random_X(x_batch_shape)
            ADV = self.generate_random_ADV(adv_batch_shape)
            C = self.generate_random_C(c_batch_shape)
            yield X, ADV, C, None

    def update_adv(self, idx, adv_new):
        """Updates ADV for caching"""
        self.ADV[:, idx, :, :] = adv_new

    def generate_random_X(self, shape):
        """Rewrite this for new distributions"""
        raise NotImplementedError

    def generate_random_ADV(self, shape):
        """Rewrite this for new distributions"""
        raise NotImplementedError

    def generate_random_C(self, shape):
        """Rewrite this for new distributions"""
        raise NotImplementedError
