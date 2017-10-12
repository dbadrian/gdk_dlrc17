__author__ = "Jonathon Luiten"
__copyright__ = "Copyright 2017, AI Research, Data Technology Centre, Volkswagen Group"
__credits__ = ["Jonathon Luiten"]
__license__ = "MIT"
__maintainer__ = "Jonathon Luiten"

import numpy as np
from sklearn import mixture
import cv2
import os

import gdk.utils as utils


class debugger():
    def __init__(self):
        self.small_ims = []
        self.latents =None # np.empty([0,32])
        self.gmm = None

    def add(self,small_ims,latents):
        for small_im in small_ims:
            self.small_ims.append(small_im)

        for latent in latents:
            if self.latents is None:
                self.latents = utils.decode_numpy_array(latent)
            else:
                self.latents = np.concatenate((self.latents,  utils.decode_numpy_array(latent)), axis=0)
            print(self.latents.shape)
            # print(latents.shape)

        # if self.latents is None:
        #     self.latents = latents
        # else:
        #     self.latents = np.concatenate((self.latents,latents),axis =0)
        # print(self.latents.shape)
        # print(latents.shape)

    def predict(self,latent):
        label = self.gmm.predict(latent)
        return label

    def cluster(self):
        n_components = 4
        try:
            self.gmm = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
            self.gmm.fit(self.latents)
            cluster_labels = self.gmm.predict(self.latents)

            #### New faster plotting code test

            fol = "/home/dlrc/aaaa/"
            num_fol = len(os.listdir(fol))
            if not os.path.exists(fol + str(num_fol).zfill(4)):
                os.makedirs(fol + str(num_fol).zfill(4))

            for label in set(cluster_labels):
                tmp_images = list(np.array(self.small_ims)[cluster_labels == label])
                n_rows = int(np.ceil(np.sqrt(len(tmp_images))))
                tot_im = np.zeros((n_rows * 64, n_rows * 64, 3), dtype=float)
                for i, img in enumerate(tmp_images):
                    _x, _y = np.unravel_index(i, (n_rows, n_rows))
                    _img = tmp_images[i]
                    tot_im[_x * 64: (_x + 1) * 64, _y * 64: (_y + 1) * 64] = _img

                num_im = len(os.listdir(fol + str(num_fol).zfill(4)))
                cv2.imwrite(fol + str(num_fol).zfill(4) + "/" + str(num_im).zfill(4) + "_output.jpg", tot_im)
        except:
            print("GMM fit error")


        # fol = "/home/dlrc/aaaa/"
        # num_fol = len(os.listdir(fol))-1
        # num_im = len(fol + str(num_fol).zfill(4))-1
        # if not os.path.exists(fol + str(num_fol).zfill(4)):
        #     os.makedirs(fol + str(num_fol).zfill(4))
        # cv2.imwrite(fol + str(num_fol).zfill(4) + "/" + str(num_im).zfill(4) + "_test.jpg", small_im)
