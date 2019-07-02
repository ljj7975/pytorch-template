import matplotlib.animation as animation
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from torch.autograd import Variable
from utils import to_categorical, save_generated_images, GifGenerator

class SampleGenerator:
    def __init__(self, output_dir, generator, latent_dim, cat_dim, cont_dim, n_row=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = generator.to(self.device)

        self.n_row = n_row
        self.latent_dim = latent_dim
        self.cat_dim = cat_dim
        self.cont_dim = cont_dim

        self.img_dirs = self._prepare_img_dirs(output_dir)

        self.figs = {}
        for key in self.img_dirs.keys():
            self.figs[key] = plt.figure(key, figsize=(10, 10))

        # Static generator inputs for sampling
        self.static_z = Variable(torch.Tensor(np.zeros((cat_dim ** 2, latent_dim))).float()).to(self.device)
        self.static_cat_c = to_categorical(
            np.array([num for _ in range(cat_dim) for num in range(cat_dim)]), num_columns=cat_dim
        ).to(self.device)
        self.static_cont_c = Variable(torch.Tensor(np.zeros((cat_dim ** 2, cont_dim))).float()).to(self.device)

        zeros = np.zeros((self.n_row ** 2, 1))
        c_varied = np.repeat(np.linspace(-1, 1, self.n_row)[:, np.newaxis], self.n_row, 0)

        self.c_variations = []
        for i in range(self.cont_dim):
            codes = []
            for j in range(self.cont_dim):
                if i == j:
                    codes.append(c_varied)
                else:
                    codes.append(zeros)
            self.c_variations.append(np.concatenate(codes, -1))

        self.gif_generator = GifGenerator(output_dir, self.figs)

    def _prepare_img_dirs(self, output_dir):
        img_dirs = {}

        category_dir = output_dir / 'category'
        category_dir.mkdir(parents=True, exist_ok=True)

        img_dirs['category'] = str(category_dir)

        for index in range(self.cont_dim):
            continous_dir = output_dir / ('continous_' + str(index))
            continous_dir.mkdir(parents=True, exist_ok=True)

            img_dirs[('continous_' + str(index))] = str(continous_dir)

        return img_dirs

    def generate_sample(self, file_name):
        """Saves a grid of generated digits ranging from 0 to cat_dim"""

        self.generator.eval()
        
        with torch.no_grad():
            cat_sample = self.generator(self.static_z, self.static_cat_c, self.static_cont_c).detach().cpu()
        plt.figure('category')
        img = save_generated_images(cat_sample, self.img_dirs['category']+"/{}.png".format(str(file_name)), nrow=self.n_row)
        self.gif_generator.add_image("category", img)

        for index in range(self.cont_dim):
            key = 'continous_' + str(index)
            dir = self.img_dirs[key]
            varied_c = Variable(torch.Tensor(self.c_variations[index]).float()).to(self.device)

            with torch.no_grad():
                cont_sample = self.generator(self.static_z, self.static_cat_c, varied_c).detach().cpu()
            plt.figure(key)
            img = save_generated_images(cont_sample, self.img_dirs[key]+"/{}.png".format(str(file_name)), nrow=self.n_row)
            self.gif_generator.add_image(key, img)

    def generate_gif(self):
        self.gif_generator.save_all()
