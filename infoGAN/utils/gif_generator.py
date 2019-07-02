import matplotlib.animation as animation
import matplotlib.pyplot as plt
import math

class GifGenerator:
    def __init__(self, output_dir, fig_mapping):
        self.output_dir = str(output_dir)
        self.images = {}
        self.figs = fig_mapping

    def save(self, key):
        if key not in self.images:
            raise KeyError("image does not exist")
            return

        if key not in self.figs:
            raise KeyError("figure does not exist")
            return

        anim = animation.ArtistAnimation(self.figs[key], self.images[key], interval=500, repeat_delay=5000, blit=True)
        anim.save(self.output_dir+"/{}.gif".format(key), dpi=320, writer='imagemagick')

    def save_all(self):
        for key in self.images.keys():
            self.save(key)

    def add_image(self, key, image):
        if key not in self.images:
            self.images[key] = []
        self.images[key].append([image])
