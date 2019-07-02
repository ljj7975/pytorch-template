import matplotlib.animation as animation
import matplotlib.pyplot as plt
import math

class GifGenerator:
    def __init__(self, gif_name):
        self.gif_name = gif_name
        self.fig = plt.figure(figsize=(10, 10))
        self.images = []

    def save(self):
        anim = animation.ArtistAnimation(self.fig, self.images, interval=500, repeat_delay=5000, blit=True)
        anim.save(self.gif_name, dpi=320, writer='imagemagick')

    def add_image(self, image):
        self.images.append([image])
