import json
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def save_generated_images(images, file_name, nrow=4):
    grid = make_grid(images, nrow=nrow, padding=1).permute(1,2,0).numpy()
    plt.imsave(file_name, grid)
    return plt.imshow(grid, animated=True)

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()
