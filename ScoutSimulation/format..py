from yapf.yapflib.yapf_api import FormatFile
from yapf.yapflib.style import CreateGoogleStyle
from functools import partial
import os

files = [
    'scout_simulation.py',
    'scout.py'
]

dirs = [
    './utils',
    './env_objects'
]

if __name__ == "__main__":
    style = CreateGoogleStyle()
    formatter = partial(FormatFile, style_config=style, in_place=True)

    for f in os.scandir('.'):
        if f.path[2:] in files:
            print(f'Reformatted -> {f}')
            formatter(filename=f.path)

    for d in dirs:
        for f in os.scandir(d):
            if '.py' in f.path:
                print(f'Reformatted -> {f}')
                formatter(filename=f.path)
