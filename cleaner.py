#! /usr/bin/python
import PIL
from PIL import Image
import os

for gencat in os.listdir('.'):
    if os.path.isdir('./' + gencat) and not gencat[0] == '.':
        os.chdir('./' + gencat)
        for imgcat in os.listdir('.'):
            if os.path.isdir('./' + imgcat) and not imgcat[0] == '.':
                os.chdir('./' + imgcat)
                for imgname in os.listdir('.'):
                    if os.path.isfile('./' + imgname) and not imgname[0] == '.':
                        try:
                            img = Image.open(imgname)
                        except:
                            os.remove(imgname)
                os.chdir('..')
        os.chdir('..')
