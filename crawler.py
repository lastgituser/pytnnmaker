#! /usr/bin/python
from icrawler.builtin import GoogleImageCrawler
import os

def download_and_rename(categorieslist):
    numcats = 0
    for k, v in categories.items():
        numcats += len(v)
    for k, v in categories.items():
        kfolder = './' + k
        if not os.path.exists(kfolder):
            os.mkdir(kfolder)
        for keywords in v:
            kwfolder = keywords.replace(' ', '_', 10)
            newfolder = kfolder + '/' + kwfolder
            if not os.path.exists(newfolder):
                os.mkdir(newfolder)
            crawler = GoogleImageCrawler(storage={'root_dir': newfolder},
                        downloader_threads=3, feeder_threads=1, 
                        parser_threads=1)
            filters = dict(
                date=((2017, 1, 1), None),
                )
            crawler.crawl(keyword=keywords,
                    filters=filters, max_num = 200)

categories = {
        'img': [
                'trash',
                'compost',
                'recyclable'
            ]
        }

if __name__ == '__main__':
    download_and_rename(categories)
