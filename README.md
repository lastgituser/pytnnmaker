# Run Demo

If you want to run the demo, simply run the command:

```
$ python final.py
```

You will need the icrawler and pytorch libraries to run the program.
The categories variable details the folder structure.
If you want to change it, you have to change the key, which is the root where
all the images will be stored, and the children, which are the keywords that 
will be searched.
If you want to change the name of the file that the model will be saved as,
change the variable save\_file.
The program will output a log file containing the number of the iteration, the
accuracy and loss of the model at a certain epoch, the epoch, and the number of
images in each folder for each category.
If you wish to change the name of the log file, change the variable logfilename.
During the course of the execution of the program, the images that have been
used to train upon will be moved to a folder that \[ROOT\]\_collective, where
the name of the root folder is \[ROOT\].

# Separate Processes

If you wish to separate the processes of crawling, cleaning, and training, you
can run this sequence of commands:

```
$ python crawler.py
$ python cleaner.py
$ python trainer.py
```

Each of those files have parameters named similarly if you wish to change
specific details like the folder to get data from, the location to save a model
and load a model from.
It would help to understand the icrawler crawl function to best make the filters
and the number of images if one wishes to change that, but one only needs to
change the variables.

# Evaluation

If you wish to evaluate the quality of a model, make sure the model exists and
there are images in the folders per category.
Then, you should run this command:

```
$ python evaluator.py
```

It will print out the accuracy of the model as well as the loss associated with
that model.
