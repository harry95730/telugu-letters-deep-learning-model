# -*- coding: utf-8 -*-
"""
Classes for controling machine learning processes
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import csv


class TrainingPlot:
    """
    Creating live plot during training
    REUIRES notebook backend: %matplotlib notebook
    @TODO Migrate to Tensorboard
    """
    train_loss = []
    train_acc = []
    valid_acc = []
    test_iter = 0
    loss_iter = 0
    interval = 0
    ax1 = None
    ax2 = None
    fig = None

    def __init__(self, steps, test_itr, loss_itr):
        self.test_iter = test_itr
        self.loss_iter = loss_itr
        self.interval = steps

        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()
        self.ax1.set_autoscaley_on(True)
        plt.ion()

        self._update_plot()

        # Description
        self.ax1.set_xlabel('Training steps')
        self.ax1.set_ylabel('Valid. Accuracy')
        #self.ax1.set_ylabel('TRAIN LOSS')
        self.ax2.set_ylabel('Valid. Accuracy')


         # Axes limits
      #  self.ax1.set_ylim([0,10])
    def _update_plot(self):
        self.fig.canvas.draw()
   # def update_loss(self, loss_train, index):
    #    self.train_loss.append(loss_train)
     #   if len(self.train_loss) == 1:
      #      self.ax1.set_ylim([0, min(10, math.ceil(loss_train))])
       # self.ax1.plot(self.interval * np.arange(len(self.train_loss)),
        #              self.train_loss, 'b', linewidth=1.0)

      #  self._update_plot()


      #  self._update_plot()

    def update_acc(self, acc_val, acc_train, index):
        self.valid_acc.append(acc_val)
        self.train_acc.append(acc_train)
        
        self.ax2.plot(self.test_iter * np.arange(len(self.valid_acc)),
                      self.valid_acc, 'r', linewidth=1.0, label='test')
        self.ax2.plot(self.test_iter * np.arange(len(self.train_acc)),
                      self.train_acc, 'g',linewidth=1.0, label='train')

        self.ax2.set_title('Valid. Accuracy: {:.4f}'.format(self.valid_acc[-1]))
        self.fig.legend("training accuracy","testing accuracy");

        self._update_plot()


        

class DataSet:
    """Class for training data and feeding train function."""
    images = None
    labels = None
    length = 0
    index = 0

    def __init__(self, img, lbl):
        self.images = img
        self.labels = lbl
        self.length = len(img)
        self.index = 0

    def next_batch(self, batch_size):
        """Return the next batch from the data set."""
        start = self.index
        self.index += batch_size

        if self.index > self.length:
            # Shuffle the data
            perm = np.arange(self.length)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.index = batch_size

        end = self.index
        return self.images[start:end], self.labels[start:end]
