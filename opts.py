import argparse
import os
import sys

class opts():
  def __init__(self):
    self.parser = argparse.ArgumentParser()

    self.parser.add_argument('--lr', type=float, default=0.001)
    self.parser.add_argument('--momentum', type=float, default=0.5)
    self.parser.add_argument('--epochs', type=int, default=10)
    self.parser.add_argument('--val_intervals', type=int, default=1)
    self.parser.add_argument('--batch_size', type=int, default=128)

  def parse(self):
    opt = self.parser.parse_args()

    opt.train_val_split = 0.8    
    opt.momentum = 0.0
    opt.test_batch_size = 128
    opt.train_face = 'training_face/*.jpg'
    opt.train_nonface = 'training_nonface/*.jpg'
    opt.test_face = 'test_face/*.jpg'
    opt.test_nonface = 'test_nonface/*.jpg'
    return opt
