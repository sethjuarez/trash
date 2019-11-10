import argparse
import tensorflow as tf


def main():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Training for Image Recognition.')
    parser.add_argument('-d', '--data', help='directory to training and test data', default='data')
    parser.add_argument('-o', '--output', help='output directory', default='outputs')
    parser.add_argument('-g', '--logs', help='log directory', default='logs')
    
    parser.add_argument('-e', '--epochs', help='number of epochs', default=5, type=int)
    parser.add_argument('-b', '--batch', help='batch size', default=100, type=int)
    parser.add_argument('-l', '--lr', help='learning rate', default=0.001, type=float)

    parser.add_argument('-m', '--model', help='model type', default='cnn', choices=['linear', 'nn', 'cnn'])

    args = parser.parse_args()
