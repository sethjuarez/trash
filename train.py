import argparse
import tensorflow as tf
from tensorflow import keras
from dataset import TrashData
from detection.models.detectors import faster_rcnn

def main():
    # dataset
    dataset = TrashData()
    dataset.load('data/dataset_surfrider_cleaned')

    # model

    model = faster_rcnn.FasterRCNN(num_classes=dataset.class_count)
    optimizer = keras.optimizers.SGD(1e-3, momentum=0.9, nesterov=True)


    for epoch in range(100):
        loss_history = []
        for (batch, inputs) in enumerate(train_tf_dataset):

            batch_imgs, batch_metas, batch_bboxes, batch_labels = inputs
            with tf.GradientTape() as tape:
                rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = model(
                    (batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)

                loss_value = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            loss_history.append(loss_value.numpy())

            if batch % 10 == 0:
                print('epoch', epoch, batch, np.mean(loss_history))
    

if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='CNN Training for Image Recognition.')
    #parser.add_argument('-d', '--data', help='directory to training and test data', default='data')
    #parser.add_argument('-o', '--output', help='output directory', default='outputs')
    #parser.add_argument('-g', '--logs', help='log directory', default='logs')
    
    #parser.add_argument('-e', '--epochs', help='number of epochs', default=5, type=int)
    #parser.add_argument('-b', '--batch', help='batch size', default=100, type=int)
    #parser.add_argument('-l', '--lr', help='learning rate', default=0.001, type=float)

    #parser.add_argument('-m', '--model', help='model type', default='cnn', choices=['linear', 'nn', 'cnn'])

    #args = parser.parse_args()

    main()