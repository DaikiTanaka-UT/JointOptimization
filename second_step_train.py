from __future__ import print_function
import os.path
import argparse
import numpy as np
import chainer
import chainercv
from chainer import training
from chainer.training import extensions, triggers
from chainer.datasets import get_cifar10
import chainer.functions as F
from net import PreActResNet
from first_step_train import train_val_split


class ValData(chainer.dataset.DatasetMixin):

    def __init__(self, base, mean):
        self.base = np.array(base)
        self.mean = mean

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, label = self.base[i]
        image = image - self.mean
        return image, label


class TrainData(chainer.dataset.DatasetMixin):

    def __init__(self, base, mean, updated_labels, updated_soft_labels):
        self.base = np.array(base)
        self.mean = mean
        self.updated_labels = updated_labels
        self.updated_soft_labels = updated_soft_labels

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        image, _ = self.base[i]
        image = image - self.mean
        c, h, w = image.shape
        image = chainercv.transforms.resize_contain(img=image, size=(h+8, w+8))
        image = chainercv.transforms.random_crop(img=image, size=(h, w))
        image = chainercv.transforms.random_flip(img=image, x_random=True)

        return image, self.updated_labels[i], self.updated_soft_labels[i]


class TrainChain(chainer.Chain):

    def __init__(self):
        super(TrainChain, self).__init__()
        with self.init_scope():
            self.model = PreActResNet()

    def __call__(self, *in_data):
        if chainer.config.train:
            image, label, soft_label = in_data
            out = self.model(image)
            loss = -F.mean(F.sum(F.log_softmax(out) * soft_label, axis=1))
        else:
            image, label = in_data
            out = self.model(image)
            loss = F.softmax_cross_entropy(out, label)

        chainer.report({'loss': loss, 'accuracy': F.accuracy(out, label)}, self)

        return loss


def main():
    parser = argparse.ArgumentParser(description='noisy CIFAR-10 training:')
    parser.add_argument('--batchsize', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', type=float, default=0.2,
                        help='Learning rate for SGD')
    parser.add_argument('--weight', type=float, default=1e-4,
                        help='Weight decay parameter')
    parser.add_argument('--epoch', type=int, default=120,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    parser.add_argument('--mean', default='mean.npy',
                        help='Mean image file')
    parser.add_argument('--label', default='result',
                        help='Directory where the labels obtained in the first step exist')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random Seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    train_val_d, test_d = get_cifar10()
    train_d, val_d = train_val_split(train_val_d, int(len(train_val_d)*0.9))

    if os.path.exists(args.mean):
        mean = np.load(args.mean)
    else:
        mean = np.mean([x for x, _ in train_d], axis=0)
        np.save(args.mean, mean)

    model = TrainChain()
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    updated_labels = np.load('{}/labels.npy'.format(args.label))
    updated_soft_labels = np.load('{}/soft_labels.npy'.format(args.label))
    train = TrainData(train_d, mean, updated_labels, updated_soft_labels)
    val = ValData(val_d, mean)
    test = ValData(test_d, mean)

    optimizer = chainer.optimizers.MomentumSGD(lr=args.learnrate, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val, args.batchsize, repeat=False, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trigger_epochs = [int(args.epoch / 3), int(args.epoch * 2 / 3)]
    trainer.extend(extensions.ExponentialShift('lr', 0.1, init=args.learnrate), trigger=triggers.ManualScheduleTrigger(trigger_epochs, 'epoch'))

    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu))
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

    test_evaluator = extensions.Evaluator(test_iter, model, device=args.gpu)
    results = test_evaluator()
    print('Test accuracy:', results['main/accuracy'])


if __name__ == '__main__':
    main()
