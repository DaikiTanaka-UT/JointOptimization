from __future__ import print_function
import os.path
import argparse
import numpy as np
import chainer
import chainercv
from chainer import training
from chainer.training import extensions
from chainer.datasets import get_cifar10
import chainer.functions as F
from net import PreActResNet


def train_val_split(train_val, train_n):
    train_val = np.array(train_val)
    train_n = int(train_n / 10)
    train_key = []
    val_key = []

    for i in range(10):
        key = np.where(train_val[:, 1] == i)[0]
        np.random.shuffle(key)
        train_key.extend(key[:train_n])
        val_key.extend(key[train_n:])
    np.random.shuffle(train_key)
    np.random.shuffle(val_key)

    return train_val[train_key], train_val[val_key]

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

    def __init__(self, base, mean, args):
        self.base = np.array(base)
        self.mean = mean
        self.args = args
        self.count = 0
        self.labels = np.zeros(len(self.base), dtype=np.int32)
        self.soft_labels = np.zeros((len(self.base), 10), dtype=np.float32)
        self.prediction = np.zeros((len(self.base), 10, 10), dtype=np.float32)

    def __len__(self):
        return len(self.base)

    def symmetric_noise(self):
        indices = np.random.permutation(len(self.base))
        for i, idx in enumerate(indices):
            image, label = self.base[idx]
            self.labels[idx] = label
            if i < self.args.percent * len(self.base):
                self.labels[idx] = np.random.randint(10, dtype=np.int32)
            self.soft_labels[idx][self.labels[idx]] = 1.

    def asymmetric_noise(self):
        for i in range(10):
            indices = np.where(self.base[:, 1] == i)[0]
            np.random.shuffle(indices)
            for j, idx in enumerate(indices):
                image, label = self.base[idx]
                self.labels[idx] = label
                if j < self.args.percent * len(indices):
                    # truck -> automobile
                    if i == 9:
                        self.labels[idx] = 1
                    # bird -> airplane
                    elif i == 2:
                        self.labels[idx] = 0
                    # cat -> dog
                    elif i == 3:
                        self.labels[idx] = 5
                    # dog -> cat
                    elif i == 5:
                        self.labels[idx] = 3
                    # deer -> horse
                    elif i == 4:
                        self.labels[idx] = 7
                self.soft_labels[idx][self.labels[idx]] = 1.

    def get_example(self, i):
        image, _ = self.base[i]
        image = image - self.mean
        c, h, w = image.shape
        image = chainercv.transforms.resize_contain(img=image, size=(h+8, w+8))
        image = chainercv.transforms.random_crop(img=image, size=(h, w))
        image = chainercv.transforms.random_flip(img=image, x_random=True)

        return image, self.labels[i], self.soft_labels[i], i

    def label_update(self, results):
        self.count += 1

        # While updating the noisy label y_i by the probability s, we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % 10
        self.prediction[:, idx] = results

        if self.count >= self.args.begin:
            self.soft_labels = self.prediction.mean(axis=1)
            self.labels = np.argmax(self.soft_labels, axis=1).astype(np.int32)

        if self.count == self.args.epoch:
            np.save('{}/labels.npy'.format(self.args.out), self.labels)
            np.save('{}/soft_labels.npy'.format(self.args.out), self.soft_labels)


class LabelUpdate(training.extension.Extension):

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, trainer):
        model = trainer.updater.get_optimizer('main').target
        self.dataset.label_update(model.results)


class TrainChain(chainer.Chain):

    def __init__(self, length, alpha, beta):
        super(TrainChain, self).__init__()
        self.results = np.zeros((length, 10), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        # We introduce a prior probability distribution p, which is a distribution of classes among all training data.
        self.p = np.ones(10, dtype=np.float32)/10.

        with self.init_scope():
            self.model = PreActResNet()

    def __call__(self, *in_data):
        if chainer.config.train:
            image, label, soft_label, idx = in_data
            out = self.model(image)

            s = F.softmax(out)
            s_ = F.mean(s, axis=0)
            p = chainer.cuda.to_gpu(self.p)

            L_c = -F.mean(F.sum(F.log_softmax(out) * soft_label, axis=1))
            L_p = -F.sum(F.log(s_) * p)
            L_e = -F.mean(F.sum(F.log_softmax(out) * s, axis=1))

            loss = L_c + self.alpha * L_p + self.beta * L_e

            idx = chainer.cuda.to_cpu(idx)
            self.results[idx] = chainer.cuda.to_cpu(s.array)

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
    parser.add_argument('--learnrate', type=float, default=0.1,
                        help='Learning rate for SGD')
    parser.add_argument('--weight', type=float, default=1e-4,
                        help='Weight decay parameter')
    parser.add_argument('--epoch', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    parser.add_argument('--mean', default='mean.npy',
                        help='Mean image file')
    parser.add_argument('--percent', type=float, default=0,
                        help='Percentage of noise')
    parser.add_argument('--begin', type=int, default=70,
                        help='When to begin updating labels')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Hyper parameter alpha of loss function')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Hyper parameter beta of loss function')
    parser.add_argument('--asym', action='store_true',
                        help='Asymmetric noise')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random Seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    train_val_d, _ = get_cifar10()
    train_d, val_d = train_val_split(train_val_d, int(len(train_val_d)*0.9))

    if os.path.exists(args.mean):
        mean = np.load(args.mean)
    else:
        mean = np.mean([x for x, _ in train_d], axis=0)
        np.save(args.mean, mean)

    model = TrainChain(length=len(train_d), alpha=args.alpha, beta=args.beta)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    train = TrainData(train_d, mean, args)
    val = ValData(val_d, mean)

    if args.asym:
        train.asymmetric_noise()
    else:
        train.symmetric_noise()

    optimizer = chainer.optimizers.MomentumSGD(lr=args.learnrate, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val, args.batchsize, repeat=False, shuffle=False)
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Updating Labels
    trainer.extend(LabelUpdate(train), trigger=(1, 'epoch'))

    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu))
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    main()
