# coding=utf-8

from config import opt
import os
import torch as t
import models
import codecs
import numpy as np
from data.dataset import DocumentPair
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils.visualize import Visualizer


def pad_sentences(sentences, sequence_length, padding_word="<PAD/>"):
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence) < sequence_length:
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
        else:
            new_sentence = sentence[:sequence_length]
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_input_data(data_left, data_right, label, vocab):
    vocabset = set(vocab.keys())
    out_left = np.array(
        [[vocab[word] if word in vocabset else vocab['<UNK/>'] for word in sentence] for sentence in data_left])
    out_right = np.array(
        [[vocab[word] if word in vocabset else vocab['<UNK/>'] for word in sentence] for sentence in data_right])
    out_y = np.array([[0, 1] if x == 1 else [1, 0] for x in label])
    return [out_left, out_right, out_y]


def load_data(batch_data, option, vocab):
    data_left = [x.strip().split(' ') for x in batch_data[0]]
    data_right = [x.strip().split(' ') for x in batch_data[1]]
    data_label = [int(x) for x in batch_data[2]]
    num_pos = sum(data_label)
    data_left = pad_sentences(data_left, option.max_len_left)
    data_right = pad_sentences(data_right, option.max_len_right)
    data_left, data_right, data_label = build_input_data(data_left, data_right, data_label, vocab)
    '''
    for i in range(10):
        print(data_left[i])
        print(data_right[i])
        print(data_label[i])
    '''
    return data_left, data_right, data_label, num_pos


def write_csv(results, file_name):
    import csv
    with codecs.open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def test(**kwargs):
    opt.parse(kwargs)
    import ipdb
    ipdb.set_trace()
    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # data
    train_data = DocumentPair(opt.test_data_root, doc_type='test')
    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (data, path) in enumerate(test_dataloader):
        inputs = t.autograd.Variable(data, volatile=True)
        if opt.use_gpu: inputs = inputs.cuda()
        score = model(inputs)
        probability = t.nn.functional.softmax(score)[:, 0].data.tolist()
        # label = score.max(dim = 1)[1].data.tolist()

        batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]

        results += batch_results
    write_csv(results, opt.result_file)

    return results


def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    # step1: configure model
    model = getattr(models, opt.model)(opt)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # step2: data
    train_data = DocumentPair(opt.train_data_root,doc_type='train', suffix='txt', load=lambda x: x.strip().split(','))
    train_data.initialize(vocab_size=opt.vocab_size)
    val_data = DocumentPair(opt.validate_data_root, doc_type='validate',
                            suffix='txt', load=lambda x: x.strip().split(','), vocab=train_data.vocab)
    val_data.initialize()
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=False, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=False, num_workers=opt.num_workers)

    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # step4: meters
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # train
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii, batch in enumerate(train_dataloader):

            data_left, data_right, label, num_pos = load_data(batch, opt, train_data.vocab)

            # train model
            input_data_left, input_data_right= Variable(t.from_numpy(data_left)), Variable(t.from_numpy(data_right))
            target = Variable(t.from_numpy(label))
            if opt.use_gpu:
                input_data_left, input_data_right = input_data_left.cuda(), input_data_right.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            scores, predictions = model((input_data_left, input_data_right))
            loss = criterion(scores, target.max(1)[1])
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.data[0])
            confusion_matrix.add(predictions.data, target.max(1)[1].data)

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])

                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        model.save()

        # validate and visualize
        val_cm, val_accuracy = val(model, val_dataloader)

        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]


def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, batch in enumerate(dataloader):
        data_left, data_right, label, num_pos = load_data(batch, opt, dataloader.dataset.vocab)
        val_input_left = Variable(t.from_numpy(data_left), volatile=True)
        val_input_right = Variable(t.from_numpy(data_right), volatile=True)
        val_label = Variable(t.from_numpy(label), volatile=True)
        if opt.use_gpu:
            val_input_left = val_input_left.cuda()
            val_input_right = val_input_right.cuda()
            val_label = val_label.cuda()
        scores, predictions = model((val_input_left, val_input_right))
        confusion_matrix.add(predictions.data, val_label.max(1)[1].data)

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def help():
    """
    打印帮助的信息： python file.py help
    """

    print(
        '''
        usage : python file.py <function> [--args=value]
        <function> := train | test | help
        example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
        avaiable args:
        '''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    import fire

    fire.Fire()