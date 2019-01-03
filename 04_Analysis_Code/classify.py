import sys
import argparse

import torch
import torch.nn as nn
from torchtext import data
# np is needed for get_confusion_matrix()
import numpy as np

from simple_ntc.rnn import RNNClassifier
from simple_ntc.cnn import CNNClassifier


def define_argparser():
    '''
    Define argument parser to take inference using pre-trained model.
    '''
    p = argparse.ArgumentParser()

    p.add_argument('--model', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)

    # valid_mode == 1 : return accuracy
    # valid_mode == 2 : return confusion matrix
    p.add_argument('--valid_mode', type=int, default=1)

    config = p.parse_args()

    return config


def read_text():
    '''
    Read text from standard input for inference.
    '''
    tags = []
    sentences = []

    # read test data file
    f = open("./data/corpus.valid.newscomment.txt", 'r') # 테스트 파일 읽기
    error_tagset_cnt = 0
    while True:
        line = f.readline()
        tagset = line.split('\t')
        if len(tagset) > 1: # split list out of bound error 방지용 코드
            tag = tagset[0] # 태그 정답 저장
            sentence = tagset[1].split(' ') # 분류할 문장 저장
            tags.append(tag)
            sentences.append(sentence)
        else:
            error_tagset_cnt += 1 # 탭으로 구분되지 않는 오류 발생시키는 문장 갯수 카운트

        if not line: break
    f.close()
    print(error_tagset_cnt)
    # for line in sys.stdin:
    #     if line.strip() != '':
    #         lines += [line.strip().split(' ')]
    # print(lines)
    return tags,sentences


def define_field():
    '''
    To avoid use DataLoader class, just declare dummy fields.
    With those fields, we can retore mapping table between words and indice.
    '''
    return (
            data.Field(use_vocab=True,
                       batch_first=True,
                       include_lengths=False
                       ),
            data.Field(sequential=False,
                       use_vocab=True,
                       unk_token=None
                       )
            )

# added method from get_confusion_matrix.py
def get_confusion_matrix(classes, y, y_hat):
    confusion_matrix = np.zeros((len(classes), len(classes)))
    mapping_table = {}

    for idx, c in enumerate(classes):
        mapping_table[c] = idx

    for y_i, y_hat_i in zip(y, y_hat):
        confusion_matrix[mapping_table[y_hat_i], mapping_table[y_i]] += 1

    print('\t'.join(c for c in classes))
    for i in range(len(classes)):
        print('\t'.join(['%4d' % confusion_matrix[i, j] for j in range(len(classes))]))


def main(config):
    saved_data = torch.load(config.model, map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id)
    train_config = saved_data['config']

    rnn_best = saved_data['rnn']
    cnn_best = saved_data['cnn']
    vocab = saved_data['vocab']
    classes = saved_data['classes']

    vocab_size = len(vocab)
    n_classes = len(classes)
    text_field, label_field = define_field()
    text_field.vocab = vocab
    label_field.vocab = classes

    # get list of test data -> tags, sentences
    tags, lines = read_text()

    with torch.no_grad():
        # Converts string to list of index.
        x = text_field.numericalize(text_field.pad(lines),
                                    device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu'
                                    )

        ensemble = []
        if rnn_best is not None:
            print("rnn_best")
            # Declare model and load pre-trained weights.
            model = RNNClassifier(input_size=vocab_size,
                                  word_vec_dim=train_config.word_vec_dim,
                                  hidden_size=train_config.hidden_size,
                                  n_classes=n_classes,
                                  n_layers=train_config.n_layers,
                                  dropout_p=train_config.dropout
                                  )
            model.load_state_dict(rnn_best['model'])
            ensemble += [model]
        if cnn_best is not None:
            print("cnn_best")
            # Declare model and load pre-trained weights.
            model = CNNClassifier(input_size=vocab_size,
                                  word_vec_dim=train_config.word_vec_dim,
                                  n_classes=n_classes,
                                  dropout_p=train_config.dropout,
                                  window_sizes=train_config.window_sizes,
                                  n_filters=train_config.n_filters
                                  )
            model.load_state_dict(cnn_best['model'])
            ensemble += [model]
        y_hats = []
        # Get prediction with iteration on ensemble.
        for model in ensemble:
            print(model)
            if config.gpu_id >= 0:
                model.cuda(config.gpu_id)
            # Don't forget turn-on evaluation mode.
            model.eval()

            y_hat = []
            for idx in range(0, len(lines), config.batch_size):
                y_hat += [model(x[idx:idx + config.batch_size])]
            # Concatenate the mini-batch wise result
            y_hat = torch.cat(y_hat, dim=0)
            # |y_hat| = (len(lines), n_classes)

            y_hats += [y_hat]
        # Merge to one tensor for ensemble result and make probability from log-prob.
        y_hats = torch.stack(y_hats).exp()
        # |y_hats| = (len(ensemble), len(lines), n_classes)
        y_hats = y_hats.sum(dim=0) / len(ensemble) # Get average
        # |y_hats| = (len(lines), n_classes)

        probs, indice = y_hats.cpu().topk(config.top_k)

        if config.valid_mode == 1:
            print("valid_mode 1: get accuracy!")
            num_of_corrects = 0
            for i in range(len(lines)):
                # 정답태그 : 분류결과태그 : 분류대상문장 출력
                # print(tags[i].strip(),":",classes.itos[indice[i][0]].strip(),":",lines[i])
                # 정답태그와 분류결과태그 일치하면 정답수(num_of_corrects) 1씩 증가
                if tags[i].strip() == classes.itos[indice[i][0]].strip():
                    num_of_corrects += 1
                # sys.stdout.write('%s\t%s\n' % (' '.join([classes.itos[indice[i][j]] for j in range(config.top_k)]),
                #                  ' '.join(lines[i]))
                #                  )

            # 정확도 = 정답수 / 전체루프수 * 100
            print("accuracy: ",num_of_corrects/(len(lines)-1)*100)
        elif config.valid_mode == 2:
            print("valid_mode 2: get confusion matrix!")
            infered_results = []
            for i in range(len(lines)):
                infered_results.append(classes.itos[indice[i][0]].strip())

            min_length = min(len(tags), len(infered_results))
            tags = tags[:min_length]
            infered_results = infered_results[:min_length]

            valid_classes = list(set(tags + infered_results))

            get_confusion_matrix(valid_classes, tags, infered_results)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
