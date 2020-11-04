import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import MDataset, createDataCSV

from model import LightXML

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
args = parser.parse_args()

if __name__ == '__main__':
    df, label_map = createDataCSV(args.dataset)
    print(f'load {args.dataset} dataset with '
          f'{len(df[df.dataType =="train"])} train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')

    xmc_models = []
    predicts = []
    berts = ['bert-base', 'roberta', 'xlnet']

    for index in range(len(berts)):
        model_name = [args.dataset, '' if berts[index] == 'bert-base' else berts[index]]
        model_name = '_'.join([i for i in model_name if i != ''])

        model = LightXML(n_labels=len(label_map), bert=berts[index])

        print(f'models/model-model_name.bin')
        model.load_state_dict(torch.load(f'models/model-{model_name}.bin'))

        tokenizer = model.get_tokenizer()
        test_d = MDataset(df, 'test', tokenizer, label_map, 128 if args.dataset == 'amazoncat13k' and berts[index] == 'xlnent' else 512)
        testloader = DataLoader(test_d, batch_size=16, num_workers=0,
                                shuffle=False)

        model.cuda()
        predicts.append(torch.Tensor(model.one_epoch(0, testloader, None, mode='test')[0]))
        xmc_models.append(model)

    df = df[df.dataType == 'test']
    total = len(df)
    acc1 = [0 for i in range(len(berts) + 1)]
    acc3 = [0 for i in range(len(berts) + 1)]
    acc5 = [0 for i in range(len(berts) + 1)]

    for index, true_labels in enumerate(df.label.values):
        true_labels = set([label_map[i] for i in true_labels.split()])

        logits = [torch.sigmoid(predicts[i][index]) for i in range(len(berts))]
        logits.append(sum(logits))
        logits = [(-i).argsort()[:10].cpu().numpy() for i in logits]

        for i, logit in enumerate(logits):
            acc1[i] += len(set([logit[0]]) & true_labels)
            acc3[i] += len(set(logit[:3]) & true_labels)
            acc5[i] += len(set(logit[:5]) & true_labels)

    for i, name in enumerate(berts + ['all']):
        p1 = acc1[i] / total
        p3 = acc3[i] / total / 3
        p5 = acc5[i] / total / 5

        with open('./results/{args.dataset}', 'a') as f:
            print(f'{name} {p1} {p3} {p5}', file=f)
            print(f'{name} {p1} {p3} {p5}')
