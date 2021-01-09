import tqdm
import torch
import numpy as np
from dataset import createDataCSV

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model1', type=str, required=False, default='')
parser.add_argument('--model2', type=str, required=False, default='')
parser.add_argument('--model3', type=str, required=False, default='')

parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')

args = parser.parse_args()

if __name__ == '__main__':
    using_group = args.dataset in ['wiki500k', 'amazon670k']
    model_labels, model_scores = [], []

    models = [args.model1, args.model2, args.model3]
    models = [i for i in models if i != '']
    for model in models:
        print(f'loading {model}')
        model_scores.append(np.load(f'./results/{model}-scores.npy', allow_pickle=True))
        if using_group:
            model_labels.append(np.load(f'./results/{model}-labels.npy', allow_pickle=True))
    
    df, label_map = createDataCSV(args.dataset)
    print(f'load {args.dataset} dataset with '
          f'{len(df[df.dataType =="train"])} train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')

    df = df[df.dataType == 'test']
    results = {k:[0, 0, 0] for k in models + ['all']}

    bar = tqdm.tqdm(total=len(df))

    for i, true_labels in enumerate(df.label.values):
        true_labels = set([label_map[i] for i in true_labels.split()])

        bar.update()

        if using_group:
            pred_labels = {}
            for j in range(len(models)):
                results[models[j]][0] += len(set([model_labels[j][i][0]]) & true_labels)
                results[models[j]][1] += len(set(model_labels[j][i][:3]) & true_labels)
                results[models[j]][2] += len(set(model_labels[j][i][:5]) & true_labels)
                for l, s in sorted(list(zip(model_labels[j][i], model_scores[j][i])), key=lambda x: x[1], reverse=True):
                #for l, s in zip(model_labels[j][i], model_scores[j][i]):
                    if l in pred_labels:
                        pred_labels[l] += s
                    else:
                        pred_labels[l] = s
            pred_labels = [k for k, v in sorted(pred_labels.items(), key=lambda item: item[1], reverse=True)]
            
            results['all'][0] += len(set([pred_labels[0]]) & true_labels)
            results['all'][1] += len(set(pred_labels[:3]) & true_labels)
            results['all'][2] += len(set(pred_labels[:5]) & true_labels)
        else:
            index = i
            logits = [torch.sigmoid(torch.tensor(model_scores[i][index])) for i in range(len(models))]
            logits.append(sum(logits))
            logits = [(-i).argsort()[:10].cpu().numpy() for i in logits]

            for i, logit in enumerate(logits):
                name = models[i] if i != len(models) else 'all'
                results[name][0] += len(set([logit[0]]) & true_labels)
                results[name][1] += len(set(logit[:3]) & true_labels)
                results[name][2] += len(set(logit[:5]) & true_labels)

    total = len(df)

    for k in results:
        p1 = results[k][0] / total
        p3 = results[k][1] / total / 3
        p5 = results[k][2] / total / 5
        print(f'{k}: p1:{p1} p3:{p3} p5:{p5}')
