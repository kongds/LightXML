import tqdm
import time
import cProfile
import numpy as np
from apex import amp

import torch
from torch import nn

from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

from tokenizers import BertWordPieceTokenizer
from transformers import RobertaTokenizerFast

def get_bert(bert_name):
    if 'roberta' in bert_name:
        print('load roberta-base')
        model_config = RobertaConfig.from_pretrained('roberta-base')
        model_config.output_hidden_states = True
        bert = RobertaModel.from_pretrained('roberta-base', config=model_config)
    elif 'xlnet' in bert_name:
        print('load xlnet-base-cased')
        model_config = XLNetConfig.from_pretrained('xlnet-base-cased')
        model_config.output_hidden_states = True
        bert = XLNetModel.from_pretrained('xlnet-base-cased', config=model_config)
    else:
        print('load bert-base-uncased')
        model_config = BertConfig.from_pretrained('bert-base-uncased')
        model_config.output_hidden_states = True
        bert = BertModel.from_pretrained('bert-base-uncased', config=model_config)
    return bert

class LightXML(nn.Module):
    def __init__(self, n_labels, group_y=None, bert='bert-base', feature_layers=5, dropout=0.5, update_count=1,
                 candidates_topk=10, 
                 use_swa=True, swa_warmup_epoch=10, swa_update_step=200, hidden_dim=300):
        super(LightXML, self).__init__()

        self.use_swa = use_swa
        self.swa_warmup_epoch = swa_warmup_epoch
        self.swa_update_step = swa_update_step
        self.swa_state = {}

        self.update_count = update_count

        self.candidates_topk = candidates_topk

        print('swa', self.use_swa, self.swa_warmup_epoch, self.swa_update_step, self.swa_state)
        print('update_count', self.update_count)

        self.bert_name, self.bert = bert, get_bert(bert)
        self.feature_layers, self.drop_out = feature_layers, nn.Dropout(dropout)

        self.group_y = group_y
        if self.group_y is not None:
            self.group_y_labels = group_y.shape[0]
            print('hidden dim:',  hidden_dim)
            print('label goup numbers:',  self.group_y_labels)

            self.l0 = nn.Linear(self.feature_layers*self.bert.config.hidden_size, self.group_y_labels)
            # hidden bottle layer
            self.l1 = nn.Linear(self.feature_layers*self.bert.config.hidden_size, hidden_dim)
            self.embed = nn.Embedding(n_labels, hidden_dim)
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            self.l0 = nn.Linear(self.feature_layers*self.bert.config.hidden_size, n_labels)

    def get_candidates(self, group_logits, group_gd=None):
        logits = torch.sigmoid(group_logits.detach())
        if group_gd is not None:
            logits += group_gd
        scores, indices = torch.topk(logits, k=self.candidates_topk)
        scores, indices = scores.cpu().detach().numpy(), indices.cpu().detach().numpy()
        candidates, candidates_scores = [], []
        for index, score in zip(indices, scores):
            candidates.append(self.group_y[index])
            candidates_scores.append([np.full(c.shape, s) for c, s in zip(candidates[-1], score)])
            candidates[-1] = np.concatenate(candidates[-1])
            candidates_scores[-1] = np.concatenate(candidates_scores[-1])
        max_candidates = max([i.shape[0] for i in candidates])
        candidates = np.stack([np.pad(i, (0, max_candidates - i.shape[0]), mode='edge') for i in candidates])
        candidates_scores = np.stack([np.pad(i, (0, max_candidates - i.shape[0]), mode='edge') for i in candidates_scores])
        return indices, candidates, candidates_scores

    def forward(self, input_ids, attention_mask, token_type_ids,
                labels=None, group_labels=None, candidates=None):
        is_training = labels is not None

        outs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[-1]

        out = torch.cat([outs[-i][:, 0] for i in range(1, self.feature_layers+1)], dim=-1)
        out = self.drop_out(out)
        group_logits = self.l0(out)
        if self.group_y is None:
            logits = group_logits
            if is_training:
                loss_fn = torch.nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, labels)
                return logits, loss
            else:
                return logits

        if is_training:
            l = labels.to(dtype=torch.bool)
            target_candidates = torch.masked_select(candidates, l).detach().cpu()
            target_candidates_num = l.sum(dim=1).detach().cpu()
        groups, candidates, group_candidates_scores = self.get_candidates(group_logits,
                                                                          group_gd=group_labels if is_training else None)
        if is_training:
            bs = 0
            new_labels = []
            for i, n in enumerate(target_candidates_num.numpy()):
                be = bs + n
                c = set(target_candidates[bs: be].numpy())
                c2 = candidates[i]
                new_labels.append(torch.tensor([1.0 if i in c else 0.0 for i in c2 ]))
                if len(c) != new_labels[-1].sum():
                    s_c2 = set(c2)
                    for cc in list(c):
                        if cc in s_c2:
                            continue
                        for j in range(new_labels[-1].shape[0]):
                            if new_labels[-1][j].item() != 1:
                                c2[j] = cc
                                new_labels[-1][j] = 1.0
                                break
                bs = be
            labels = torch.stack(new_labels).cuda()
        candidates, group_candidates_scores =  torch.LongTensor(candidates).cuda(), torch.Tensor(group_candidates_scores).cuda()

        emb = self.l1(out)
        embed_weights = self.embed(candidates) # N, sampled_size, H
        emb = emb.unsqueeze(-1)
        logits = torch.bmm(embed_weights, emb).squeeze(-1)

        if is_training:
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels) + loss_fn(group_logits, group_labels)
            return logits, loss
        else:
            candidates_scores = torch.sigmoid(logits)
            candidates_scores = candidates_scores * group_candidates_scores
            return group_logits, candidates, candidates_scores

    def save_model(self, path):
        self.swa_swap_params()
        torch.save(self.state_dict(), path)
        self.swa_swap_params()

    def swa_init(self):
        self.swa_state = {'models_num': 1}
        for n, p in self.named_parameters():
            self.swa_state[n] = p.data.cpu().clone().detach()

    def swa_step(self):
        if 'models_num' not in self.swa_state:
            return
        self.swa_state['models_num'] += 1
        beta = 1.0 / self.swa_state['models_num']
        with torch.no_grad():
            for n, p in self.named_parameters():
                self.swa_state[n].mul_(1.0 - beta).add_(beta, p.data.cpu())

    def swa_swap_params(self):
        if 'models_num' not in self.swa_state:
            return
        for n, p in self.named_parameters():
            self.swa_state[n], p.data =  self.swa_state[n].cpu(), p.data.cpu()
            self.swa_state[n], p.data =  p.data.cpu(), self.swa_state[n].cuda()

    def get_fast_tokenizer(self):
        if 'roberta' in self.bert_name:
            tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case=True)
        elif 'xlnet' in self.bert_name:
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased') 
        else:
            tokenizer = BertWordPieceTokenizer(
                "data/.bert-base-uncased-vocab.txt",
                lowercase=True)
        return tokenizer

    def get_tokenizer(self):
        if 'roberta' in self.bert_name:
            print('load roberta-base tokenizer')
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        elif 'xlnet' in self.bert_name:
            print('load xlnet-base-cased tokenizer')
            tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        else:
            print('load bert-base-uncased tokenizer')
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        return tokenizer

    def get_accuracy(self, candidates, logits, labels):
        if candidates is not None:
            candidates = candidates.detach().cpu()
        scores, indices = torch.topk(logits.detach().cpu(), k=10)

        acc1, acc3, acc5, total = 0, 0, 0, 0
        for i, l in enumerate(labels):
            l = set(np.nonzero(l)[0])

            if candidates is not None:
                labels = candidates[i][indices[i]].numpy()
            else:
                labels = indices[i, :5].numpy()

            acc1 += len(set([labels[0]]) & l)
            acc3 += len(set(labels[:3]) & l)
            acc5 += len(set(labels[:5]) & l)
            total += 1

        return total, acc1, acc3, acc5

    def one_epoch(self, epoch, dataloader, optimizer,
                  mode='train', eval_loader=None, eval_step=20000, log=None):

        bar = tqdm.tqdm(total=len(dataloader))
        p1, p3, p5 = 0, 0, 0
        g_p1, g_p3, g_p5 = 0, 0, 0
        total, acc1, acc3, acc5 = 0, 0, 0, 0
        g_acc1, g_acc3, g_acc5 = 0, 0, 0
        train_loss = 0

        if mode == 'train':
            self.train()
        else:
            self.eval()

        if self.use_swa and epoch == self.swa_warmup_epoch and mode == 'train':
            self.swa_init()

        if self.use_swa and mode == 'eval':
            self.swa_swap_params()

        pred_scores, pred_labels = [], []
        bar.set_description(f'{mode}-{epoch}')

        with torch.set_grad_enabled(mode == 'train'):
            for step, data in enumerate(dataloader):
                batch = tuple(t for t in data)
                have_group = len(batch) > 4
                inputs = {'input_ids':      batch[0].cuda(),
                          'attention_mask': batch[1].cuda(),
                          'token_type_ids': batch[2].cuda()}
                if mode == 'train':
                    inputs['labels'] = batch[3].cuda()
                    if self.group_y is not None:
                        inputs['group_labels'] = batch[4].cuda()
                        inputs['candidates'] = batch[5].cuda()

                outputs = self(**inputs)

                bar.update(1)

                if mode == 'train':
                    loss = outputs[1]
                    loss /= self.update_count
                    train_loss += loss.item()

                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
    
                    if step % self.update_count == 0:
                        optimizer.step()
                        self.zero_grad()

                    if step % eval_step == 0 and eval_loader is not None and step != 0:
                        results = self.one_epoch(epoch, eval_loader, optimizer, mode='eval')
                        p1, p3, p5 = results[3:6]
                        g_p1, g_p3, g_p5 = results[:3]
                        if self.group_y is not None:
                            log.log(f'{epoch:>2} {step:>6}: {p1:.4f}, {p3:.4f}, {p5:.4f}'
                                    f' {g_p1:.4f}, {g_p3:.4f}, {g_p5:.4f}')
                        else:
                            log.log(f'{epoch:>2} {step:>6}: {p1:.4f}, {p3:.4f}, {p5:.4f}')
                        # NOTE: we don't reset model to train mode and keep model in eval mode
                        # which means all dropout will be remove after `eval_step` in every epoch
                        # this tricks makes LightXML converge fast
                        # self.train()

                    if self.use_swa and step % self.swa_update_step == 0:
                        self.swa_step()

                    bar.set_postfix(loss=loss.item())
                elif self.group_y is None:
                    logits = outputs
                    if mode == 'eval':
                        labels = batch[3]
                        _total, _acc1, _acc3, _acc5 =  self.get_accuracy(None, logits, labels.cpu().numpy())
                        total += _total; acc1 += _acc1; acc3 += _acc3; acc5 += _acc5
                        p1 = acc1 / total
                        p3 = acc3 / total / 3
                        p5 = acc5 / total / 5
                        bar.set_postfix(p1=p1, p3=p3, p5=p5)
                    elif mode == 'test':
                        pred_scores.append(logits.detach().cpu())
                else:
                    group_logits, candidates, logits = outputs

                    if mode == 'eval':
                        labels = batch[3]
                        group_labels = batch[4]

                        _total, _acc1, _acc3, _acc5 = self.get_accuracy(candidates, logits, labels.cpu().numpy())
                        total += _total; acc1 += _acc1; acc3 += _acc3; acc5 += _acc5
                        p1 = acc1 / total
                        p3 = acc3 / total / 3
                        p5 = acc5 / total / 5
    
                        _, _g_acc1, _g_acc3, _g_acc5 = self.get_accuracy(None, group_logits, group_labels.cpu().numpy())
                        g_acc1 += _g_acc1; g_acc3 += _g_acc3; g_acc5 += _g_acc5
                        g_p1 = g_acc1 / total
                        g_p3 = g_acc3 / total / 3
                        g_p5 = g_acc5 / total / 5
                        bar.set_postfix(p1=p1, p3=p3, p5=p5, g_p1=g_p1, g_p3=g_p3, g_p5=g_p5)
                    elif mode == 'test':
                        _scores, _indices = torch.topk(logits.detach().cpu(), k=100)
                        _labels = torch.stack([candidates[i][_indices[i]] for i in range(_indices.shape[0])], dim=0)
                        pred_scores.append(_scores.cpu())
                        pred_labels.append(_labels.cpu())


        if self.use_swa and mode == 'eval':
            self.swa_swap_params()
        bar.close()

        if mode == 'eval':
            return g_p1, g_p3, g_p5, p1, p3, p5
        elif mode == 'test':
            return torch.cat(pred_scores, dim=0).numpy(), torch.cat(pred_labels, dim=0).numpy() if len(pred_labels) != 0 else None
        elif mode == 'train':
            return train_loss
