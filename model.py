import torch
import numpy as np
from transformers import AutoConfig, AutoModel
from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn.functional as F
from torch.autograd import Function


class SentenceTransformer(torch.nn.Module):
    def __init__(self, args):
        super(SentenceTransformer, self).__init__()
        transformer_config = AutoConfig.from_pretrained(args.pretrained_model, return_dict=True,
                                                        output_attentions=True, output_hidden_states=True)
        self.transformer = AutoModel.from_pretrained(args.pretrained_model, config=transformer_config)
        self.dropout = torch.nn.Dropout(p=args.dropout)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(transformer_config.hidden_size, transformer_config.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(transformer_config.hidden_size, args.num_labels))

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_id, attention_mask):
        # list of input_ids and attention mask
        input_id = input_id.squeeze(dim=1)  # reduce dimension
        attention_mask = attention_mask.squeeze(dim=1)
        output = self.transformer(input_ids=input_id, attention_mask=attention_mask)
        post_encoding = self.mean_pooling(output['last_hidden_state'], attention_mask)
        attentions = output['attentions']
        features = self.dropout(post_encoding)
        logits = self.linear(features)
        predictions = F.softmax(logits, dim=1).detach().cpu().numpy()
        predictions = predictions.flatten()
        predictions = predictions[1] - predictions[0]
        return predictions, logits, attentions, features


# class ReverseLayerF(Function):
#
#     @staticmethod
#     def forward(ctx, x, lambd):
#         ctx.lambd = lambd
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         output = grad_output.neg() * ctx.lambd
#         return output, None


class SentenceTransformerAdversarial(SentenceTransformer):
    def __init__(self, args):
        super(SentenceTransformerAdversarial, self).__init__(args)
        transformer_config = AutoConfig.from_pretrained(args.pretrained_model, return_dict=True,
                                                        output_attentions=True, output_hidden_states=True)
        self.transformer = AutoModel.from_pretrained(args.pretrained_model, config=transformer_config)
        self.dropout = torch.nn.Dropout(p=args.dropout)

        self.linear = \
            torch.nn.Sequential(
                torch.nn.Linear(transformer_config.hidden_size, transformer_config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(transformer_config.hidden_size, args.num_labels))

        self.lang_classifier = \
            torch.nn.Sequential(
                torch.nn.Linear(transformer_config.hidden_size, transformer_config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(transformer_config.hidden_size, 5))  # we have 5 langs

        self.adversarial = args.adversarial

    def forward(self, input_id, attention_mask):
        # list of input_ids and attention mask
        input_id = input_id.squeeze(dim=1)  # reduce dimension
        attention_mask = attention_mask.squeeze(dim=1)
        output = self.transformer(input_ids=input_id, attention_mask=attention_mask)
        post_encoding = self.mean_pooling(output['last_hidden_state'], attention_mask)
        attentions = output['attentions']
        features= self.dropout(post_encoding)
        logits = self.linear(features)
        predictions = F.softmax(logits, dim=1).detach().cpu().numpy()
        predictions = predictions.flatten()
        predictions = predictions[1] - predictions[0]

        # if self.adversarial:
        #     reverse_features = ReverseLayerF.apply(post_encoding, grl_lambda)
        #     lang_logits = self.lang_classifier(reverse_features)
        # else:
        lang_logits = self.lang_classifier(post_encoding)

        return predictions, logits, attentions, lang_logits, features


class Trainer:
    def __init__(self, args):
        self.args = args
        self.use_gpu = args.cuda

    def train(self, dataloader, model):
        criterion = torch.nn.CrossEntropyLoss()
        num_total_steps = len(dataloader) * self.args.epochs
        num_warmup_steps = num_total_steps * 0.1

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_total_steps
        )

        if self.use_gpu:
            model.to(torch.device('cuda'))

        for epoch in range(self.args.epochs):
            model.train()
            total_sample = 0
            correct = 0
            loss_total = 0
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].squeeze(1).cuda() if self.use_gpu else batch['input_ids'].squeeze(
                    dim=1)
                attention_mask = batch['attention_mask'].cuda() if self.use_gpu else batch['attention_mask']
                targets = batch['label'].squeeze(1).cuda() if self.use_gpu else batch['label'].squeeze(1)
                _, outputs, attentions, _ = model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                _, predictions = torch.max(outputs.data, 1)
                correct += (predictions.cpu().detach().numpy() == targets.cpu().detach().numpy()).sum()
                total_sample += input_ids.shape[0]
                loss_step = loss.item()
                loss_total += loss_step
                print(f'epoch {epoch}, step {batch_idx}, loss {loss_step}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                scheduler.step()

            loss_total = loss_total / total_sample
            acc_total = correct / total_sample
            print(f'epoch {epoch}, step {batch_idx}, loss {loss_total}, acc {acc_total}')

        return model

    def test(self, dataloader, model):
        model.eval()
        with torch.no_grad():
            labels = []
            targs = []
            probs = []
            topics = []
            tweet_ids = []
            features_all=[]
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].squeeze(1).cuda() if self.use_gpu else batch['input_ids'].squeeze(
                    dim=1)
                tweet_id = batch['tweet_id'].detach().numpy()
                topic_id = batch['topic_id']
                attention_mask = batch['attention_mask'].cuda() if self.use_gpu else batch['attention_mask']

                predictions, outputs, attentions, features = model(input_ids, attention_mask)

                _, logits = torch.max(outputs.data, 1)
                probs.append(predictions)
                labels.append(logits.cpu().detach().numpy())
                features_all.extend(features.cpu().detach().numpy())

                if 'label' in batch:
                    targets = batch['label'].squeeze(1).cuda() if self.use_gpu else batch['label'].squeeze(1)
                    targs.append(targets.cpu().detach().numpy())
                tweet_ids.extend(tweet_id)
                topics.extend(topic_id)

        labels = np.asarray(labels).flatten()
        targs = np.asarray(targs).flatten()
        probs = np.asarray(probs).flatten()


        return dict(
            topics=topics,
            tweet_ids=tweet_ids,
            probs=probs,
            labels=labels,
            targets=targs,
            features=features_all)


class AdversarialTrainer(Trainer):
    def __init__(self, args):
        super(AdversarialTrainer, self).__init__(args)

    def train(self, dataloader, model):
        criterion = torch.nn.CrossEntropyLoss()
        num_total_steps = len(dataloader) * self.args.epochs
        num_warmup_steps = num_total_steps * 0.1

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_total_steps
        )

        if self.use_gpu:
            model.to(torch.device('cuda'))

        for epoch in range(self.args.epochs):
            model.train()
            total_sample = 0
            correct = 0
            loss_total = 0
            for batch_idx, batch in enumerate(dataloader):
                p = float(batch_idx + epoch * self.args.train_batch_size) / (
                        self.args.epochs * self.args.train_batch_size)

                input_ids = batch['input_ids'].squeeze(1).cuda() if self.use_gpu else batch['input_ids'].squeeze(
                    dim=1)
                attention_mask = batch['attention_mask'].cuda() if self.use_gpu else batch['attention_mask']
                targets = batch['label'].squeeze(1).cuda() if self.use_gpu else batch['label'].squeeze(1)
                encoded_langs = batch['encoded_langs'].squeeze(1).cuda() if self.use_gpu else batch[
                    'encoded_langs'].squeeze(1)
                _, outputs, attentions, lang_outputs,_ = model(input_ids, attention_mask)
                loss = criterion(outputs, targets)
                lang_loss = criterion(lang_outputs, encoded_langs)
                loss = self.args.claim_weight * loss + (1 - self.args.claim_weight) * lang_loss

                optimizer.zero_grad()
                _, predictions = torch.max(outputs.data, 1)
                correct += (predictions.cpu().detach().numpy() == targets.cpu().detach().numpy()).sum()
                total_sample += input_ids.shape[0]
                loss_step = loss.item()
                loss_total += loss_step
                print(f'epoch {epoch}, step {batch_idx}, loss {loss_step}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                scheduler.step()

            loss_total = loss_total / total_sample
            acc_total = correct / total_sample
            print(f'epoch {epoch}, step {batch_idx}, loss {loss_total}, acc {acc_total}')

        return model

    def test(self, dataloader, model):
        model.eval()
        with torch.no_grad():
            labels = []
            targs = []
            probs = []
            topics = []
            tweet_ids = []
            lang_labels = []
            features_all = []
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].squeeze(1).cuda() if self.use_gpu else batch['input_ids'].squeeze(
                    dim=1)
                tweet_id = batch['tweet_id'].detach().numpy()
                topic_id = batch['topic_id']
                attention_mask = batch['attention_mask'].cuda() if self.use_gpu else batch['attention_mask']

                if 'label' in batch:
                    targets = batch['label'].squeeze(1).cuda() if self.use_gpu else batch['label'].squeeze(1)
                    targs.append(targets.cpu().detach().numpy())
                # targets = batch['label'].squeeze(1).cuda() if self.use_gpu else batch['label'].squeeze(1)

                encoded_langs = batch['encoded_langs'].squeeze(1).cuda() if self.use_gpu else batch[
                    'encoded_langs'].squeeze(1)

                predictions, outputs, attentions, langs_output, features = model(input_ids, attention_mask)

                _, logits = torch.max(outputs.data, 1)
                _, lang_logits = torch.max(langs_output.data, 1)
                probs.append(predictions)
                features_all.append(features.cpu().detach().numpy())
                labels.append(logits.cpu().detach().numpy())
                lang_labels.append(lang_logits.cpu().detach().numpy())
                tweet_ids.extend(tweet_id)
                topics.extend(topic_id)

        labels = np.asarray(labels).flatten()
        lang_labels = np.asarray(lang_labels).flatten()
        targs = np.asarray(targs).flatten()
        probs = np.asarray(probs).flatten()

        return dict(
            topics=topics,
            tweet_ids=tweet_ids,
            probs=probs,
            labels=labels,
            targets=targs,
            features=features_all,
            lang_labels=lang_labels)


TRANSFORMER_MODELS = {'sentence_transformer':
                          SentenceTransformer,
                      'adversarial_sentence_transformer':
                          SentenceTransformerAdversarial}

TRAINER = {
    'normal': Trainer,
    'adversarial': AdversarialTrainer
}
