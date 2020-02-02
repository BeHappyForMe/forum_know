import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    BertTokenizer,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers.data.processors.utils import InputExample, DataProcessor

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import code
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from utils import tfidf_similarity
from metric import mean_reciprocal_rank,mean_average_precision

'''
    bert同义句训练模型
'''

logger = logging.getLogger(__name__)
MODEL_CLASS = {"bert":(BertConfig,BertTokenizer)}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class FAQProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def prepare_data(self,file_dir):
        train_df = pd.read_csv(file_dir+'baoxian_preprocessed_synonymous.csv',sep='\t')
        self.candidate_title = train_df['best_title'].astype("str").tolist()
        self.candidate_reply = train_df['reply'].astype("str").tolist()
        self.candidate_translated = train_df['translated'].astype("str").tolist()
        self.cv = TfidfVectorizer(tokenizer=lambda x:x.split())

    def get_train(self,set_type):
        """根据训练方式不同获取不同格式训练数据"""
        return self._create_examples(self.candidate_title,self.candidate_translated,set_type)

    def get_evaluate(self,file_dir,data_type):
        examples = None
        matched_questions = None
        if data_type == 'eval_original':
            examples = [InputExample(guid="%s_%s_%s" % ('eval', data_type, idx), text_a=title, text_b=None, label=1) 
                        for idx,title in enumerate(self.candidate_title)]
        else:
            evulate_df = pd.read_csv(file_dir + 'baoxian_evaluate.csv', sep='\t')
            evulate_df = evulate_df[['your_question', 'matched_question']]
            evulate_df = evulate_df[evulate_df['your_question'].notna()]
            evulate_df = evulate_df[evulate_df['matched_question'].notna()]

            questions = evulate_df['your_question'].tolist()
            matched_questions = evulate_df['matched_question'].tolist()
            if data_type == 'eval_cosine':
                # 用于计算mrr指标
                matched_questions_indexs = []
                for q in matched_questions:
                    flag = False
                    for i, _q in enumerate(self.candidate_title):
                        if q == _q:
                            matched_questions_indexs.append([i])
                            flag = True
                            break
                    if not flag:
                        # 没找到匹配的
                        matched_questions_indexs.append([-1])
                matched_questions_indexs = np.asarray(matched_questions_indexs)
                examples = [
                    InputExample(guid="%s_%s_%s" % ('eval', data_type, idx), text_a=question, text_b=None, label=1)
                    for idx, question in enumerate(questions)]
                
            elif data_type == 'eval_concate':
                examples = []
                for (idx, (line_a, line_b)) in enumerate(zip(questions, matched_questions)):
                    guid = "%s_%s_%s" % (data_type, 'pos', idx)
                    examples.append(InputExample(guid=guid, text_a=line_a, text_b=line_b, label=1))
                    
        return examples, matched_questions_indexs

    def _create_examples(self,lines_a,lines_b,set_type):
        if set_type == 'cosine':
            original_examples = []
            pos_examples = []
            neg_examples = []
            for (i, (line_a, line_b)) in enumerate(zip(lines_a, lines_b)):
                original_guid = "%s_%s_%s" % (set_type, 'original', i)
                original_examples.append(InputExample(guid=original_guid, text_a=line_a, text_b=None, label=1))

                pos_guid = "%s_%s_%s" % (set_type, 'pos', i)
                pos_examples.append(InputExample(guid=pos_guid, text_a=line_b, text_b=None, label=1))

                neg_guid = "%s_%s_%s" % (set_type, 'neg', i)
                neg_line = self.get_neg_sent(line_a)
                neg_examples.append(InputExample(guid=neg_guid, text_a=neg_line, text_b=None, label=1))
            return original_examples, pos_examples, neg_examples
        elif set_type == 'concate':
            """创建pair句子对训练数据"""
            pos_examples = []
            neg_examples = []
            for (i, (line_a, line_b)) in enumerate(zip(lines_a, lines_b)):
                pos_guid = "%s_%s_%s" % (set_type, 'pos', i)
                pos_examples.append(InputExample(guid=pos_guid, text_a=line_a, text_b=line_b, label=1))

                neg_guid = "%s_%s_%s" % (set_type, 'neg', i)
                neg_line = self.get_neg_sent(line_a)
                neg_examples.append(InputExample(guid=neg_guid, text_a=line_a, text_b=neg_line, label=1))
            return pos_examples, neg_examples

    def get_neg_sent(self,original):
        # 随机取10个，再取tf-idf最大的作为neg(即不是同义句但相似度很高)
        neg_lines = random.sample(self.candidate_title,10)
        neg_line = tfidf_similarity(self.cv,original,neg_lines)
        return neg_line

class BertForFAQHinge(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForFAQHinge, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.scoring = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_out = outputs[0]
        sequence_out = sequence_out.mean(1)

        sequence_out = self.dropout(sequence_out)
        score = F.tanh(self.scoring(sequence_out))

        outputs = (score,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # score, (hidden_states), (attentions)

def train(args, train_dataset, model, processor, tokenizer):
    """ train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter('./runs/bert_syno/'+args.loss_type)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataset) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataset) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate,eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_steps,num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    epochs_trained = 0
    global_step = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,int(args.num_train_epochs),desc="Epoch",disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)
    for k in train_iterator:
        # TODO 这是为啥
        # if k!=0:
        #     train_dataset,_,_=load_examples(args,args.task_name,tokenizer,processor)
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = SequentialSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,sampler=train_sampler,batch_size=args.train_batch_size)
        epoch_iterator = tqdm(train_dataloader,desc="Iteration",disable=args.local_rank not in [-1, 0])
        for step , batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if args.loss_type == 'concate':
                pos_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
                neg_inputs = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}
                # 分别训练同义句pair，及非同义句pari使用hingeloss计算损失
                pos_outputs = model(**pos_inputs)
                neg_outputs = model(**neg_inputs)
                pos_score = pos_outputs[0]
                neg_score = neg_outputs[0]
            elif args.loss_type == 'cosine':
                original_inputs =  {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
                pos_inputs = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}
                neg_inputs = {"input_ids": batch[6], "attention_mask": batch[7], "token_type_ids": batch[8]}
                original_outputs = model(**original_inputs)[0].mean(1)
                pos_outputs = model(**pos_inputs)[0].mean(1)
                neg_outputs = model(**neg_inputs)[0].mean(1)
                pos_score = F.cosine_similarity(original_outputs,pos_outputs)
                neg_score = F.cosine_similarity(original_outputs,neg_outputs)

            #计算hingeloss
            loss = args.margin + neg_score - pos_score
            loss[loss<0] = 0
            loss = torch.mean(loss)
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
                model.zero_grad()
                optimizer.step()
                scheduler.step()
                global_step+=1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    # 对于cosine可以用mrr评估，对于concate用score>0.5评估准确率
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        data_type = 'eval_cosine' if args.loss_type == 'cosine' else 'eval_concate'
                        eval_dataset,matched_questions_indexs = load_and_cache_examples(args,args.tas_name,tokenizer,processor,data_type=data_type)
                        results = evaluate(args, model, tokenizer, processor,eval_dataset,matched_questions_indexs)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

        logger.info("average loss: " + str(tr_loss / global_step))
    return global_step,tr_loss/global_step

def evaluate(args,model,tokenizer, processor,eval_dataset,matched_questions_indexs,prefix=""):

    # 对于cosine、concate评估方式不同，cosine计算mrr，concate计算acc
    results = {}
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.train_batch_size)
    # Eval!
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    epoch_iterator = tqdm(eval_dataloader, desc="Eval_Iteration", disable=args.local_rank not in [-1, 0])

    if args.loss_type == 'cosine':
        eval_original,_ = load_and_cache_examples(args,args.task_name,tokenizer,processor,data_type='eval_original')
        original_iterator = tqdm(eval_original, desc="Original_Iteration", disable=args.local_rank not in [-1, 0])
        original_embeddings = []
        eval_question_embeddings = []
        for step,batch in enumerate(original_iterator):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                original_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
                original_outputs = model(**original_inputs)[0].mean(1)
                original_embeddings.append(original_outputs)
        original_embeddings = torch.cat([embed.cpu().data for embed in original_embeddings]).numpy()
        for step,batch in enumerate(epoch_iterator):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                eval_question_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
                eval_questions_outputs = model(**eval_question_inputs)[0].mean(1)
                eval_question_embeddings.append(eval_questions_outputs)
        eval_question_embeddings = torch.cat([o.cpu().data for o in eval_question_embeddings]).numpy()

        scores = cosine_similarity(eval_question_embeddings,original_embeddings)
        sorted_indices = scores.argsort()[:,::-1]
        mmr = mean_reciprocal_rank(matched_questions_indexs == sorted_indices)
        map = mean_average_precision(matched_questions_indexs == sorted_indices)
        print("mean reciprocal rank: {}".format(mmr))
        print("mean average precision: {}".format(map))
        results['mmr'] = mmr
        results['map'] = map

    elif args.loss_type == 'concate':
        scores = []
        for step, batch in enumerate(epoch_iterator):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
                outputs = model(**inputs)
                score = outputs[0]
                scores.append(score)
        scores = torch.cat([o.cpu().data for o in scores]).numpy().reshape((-1))
        acc = scores[scores>0.5].sum()/len(scores)
        print("concate acc: {}".format(acc))
        results['acc'] = acc

    output_eval_file = os.path.join(args.output_dir,prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key,value in results.items():
            logger.info("  %s = %s", key, str(value))
            writer.write("%s = %s\n" % (key, value))

    return results

def predict(args, model, eval_dataset):
    # 把所有的问题与新问题pair全部用BERT编码
    # 输出score
    scores = []

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            outputs = model(**inputs)

            score = outputs[0]
            scores.append(score)

    scores = [score.data.cpu().view(-1) for score in scores]
    scores = torch.cat(scores)

    return scores.numpy()

def load_and_cache_examples(args, task, tokenizer,processor,data_type='eval_cosine'):
    """
        Load data features from cache or dataset file
    :param processor: 
    :param data_type: eval_original、eval_cosine、eval_concate
    :return: 
    """
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "evaluate",
            str(args.max_seq_length),
            str(task),
            data_type
        ),
    )
    features = None
    matched_questions_indexs = None
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features,matched_questions_indexs = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples,matched_questions_indexs = processor.get_evaluate(args.data_dir,args.data_type)
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=[1],
            max_length=args.max_seq_length,
            output_mode="classification",
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save([features,matched_questions_indexs], cached_features_file)

    # Convert to Tensors and build dataset
    eva_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    eva_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    eva_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(eva_input_ids, eva_attention_mask, eva_token_type_ids)
    return dataset,matched_questions_indexs

def load_train_examples(args,task,tokenizer,processor):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            "train",
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        datas = processor.get_train(args.loss_type)
        # Load data features from cache or dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        features = []
        for data in datas:
            feature = convert_examples_to_features(
                data,
                tokenizer,
                label_list=[1],
                output_mode="classification",
                max_length=args.max_seq_length,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            )
            features.append(feature)
        torch.save(features,cached_features_file)
    # Convert to Tensors and build dataset
    if args.loss_type == 'concate':
        pos_input_ids = torch.tensor([f.input_ids for f in features[0]], dtype=torch.long)
        pos_attention_mask = torch.tensor([f.attention_mask for f in features[0]], dtype=torch.long)
        pos_token_type_ids = torch.tensor([f.token_type_ids for f in features[0]], dtype=torch.long)
        neg_input_ids = torch.tensor([f.input_ids for f in features[1]], dtype=torch.long)
        neg_attention_mask = torch.tensor([f.attention_mask for f in features[1]], dtype=torch.long)
        neg_token_type_ids = torch.tensor([f.token_type_ids for f in features[1]], dtype=torch.long)
        dataset = TensorDataset(pos_input_ids, pos_attention_mask, pos_token_type_ids, neg_input_ids,
                                neg_attention_mask, neg_token_type_ids)
    elif args.loss_type == 'cosine':
        original_input_ids = torch.tensor([f.input_ids for f in features[0]], dtype=torch.long)
        original_attention_mask = torch.tensor([f.attention_mask for f in features[0]], dtype=torch.long)
        original_token_type_ids = torch.tensor([f.token_type_ids for f in features[0]], dtype=torch.long)
        pos_input_ids = torch.tensor([f.input_ids for f in features[1]], dtype=torch.long)
        pos_attention_mask = torch.tensor([f.attention_mask for f in features[1]], dtype=torch.long)
        pos_token_type_ids = torch.tensor([f.token_type_ids for f in features[1]], dtype=torch.long)
        neg_input_ids = torch.tensor([f.input_ids for f in features[2]], dtype=torch.long)
        neg_attention_mask = torch.tensor([f.attention_mask for f in features[2]], dtype=torch.long)
        neg_token_type_ids = torch.tensor([f.token_type_ids for f in features[2]], dtype=torch.long)
        dataset = TensorDataset(original_input_ids, original_attention_mask, original_token_type_ids,
                                pos_input_ids, pos_attention_mask, pos_token_type_ids,
                                neg_input_ids, neg_attention_mask, neg_token_type_ids)

    return dataset, processor.candidate_title, processor.candidate_reply


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",default='../data/',type=str,required=False,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",)
    parser.add_argument("--model_type",default='bert',type=str,required=False,
        help="Model type selected in the list")

    parser.add_argument("--loss_type",default="cosine",type=str,required=False,
                        choices=['cosine','concate'])

    parser.add_argument("--model_name_or_path",default='/Users/zhoup/develop/NLPSpace/my-pre-models/chinese_wwm_pytorch',type=str,required=False,
        help="Path to pre-trained model or shortcut name selected in the list",)
    parser.add_argument("--task_name",default='synonymous_faq',type=str,required=False,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),)
    parser.add_argument("--output_dir",default='./models/synonymous_model',type=str,required=False,
        help="The output directory where the model predictions and checkpoints will be written.",)

    # Other parameters
    parser.add_argument("--cache_dir",default="",type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--max_seq_length",default=256,type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--do_train",default=True, action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval",default=True, action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to predict.")
    parser.add_argument(
        "--evaluate_during_training", default=True ,action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case",default=True, action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--margin", default=1, type=float, help="The margin of hinge loss.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        default=True,
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, tokenizer_class = MODEL_CLASS[args.model_type]
    if args.loss_type == 'concate':
        model_class = BertForFAQHinge
    elif args.loss_type == 'cosine':
        model_class = BertModel
    config = config_class.from_pretrained(
        args.model_name_or_path,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else "./models/synonymous_model/cache",
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else "./model_pkl/synonymous_model/cache",
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else "./model_pkl/synonymous_model/cache",
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    processor = FAQProcessor()
    processor.prepare_data(args.data_dir)
    dataset, candidate_title, candidate_reply = load_train_examples(args, args.task_name, tokenizer, processor)
    if args.do_train:
        train(args, dataset, model, processor, tokenizer)
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            data_type = 'eval_cosine' if args.loss_type == 'cosine' else 'eval_concate'
            eval_dataset, matched_questions_indexs = load_and_cache_examples(args, args.tas_name, tokenizer, processor,
                                                                             data_type=data_type)
            result = evaluate(args, model, tokenizer, processor, eval_dataset, matched_questions_indexs,prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    #TODO
    if args.do_predict:
        while True:
            title = input("你的问题是？\n")
            if len(title.strip()) == 0:
                continue

            #将输入问题与所有训练数据的title组成句子pair，分别计算出score
            examples = [InputExample(guid=0, text_a=title, text_b=c, label=1) for c in candidate_title]
            features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=[1],
                output_mode="classification",
                max_length=args.max_seq_length,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            )

            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
            scores = evaluate(args, model, dataset)
            top5_indices = scores.argsort()[-5:][::-1]
            for index in top5_indices:
                print("可能的答案，参考问题：" + candidate_title[index] + "\t答案：" + candidate_reply[index] + "\t得分：" + str(
                    scores[index]))
                print()

if __name__ == "__main__":
    main()