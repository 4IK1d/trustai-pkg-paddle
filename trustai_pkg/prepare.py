import abc
from functools import partial

import numpy as np
# import paddle
# import paddle.nn as nn
from paddle.io import BatchSampler, DistributedBatchSampler, DataLoader
# from paddlenlp.transformers import PegasusForConditionalGeneration, PegasusChineseTokenizer
# from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger
# from paddlenlp.metrics import BLEU
from paddlenlp.data import Pad, Stack, Dict, DataCollatorForSeq2Seq

class ModelSet:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

class SimilarityPrepareConfig:
    def __init__(self, text_column, target_column, max_source_length,max_seq_length, max_target_length, min_target_length, batch_size):
        """
        # 原始字段需要移除
        remove_columns = ['content', 'title']
        # 文本的最大长度
        max_source_length = 128
        # 摘要的最大长度
        max_target_length = 64
        """
        self.text_column = text_column 
        self.target_column = target_column 
        self.max_source_length = max_source_length
        self.max_seq_length =  max_seq_length
        self.max_target_length = max_target_length
        self.min_target_length = min_target_length 
        self.batch_size = batch_size

class ConvertersMixin:
    """Mixin for prepare tools to convert different example.
    """
    def _general_converter(self, example, is_test=False):
        """
        Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens. And creates a mask from the two sequences passed 
        to be used in a sequence-pair classification task.
            
        A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        A BERT sequence pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        If only one sequence, only returns the first portion of the mask (0's).
        Args:
            example(obj:`list[str]`): List of input data, containing text and label if it have label.
            tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
                which contains most of the methods. Users should refer to the superclass for more information regarding methods.
            max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
                Sequences longer than this will be truncated, sequences shorter will be padded.
            is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.
        Returns:
            input_ids(obj:`list[int]`): The list of token ids.
            token_type_ids(obj: `list[int]`): List of sequence pair mask.
            label(obj:`numpy.array`, data type of int64, optional): The input label if not is_test.
        """
        possible_encoded_input_names = [
            "text", "sentence", "text_a", "text_b", "sentence1", "sentence2", "query", "title", "context"
        ]
        possible_label_names = ["label", "labels"]

        # Search a possible name
        encoded_input_names = []
        for n in possible_encoded_input_names:
            if n in example:
                encoded_input_names.append(n)

        encoded_label_name = None
        for n in possible_label_names:
            if n in example:
                encoded_label_name = n
                break

        if len(encoded_input_names) == 1:
            encoded_inputs = self.model_set.tokenizer(text=example[encoded_input_names[0]], max_seq_len=self.prepare_config.max_seq_length)
        elif len(encoded_input_names) == 2:
            encoded_inputs = self.model_set.tokenizer(text=example[encoded_input_names[0]],
                                    text_pair=example[encoded_input_names[1]],
                                    max_seq_len=self.prepare_config.max_seq_length)
        else:
            raise ValueError("error input_names.")

        # encoded_inputs = tokenizer(text=example[encoded_input_name], max_seq_len=max_seq_length)
        # input_ids = encoded_inputs["input_ids"]
        # token_type_ids = encoded_inputs["token_type_ids"]

        if not is_test:
            encoded_inputs['label'] = np.array([example[encoded_label_name]], dtype="int64")

        return encoded_inputs
    
    def _generation_converter(self, example):
        """
        构造模型的输入.
        """
        inputs = example[self.prepare_config.text_column]
        targets = example[self.prepare_config.target_column]
        # 分词
        model_inputs = self.model_set.tokenizer(inputs,
                                max_length=self.prepare_config.max_source_length,
                                padding=False,
                                truncation=True,
                                return_attention_mask=True)
        labels = self.model_set.tokenizer(targets,
                        max_length=self.prepare_config.max_target_length,
                        padding=False,
                        truncation=True)
        # 得到labels，后续通过DataCollatorForSeq2Seq进行移位
        model_inputs["label"] = labels["input_ids"]
        return model_inputs

class BasePrepare(abc.ABC):
    """Base class for preparation.
    """
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def get_dataset(self):
        pass

    @abc.abstractmethod
    def get_dataloader(self):
        pass

class SimilarityPrepare(BasePrepare, ConvertersMixin):
    """Prepare data for NLG tasks abstract generation.
    """
    def __init__(self, model_set: ModelSet, prepare_config: SimilarityPrepareConfig):
        super(SimilarityPrepare, self).__init__()
        self.model_set = model_set
        self.prepare_config = prepare_config

    def get_dataset(self, train_dataset, dev_dataset, test_dataset=None, trans_func = None):
        # 定义转换器
        trans_func = self._general_converter if not trans_func else trans_func
        trans_func = partial(trans_func, is_test=False) 

        # train_dataset和dev_dataset分别转换
        train_dataset = train_dataset.map(trans_func, batched=False)
        dev_dataset = dev_dataset.map(trans_func, batched=False)

        if test_dataset:
            logger.debug('tt:{}'.format(type(train_dataset)))
            logger.debug('t:{}'.format(type(test_dataset)))
            test_func = partial(trans_func, is_test=True)
            test_dataset = test_dataset.map(test_func, batched=False)
            return train_dataset, dev_dataset, test_dataset

        return train_dataset, dev_dataset
    
    def get_dataloader(self, train_dataset, dev_dataset, test_dataset=None):
        """获取模型所需的batchfy的dataloader。
        """
        # 组装 Batch 数据 & Padding
        # batchify_fn = DataCollatorForSeq2Seq(tokenizer=self.model_set.tokenizer, model=self.model_set.model)
        # TODO create collector
        batchify_fn = lambda samples, fn=Dict({
            'input_ids': Pad(axis=0, pad_val=self.model_set.tokenizer.pad_token_id),
            'token_type_ids': Pad(axis=0, pad_val=self.model_set.tokenizer.pad_token_type_id),
            'label': Stack(dtype="int64")
        }): fn(samples)
        # : [data for data in fn(samples)]
        # 分布式批采样器，用于多卡分布式训练
        train_batch_sampler = DistributedBatchSampler(
            train_dataset, batch_size=self.prepare_config.batch_size, shuffle=True)

        # 构造训练Dataloader
        train_data_loader = DataLoader(dataset=train_dataset,
                                    batch_sampler=train_batch_sampler,
                                    num_workers=0,
                                    collate_fn=batchify_fn,
                                    return_list=True)

        dev_batch_sampler = BatchSampler(dev_dataset,
                                        batch_size=self.prepare_config.batch_size,
                                        shuffle=False)
         # 构造验证Dataloader
        dev_data_loader = DataLoader(dataset=dev_dataset,
                                    batch_sampler=dev_batch_sampler,
                                    num_workers=0,
                                    collate_fn=batchify_fn,
                                    return_list=True)

        if test_dataset:
            # NOTE:test set do no shuffle
            test_batch_sampler = BatchSampler(test_dataset,
                                            batch_size=self.prepare_config.batch_size,
                                            shuffle=False)
            # 构造测试Dataloader
            test_data_loader = DataLoader(dataset=test_dataset,
                                        batch_sampler=test_batch_sampler,
                                        num_workers=0,
                                        collate_fn=batchify_fn,
                                        return_list=True)
            return train_data_loader, dev_data_loader, test_data_loader

        return train_data_loader, dev_data_loader
