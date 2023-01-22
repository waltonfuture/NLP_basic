import json
import random
from tqdm import tqdm
import time
import pickle
import numpy as np
from typing import Dict, Tuple
from collections import Counter
import warnings
import os
from typing import List
import re
import importlib
import sys

assert sys.version_info[0] == 3
assert sys.version_info[1] >= 6

requirements = ["numpy", "tqdm"]
_OK = True

for name in requirements:
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"Require: {name}")
        _OK = False

if not _OK:
    exit(-1)
else:
    print("All libraries are satisfied.")




# ### 分词器
#
# 该分词器会：
# 1. 将所有字母转为小写；
# 2. 将句子分为连续的字母序列（word）


def tokenizer(line: str) -> List[str]:
    line = line.lower()
    tokens = list(filter(lambda x: len(x) > 0, re.split(r"\W", line)))
    return tokens


print(tokenizer("It's  useful. "))


# ### 数据集类
# 通过设定窗长`window_size`，该数据集类会读取`corpus`中的行，并解析返回`(context, target)`元组。
# 假如一个句子序列为`a b c d e`，且此时`window_size=2`，`Dataset`会返回：
# ```
# ([b, c], a)
# ([a, c, d], b)
# ([a, b, d, e], c)
# ([b, c, e], d)
# ([c, d], e)
# ```

class Dataset:
    def __init__(self, corpus: str, window_size: int):
        """
        :param corpus: 语料路径
        :param window_size: 窗口长度
        """
        self.corpus = corpus
        self.window_size = window_size

    def __iter__(self):
        with open(self.corpus, encoding="utf8") as f:
            for line in f:
                tokens = tokenizer(line)
                if len(tokens) <= 1:
                    continue
                for i, target in enumerate(tokens):
                    left_context = tokens[max(0, i - self.window_size): i]
                    right_context = tokens[i + 1: i + 1 + self.window_size]
                    context = left_context + right_context
                    yield context, target

    def __len__(self):
        """ 统计样本语料中的样本个数 """
        len_ = getattr(self, "len_", None)
        if len_ is not None:
            return len_

        len_ = 0
        for _ in iter(self):
            len_ += 1

        setattr(self, "len_", len_)
        return len_


debug_dataset = Dataset("./data/debug.txt", window_size=3)
print(len(debug_dataset))

for i, pair in enumerate(iter(debug_dataset)):
    print(pair)
    if i >= 3:
        break

del debug_dataset

# ### 词表类
#
# `Vocab`可以用`token_to_idx`把token(str)映射为索引(int)，也可以用`idx_to_token`找到索引对应的token。
#
# 实例化`Vocab`有两种方法：
# 1. 读取`corpus`构建词表。
# 2. 通过调用`Vocab.load_vocab`，可以从已训练的中构建`Vocab`实例。


class Vocab:
    VOCAB_FILE = "vocab.txt"
    UNK = "<unk>"

    def __init__(self, corpus: str = None, max_vocab_size: int = -1):
        """
        :param corpus:         语料文件路径
        :param max_vocab_size: 最大词表数量，-1表示不做任何限制
        """
        self._token_to_idx: Dict[str, int] = {}
        self.token_freq: List[Tuple[str, int]] = []

        if corpus is not None:
            self.build_vocab(corpus, max_vocab_size)

    def build_vocab(self, corpus: str, max_vocab_size: int = -1):
        """ 统计词频，并保留高频词 """
        counter = Counter()
        with open(corpus, encoding="utf8") as f:
            for line in f:
                tokens = tokenizer(line)
                counter.update(tokens)

        print(f"总Token数: {sum(counter.values())}")

        # 将找到的词按照词频从高到低排序
        self.token_freq = [(self.UNK, 1)] + sorted(counter.items(), key=lambda x: x[1], reverse=True)
        if max_vocab_size > 0:
            self.token_freq = self.token_freq[:max_vocab_size]

        print(f"词表大小: {len(self.token_freq)}")

        for i, (token, _freq) in enumerate(self.token_freq):
            self._token_to_idx[token] = i

    def __len__(self):
        return len(self.token_freq)

    def __contains__(self, token: str):
        return token in self._token_to_idx

    def token_to_idx(self, token: str, warn: bool = False) -> int:
        """ Map the token to index """
        token = token.lower()
        if token not in self._token_to_idx:
            if warn:
                warnings.warn(f"{token} => {self.UNK}")
            token = self.UNK
        return self._token_to_idx[token]

    def idx_to_token(self, idx: int) -> str:
        """ Map the index to token """
        assert 0 <= idx < len(self)
        return self.token_freq[idx][0]

    def save_vocab(self, path: str):
        with open(os.path.join(path, self.VOCAB_FILE), "w", encoding="utf8") as f:
            lines = [f"{token} {freq}" for token, freq in self.token_freq]
            f.write("\n".join(lines))

    @classmethod
    def load_vocab(cls, path: str):
        vocab = cls()

        with open(os.path.join(path, cls.VOCAB_FILE), encoding="utf8") as f:
            lines = f.read().split("\n")

        for i, line in enumerate(lines):
            token, freq = line.split()
            vocab.token_freq.append((token, int(freq)))
            vocab._token_to_idx[token] = i

        return vocab


debug_vocab = Vocab("./data/debug.txt")
print(debug_vocab.token_freq)
del debug_vocab




def one_hot(dim: int, idx: int) -> np.ndarray:
    # TODO: 实现one-hot函数
    v = np.zeros(dim)
    v[idx] = 1
    return v


print(one_hot(4, 1))




def softmax(x: np.ndarray) -> np.ndarray:
    # TODO: 实现softmax函数
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)),axis=0)
    #return np.exp(x) / np.sum(np.exp(x), axis=0)


print(softmax(np.array([i for i in range(10)])))



class CBOW:
    def __init__(self, vocab: Vocab, vector_dim: int):
        self.vocab = vocab
        self.vector_dim = vector_dim

        self.U = np.random.uniform(-1, 1, (len(self.vocab), self.vector_dim))  # vocab_size x vector_dim
        self.V = np.random.uniform(-1, 1, (self.vector_dim, len(self.vocab)))  # vector_dim x vocab_size

    def train(self, corpus: str, window_size: int, train_epoch: int, learning_rate: float, save_path: str = None):
        dataset = Dataset(corpus, window_size)
        start_time = time.time()

        for epoch in range(1, train_epoch + 1):
            self.train_one_epoch(epoch, dataset, learning_rate)
            if save_path is not None:
                self.save_model(save_path)

        end_time = time.time()
        print(f"总耗时 {end_time - start_time:.2f}s")

    def train_one_epoch(self, epoch: int, dataset: Dataset, learning_rate: float):
        steps, total_loss = 0, 0.0

        with tqdm(iter(dataset), total=len(dataset), desc=f"Epoch {epoch}", ncols=80) as pbar:
            for sample in pbar:
                context_tokens, target_token = sample
                loss = self.train_one_step(context_tokens, target_token, learning_rate)
                total_loss += loss
                steps += 1
                if steps % 10 == 0:
                    pbar.set_postfix({"Avg. loss": f"{total_loss / steps:.2f}"})

        return total_loss / steps

    def train_one_step(self, context_tokens: List[str], target_token: str, learning_rate: float) -> float:
        """
        :param context_tokens:  目标词周围的词
        :param target_token:    目标词
        :param learning_rate:   学习率
        :return:    loss值 (标量)
        """
        C = len(context_tokens)
        # TODO: 构造输入向量和目标向量
        # context: 构造输入向量
        # target:  目标one-hot向量
        '''context = []
        for token in context_tokens:
            context.append(one_hot(len(self.vocab),self.vocab.token_to_idx(token)))
        context = np.array(context).T 
        context_idx = np.array([self.vocab.token_to_idx(context_tokens[i]) for i in range(C)]) 
        target = one_hot(len(self.vocab),self.vocab.token_to_idx(target_token)) 
        # TODO: 前向步骤
        h = np.dot(self.U.T,context).sum(axis=1) / C 
        y = np.dot(self.V.T,h) 
        pre = softmax(y) 
        # TODO: 计算loss
        loss = - y[self.vocab.token_to_idx(target_token)] + np.log(np.sum(np.exp(y))) 
        # TODO: 更新参数
        err = np.subtract(pre,target) #
        dl_dw1 = np.dot(self.V, err) / C 
        dl_dw2 = np.outer(h, err)  
        for i in range(C):
            self.U[context_idx[i]] -= learning_rate / C * dl_dw1 
        self.V -= (learning_rate * dl_dw2)  '''
        vocab_len = len(self.vocab)
        target_id = self.vocab.token_to_idx(target_token) # 目标词的id
        target = one_hot(vocab_len, target_id) # 目标词的向量
        context_id = np.array([self.vocab.token_to_idx(context_tokens[i]) for i in range(C)]) # 目标词周围的词的id
        context = np.array([one_hot(vocab_len, context_id[i]) for i in range(C)]) # 目标词周围的词的向量
        # TODO: 前向步骤
        h = np.average(np.dot(np.transpose(self.U), np.transpose(context)), axis=1) # hidden state
        y = np.dot(np.transpose(self.V), h) # 预测值
        pre = softmax(y) # 经过softmax层
        err = pre - target # 计算误差向量

        # TODO: 计算loss
        loss = -y[target_id] + np.log(np.sum(np.exp(y))) # 计算熵值
        # TODO: 更新参数
        dEdU = np.dot(err, np.transpose(self.V)) # 损失函数对U的梯度
        dEdV = np.outer((h), err) # 损失函数对V的梯度
        for i in range(C):
            self.U[context_id[i]] -= learning_rate * dEdU / C # 对U反向传播
        self.V -= learning_rate * dEdV # 对V反向传播
        return loss

    def similarity(self, token1: str, token2: str):
        """ 计算两个词的相似性 """
        v1 = self.U[self.vocab.token_to_idx(token1)]
        v2 = self.U[self.vocab.token_to_idx(token2)]
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        return np.dot(v1, v2)

    def most_similar_tokens(self, token: str, n: int):
        """ 召回与token最相似的n个token """
        norm_U = self.U / np.linalg.norm(self.U, axis=1, keepdims=True)

        idx = self.vocab.token_to_idx(token, warn=True)
        v = norm_U[idx]

        cosine_similarity = np.dot(norm_U, v)
        nbest_idx = np.argsort(cosine_similarity)[-n:][::-1]

        results = []
        for idx in nbest_idx:
            _token = self.vocab.idx_to_token(idx)
            results.append((_token, cosine_similarity[idx]))

        return results

    def save_model(self, path: str):
        """ 将模型保存到`path`路径下，如果不存在`path`会主动创建 """
        os.makedirs(path, exist_ok=True)
        self.vocab.save_vocab(path)

        with open(os.path.join(path, "wv.pkl"), "wb") as f:
            param = {"U": self.U, "V": self.V}
            pickle.dump(param, f)

    @classmethod
    def load_model(cls, path: str):
        """ 从`path`加载模型 """
        vocab = Vocab.load_vocab(path)

        with open(os.path.join(path, "wv.pkl"), "rb") as f:
            param = pickle.load(f)

        U, V = param["U"], param["V"]
        model = cls(vocab, U.shape[1])
        model.U, model.V = U, V

        return model


# ## 测试
#
# 测试部分可用于验证CBOW实现的正确性
#
# ### 测试1
#
# 本测试可用于调试，最终一个epoch的平均loss约为0.5，并且“i”、“he”和“she”的相似性较高。


def test1():
    random.seed(42)
    np.random.seed(42)

    vocab = Vocab(corpus="./data/debug.txt")
    cbow = CBOW(vocab, vector_dim=8)
    cbow.train(corpus="./data/debug.txt", window_size=3,
               train_epoch=10, learning_rate=1.0)

    print(cbow.most_similar_tokens("i", 5))
    print(cbow.most_similar_tokens("he", 5))
    print(cbow.most_similar_tokens("she", 5))


test1()




# ### 测试2
#
# 本测试将会在`treebank.txt`上训练词向量，为了加快训练流程，实验只保留高频的4000词，且词向量维度为20。
#
# 在每个epoch结束后，会在`data/treebank.txt`中测试词向量的召回能力。如下所示，`data/treebank.txt`中每个样例为`word`以及对应的同义词，同义词从wordnet中获取。
#
# ```
# [
#   "about",
#   [
#     "most",
#     "virtually",
#     "around",
#     "almost",
#     "near",
#     "nearly",
#     "some"
#   ]
# ]
# ```
#

#
# > 最后一个epoch平均loss降至5.1左右，并且在同义词上的召回率约为20%左右


def calculate_recall_rate(model: CBOW, word_synonyms: List[Tuple[str, List[str]]], topn: int) -> float:
    """ 测试CBOW的召回率 """
    hit, total = 0, 1e-9
    for word, synonyms in word_synonyms:
        synonyms = set(synonyms)
        recalled = set([w for w, _ in model.most_similar_tokens(word, topn)])
        hit += len(synonyms & recalled)
        total += len(synonyms)

    print(f"Recall rate: {hit / total:.2%}")
    return hit / total


def test2():
    random.seed(42)
    np.random.seed(42)

    corpus = "./data/treebank.txt"
    lr = 1e-1
    topn = 40

    vocab = Vocab(corpus, max_vocab_size=4000)
    model = CBOW(vocab, vector_dim=20)

    dataset = Dataset(corpus, window_size=4)

    with open("data/synonyms.json", encoding="utf8") as f:
        word_synonyms: List[Tuple[str, List[str]]] = json.load(f)

    for epoch in range(1, 11):
        model.train_one_epoch(epoch, dataset, learning_rate=lr)
        calculate_recall_rate(model, word_synonyms, topn)


test2()
