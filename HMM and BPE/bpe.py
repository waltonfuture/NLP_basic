


import re
import functools


# 构建空格分词器，将语料中的句子以空格切分成单词，然后将单词拆分成字母加`</w>`的形式。例如`apple`将变为`a p p l e </w>`。
_splitor_pattern = re.compile(r"[^a-zA-Z']+|(?=')")
_digit_pattern = re.compile(r"\d+")

def white_space_tokenize(corpus):
    """
    先正则化（字母转小写、数字转为N、除去标点符号），然后以空格分词语料中的句子，例如：
    输入 corpus=["I am happy.", "I have 10 apples!"]，
    得到 [["i", "am", "happy"], ["i", "have", "N", "apples"]]

    Args:
        corpus: List[str] - 待处理的语料

    Return:
        List[List[str]] - 二维List，内部的List由每个句子的单词str构成
    """

    tokeneds = [list(
        filter(lambda tkn: len(tkn)>0, _splitor_pattern.split(_digit_pattern.sub("N", stc.lower())))) for stc in corpus
    ]
    
    return tokeneds


# 编写相应函数构建BPE算法需要用到的初始状态词典
def build_bpe_vocab(corpus):
    """
    将语料进行white_space_tokenize处理后，将单词每个字母以空格隔开、结尾加上</w>后，构建带频数的字典，例如：
    输入 corpus=["I am happy.", "I have 10 apples!"]，
    得到
    {
        'i </w>': 2,
        'a m </w>': 1,
        'h a p p y </w>': 1,
        'h a v e </w>': 1,
        'N </w>': 1,
        'a p p l e s </w>': 1
     }

    Args:
        corpus: List[str] - 待处理的语料

    Return:
        Dict[str, int] - "单词分词状态->频数"的词典
    """

    tokenized_corpus = white_space_tokenize(corpus)

    bpe_vocab = dict()
    
    # TODO: 完成函数体（1分）
    # 对每个句子中的每个词单独处理🙂
    for se in tokenized_corpus:
        for word in se:
            ch = " ".join(word) # 将单词每个字母以空格隔开
            ch += ' </w>' # 结尾加上</w>
            if ch in bpe_vocab:
                bpe_vocab[ch] += 1 # 构建带频数的字典（表中有该单词则词频增加，没有则创立新的键值对）
            else:
                bpe_vocab[ch] = 1
    return bpe_vocab


# 编写所需的其他函数
def get_bigram_freq(bpe_vocab):
    """
    统计"单词分词状态->频数"的词典中，各bigram的频次（假设该词典中，各个unigram以空格间隔），例如：
    输入 bpe_vocab=
    {
        'i </w>': 2,
        'a m </w>': 1,
        'h a p p y </w>': 1,
        'h a v e </w>': 1,
        'N </w>': 1,
        'a p p l e s </w>': 1
    }
    得到
    {
        ('i', '</w>'): 2,
        ('a', 'm'): 1,
        ('m', '</w>'): 1,
        ('h', 'a'): 2,
        ('a', 'p'): 2,
        ('p', 'p'): 2,
        ('p', 'y'): 1,
        ('y', '</w>'): 1,
        ('a', 'v'): 1,
        ('v', 'e'): 1,
        ('e', '</w>'): 1,
        ('N', '</w>'): 1,
        ('p', 'l'): 1,
        ('l', 'e'): 1,
        ('e', 's'): 1,
        ('s', '</w>'): 1
    }

    Args:
        bpe_vocab: Dict[str, int] - "单词分词状态->频数"的词典

    Return:
        Dict[Tuple(str, str), int] - "bigram->频数"的词典
    """

    bigram_freq = dict()
    
    # TODO: 完成函数体（1分）
    for ch, fr in bpe_vocab.items(): #遍历词表的每个词、词频
        word = ch.split() #先将每个以空格隔开的字符存入列表
        for i in range(1, len(word)):
            if tuple(word[i-1:i+1]) in bigram_freq: #对每个词进行字母两两组合
                bigram_freq[tuple(word[i - 1:i + 1])] += fr #填入新的bigram词表（表中有bigram则词频增加，没有则创立新的键值对）
            else:
                bigram_freq[tuple(word[i - 1:i + 1])] = fr
    return bigram_freq


def refresh_bpe_vocab_by_merging_bigram(bigram, old_bpe_vocab):
    """
    在"单词分词状态->频数"的词典中，合并指定的bigram（即去掉对应的相邻unigram之间的空格），最后返回新的词典，例如：
    输入 bigram=('i', '</w>')，old_bpe_vocab=
    {
        'i </w>': 2,
        'a m </w>': 1,
        'h a p p y </w>': 1,
        'h a v e </w>': 1,
        'N </w>': 1,
        'a p p l e s </w>': 1
    }
    得到
    {
        'i</w>': 2,
        'a m </w>': 1,
        'h a p p y </w>': 1,
        'h a v e </w>': 1,
        'N </w>': 1,
        'a p p l e s </w>': 1
    }

    Args:
        old_bpe_vocab: Dict[str, int] - 初始"单词分词状态->频数"的词典

    Return:
        Dict[str, int] - 合并后的"单词分词状态->频数"的词典
    """
    
    new_bpe_vocab = dict()

    # TODO: 完成函数体（1分）
    '''
    word = " ".join(bigram) #将bigram元组改为以空格为间的字符串
    for ch, fr in old_bpe_vocab.items(): # 遍历旧词表的每个词、词频
        if word in ch: # bigram匹配上了某个词
            lft = ch.index(word) #找到bigram在词中的起始位
            rht = lft + len(word) #找到bigram在词中的结束位
            new_bpe_vocab[ch[0:lft] + "".join(bigram) + ch[rht:]] = fr #合并bigram（即去掉对应的相邻unigram之间的空格）
        else:
            new_bpe_vocab[ch] = fr #bigram未匹配，则直接录入
    return new_bpe_vocab
    '''
    word = " ".join(bigram)  # 将bigram元组改为以空格为间的字符串
    for ch, fr in old_bpe_vocab.items():  # 遍历旧词表的每个词、词频
        if word in ch:  # bigram匹配上了某个词
            lft = 0  # 找到bigram在词中的起始位
            rht = lft + len(word)  # 找到bigram在词中的结束位
            new_ch = ''
            for i in range(ch.count(word)):
                lft1 = ch[lft:].index(word) + len(ch[0:lft])  # 找到bigram在词中的起始位
                rht = lft1 + len(word)  # 找到bigram在词中的结束位
                new_ch += ch[lft:lft1] + "".join(bigram)
                lft = rht
            new_ch += ch[rht:]
            new_bpe_vocab[new_ch] = fr  # 合并bigram（即去掉对应的相邻unigram之间的空格）
        else:
            new_bpe_vocab[ch] = fr  # bigram未匹配，则直接录入
    return new_bpe_vocab


def get_bpe_tokens(bpe_vocab):
    """
    根据"单词分词状态->频数"的词典，返回所得到的BPE分词列表，并将该列表按照分词长度降序排序返回，例如：
    输入 bpe_vocab=
    {
        'i</w>': 2,
        'a m </w>': 1,
        'ha pp y </w>': 1,
        'ha v e </w>': 1,
        'N </w>': 1,
        'a pp l e s </w>': 1
    }
    得到
    [
        ('i</w>', 2),
        ('ha', 2),
        ('pp', 2),
        ('a', 2),
        ('m', 1),
        ('</w>', 5),
        ('y', 1),
        ('v', 1),
        ('e', 2),
        ('N', 1),
        ('l', 1),
        ('s', 1)
     ]

    Args:
        bpe_vocab: Dict[str, int] - "单词分词状态->频数"的词典

    Return:
        List[Tuple(str, int)] - BPE分词和对应频数组成的List
    """
    
    # TODO: 完成函数体（2分）
    bpe_token = dict() #创建字典临时存储
    for ch, fr in bpe_vocab.items():
        word = ch.split() #先将每个以空格隔开的字符存入列表
        for gram in word: #遍历字符
            if gram in bpe_token:
                bpe_token[gram] += fr # 存入字符、字频（表中有该字符则字频增加，没有则创立新的键值对）
            else:
                bpe_token[gram] = fr
    bpe_tokens = []
    for ch, fr in bpe_token.items(): #将统计好的字典输入列表
        bpe_tokens.append((ch, fr))
    # 将该列表按照分词长度排序返回（'</w>'计为一个长度）
    bpe_tokens = sorted(bpe_tokens, key=lambda pair: len(pair[0]) if '</w>' not in pair[0] else len(pair[0]) - 3)
    bpe_tokens.reverse() #改为降序
    return bpe_tokens


def print_bpe_tokenize(word, bpe_tokens):
    """
    根据按长度降序的BPE分词列表，将所给单词进行BPE分词，最后打印结果。
    
    思想是，对于一个待BPE分词的单词，按照长度顺序从列表中寻找BPE分词进行子串匹配，
    若成功匹配，则对该子串左右的剩余部分递归地进行下一轮匹配，直到剩余部分长度为0，
    或者剩余部分无法匹配（该部分整体由"<unknown>"代替）
    
    例1：
    输入 word="supermarket", bpe_tokens=[
        ("su", 20),
        ("are", 10),
        ("per", 30),
    ]
    最终打印 "su per <unknown>"

    例2：
    输入 word="shanghai", bpe_tokens=[
        ("hai", 1),
        ("sh", 1),
        ("an", 1),
        ("</w>", 1),
        ("g", 1)
    ]
    最终打印 "sh an g hai </w>"

    Args:
        word: str - 待分词的单词str
        bpe_tokens: List[Tuple(str, int)] - BPE分词和对应频数组成的List
    """
    
    # TODO: 请尝试使用递归函数定义该分词过程（2分）
    def bpe_tokenize(sub_word):
        if len(sub_word) == 0: #递归结束条件：剩余部分长度为0
            return ""
        for i, pair in enumerate(bpe_tokens): #遍历bpr分词列表
            if pair[0] in sub_word: #成功匹配
                lft = sub_word.index(pair[0]) #找出bpe分词在单词中的起始位
                rht = lft + len(pair[0]) #找出bpe分词在单词中的结束位
                return bpe_tokenize(sub_word[0:lft]) + pair[0] + " " + bpe_tokenize(sub_word[rht:]) #对该子串左右的剩余部分递归地进行下一轮匹配
            else:
                if i == len(bpe_tokens) - 1: #递归结束条件：剩余部分无法匹配（该部分整体由"<unknown>"代替）
                    return "<unknown>" + " "
        return ""

    res = bpe_tokenize(word+"</w>")
    print(res)


# 开始读取数据集并训练BPE分词器
with open("data/news.2007.en.shuffled.deduped.train", encoding="utf-8") as f:
    training_corpus = list(map(lambda l: l.strip(), f.readlines()[:1000]))

print("Loaded training corpus.")

training_iter_num = 300

training_bpe_vocab = build_bpe_vocab(training_corpus)
for i in range(training_iter_num):
    # TODO: 完成训练循环内的代码逻辑（2分）
    bigram_freq = get_bigram_freq(training_bpe_vocab) #创建bigram词表
    max_key = max(bigram_freq, key=bigram_freq.get) #找到bigram词表中最常见的一个bigram
    training_bpe_vocab = refresh_bpe_vocab_by_merging_bigram(max_key, training_bpe_vocab) #将最常见的bigram捏合成新的token，构成新词表

training_bpe_tokens = get_bpe_tokens(training_bpe_vocab)


# 测试BPE分词器的分词效果
test_word = "naturallanguageprocessing"

print("naturallanguageprocessing 的分词结果为：")
print_bpe_tokenize(test_word, training_bpe_tokens)
# result: n atur al lan gu age pro ce s sing</w>