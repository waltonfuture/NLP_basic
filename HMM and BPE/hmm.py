

import pickle
import numpy as np


# 导入HMM参数，初始化所需的起始概率矩阵，转移概率矩阵，发射概率矩阵
with open("hmm_parameters.pkl", "rb") as f:
    hmm_parameters = pickle.load(f)

# 非断字（B）为第0行，断字（I）为第1行
# 发射概率矩阵中，词典大小为65536，以汉字的ord作为行key
start_probability = hmm_parameters["start_prob"]  # shape(2,)
trans_matrix = hmm_parameters["trans_mat"]  # shape(2, 2)
emission_matrix = hmm_parameters["emission_mat"]  # shape(2, 65536)


# TODO: 将input_sentence中的xxx替换为你的姓名
input_sentence = "xxx是一名优秀的学生"


# 实现viterbi算法，并以此进行中文分词
def viterbi(sent_orig, start_prob, trans_mat, emission_mat):
    """
    viterbi算法进行中文分词

    Args:
        sent_orig: str - 输入的句子
        start_prob: numpy.ndarray - 起始概率矩阵
        trans_mat: numpy.ndarray - 转移概率矩阵
        emission_mat: numpy.ndarray - 发射概率矩阵

    Return:
        str - 中文分词的结果
    """
    
    #  将汉字转为数字表示
    sent_ord = [ord(x) for x in sent_orig]
    
    # `dp`用来储存不同位置每种标注（B/I）的最大概率值
    dp = np.zeros((2, len(sent_ord)), dtype=float)
    
    # `path`用来储存最大概率对应的上步B/I选择
    #  例如 path[1][7] == 1 意味着第8个（从1开始计数）字符标注I对应的最大概率，其前一步的隐状态为1（I）
    #  例如 path[0][5] == 1 意味着第6个字符标注B对应的最大概率，其前一步的隐状态为1（I）
    #  例如 path[1][1] == 0 意味着第2个字符标注I对应的最大概率，其前一步的隐状态为0（B）
    path = np.zeros((2, len(sent_ord)), dtype=int)
    
    #  TODO: 第一个位置的最大概率值计算
    # 第一个位置即为P(hi)*P(v1|hi)
    for i in range(2):
        dp[i][0] = start_probability[i] * emission_mat[i][sent_ord[0]]

    #  TODO: 其余位置的最大概率值计算（填充dp和path矩阵）
    # 对vt中的hi: 求max(left*P(hi|left_tag))P(vt|hi))
    # 即进行两次对tag的遍历，计算每个位置的概率同时记录每个状态来自于前面哪个状态(N*S^2)
    for i, ch in enumerate(sent_ord[1:]): # 对词序列遍历（N）
        for s in range(2):
            (prob, last_state) = max([(dp[ls, i] * trans_mat[ls][s] * emission_mat[s][ch] ,ls)  for ls in range(2)]) # 动态规划
            dp[s][i+1] = prob # 记录概率
            path[s][i+1] = last_state # 记录路径


    #  `labels`用来储存每个位置最有可能的隐状态
    labels = [0 for _ in range(len(sent_ord))]
    
    #  TODO：计算labels每个位置上的值（填充labels矩阵）
    (end_prob, state) = max([(dp[s][len(sent_ord)-1], s) for s in range(2)]) # 找到最优路径的最后一个词的tag
    labels[len(sent_ord)-1] = state
    for i in range(len(sent_ord) - 1, 0, -1): # 回溯，找到每个词最可能对应的tag
        state = path[state][i] #根据之前记录的path回溯找出最优路径上的每个状态（tag）
        labels[i-1] = state

    #  根据lalels生成切分好的字符串
    sent_split = []
    for idx, label in enumerate(labels):
        if label == 1:
            sent_split += [sent_ord[idx], ord("/")]
        else:
            sent_split += [sent_ord[idx]]
    sent_split_str = "".join([chr(x) for x in sent_split])

    return sent_split_str


# 实现前向算法，计算该句子的概率值
def compute_prob_by_forward(sent_orig, start_prob, trans_mat, emission_mat):
    """
    前向算法，计算输入中文句子的概率值

    Args:
        sent_orig: str - 输入的句子
        start_prob: numpy.ndarray - 起始概率矩阵
        trans_mat: numpy.ndarray - 转移概率矩阵
        emission_mat: numpy.ndarray - 发射概率矩阵

    Return:
        float - 概率值
    """
    
    #  将汉字转为数字表示
    sent_ord = [ord(x) for x in sent_orig]

    # `dp`用来储存不同位置每种隐状态（B/I）下，到该位置为止的句子的概率
    dp = np.zeros((2, len(sent_ord)), dtype=float)

    # TODO: 初始位置概率的计算
    # 与viterbi一样
    for i in range(2):
        dp[i][0] = start_probability[i] * emission_mat[i][sent_ord[0]]
    
    # TODO: 先计算其余位置的概率（填充dp矩阵），然后return概率值
    # 思路与viterbi一样，只是max改成sum，并且不用记录每一步的状态
    for i, ch in enumerate(sent_ord[1:]): # 对词序列遍历（N）
        for s in range(2):
            dp[s][i+1] = sum(dp[ls, i] * trans_mat[ls][s] * emission_mat[s][ch] for ls in range(2)) #进行两次对tag的遍历，计算每个位置的概率（S^2）

    return sum([dp[i][len(sent_ord)-1] for i in range(2)])


# 实现后向算法，计算该句子的概率值
def compute_prob_by_backward(sent_orig, start_prob, trans_mat, emission_mat):
    """
    后向算法，计算输入中文句子的概率值

    Args:
        sent_orig: str - 输入的句子
        start_prob: numpy.ndarray - 起始概率矩阵
        trans_mat: numpy.ndarray - 转移概率矩阵
        emission_mat: numpy.ndarray - 发射概率矩阵

    Return:
        float - 概率值
    """
    
    #  将汉字转为数字表示
    sent_ord = [ord(x) for x in sent_orig]

    # `dp`用来储存不同位置每种隐状态（B/I）下，从结尾到该位置为止的句子的概率
    dp = np.zeros((2, len(sent_ord)), dtype=float)

    # TODO: 终末位置概率的初始化
    # 最后一个词的beta记为1
    n = len(sent_ord) - 1
    for i in range(2):
        dp[i][n] = 1
    
    # TODO: 先计算其余位置的概率（填充dp矩阵），然后return概率值
    # 对vt中的hi: 求sum(right*P(right_tag|hi))P(vt+1|right_tag))
    sent_ord.reverse()
    for i, ch in enumerate(sent_ord[0:-1]): #对词序列逆序遍历（N）
        for s in range(2):
            dp[s][n-i-1] = sum(dp[ls, n-i] * trans_mat[s][ls] * emission_mat[ls][ch] for ls in range(2)) #进行两次对tag的遍历，计算每个位置的概率（S^2）
    sent_ord.reverse()
    return sum([dp[i][0] * start_prob[i] * emission_mat[i][sent_ord[0]] for i in range(2)])




print("viterbi算法分词结果：", viterbi(input_sentence, start_probability, trans_matrix, emission_matrix))
print("前向算法概率：", compute_prob_by_forward(input_sentence, start_probability, trans_matrix, emission_matrix))
print("后向算法概率：", compute_prob_by_backward(input_sentence, start_probability, trans_matrix, emission_matrix))

