# 作业三：实现Word2Vec的CBOW

## 作业要求

基于提供的Python文件/Jupyter Notebook文件，以代码填空的形式，实现Word2Vec的连续词袋模型（CBOW）的相关代码，填空完毕后，需展示代码中相应测试部分的输出结果。

本次作业共计15分。提示：只需填写代码中TODO标记的空缺位置即可，具体的代码说明和评分细则详见提供的脚本文件。

## 提交方式

以下两种方式选择其一提交至Canvas平台即可：
1. 补全`w2v.ipynb`代码后运行获得结果，并把notebook导出为`w2v.pdf`，将两者打包提交。
2. 补全`w2v.py`文件，完成实验报告，报告中要求对代码作必要的说明并展示运行结果。

## 文件说明

```
├── data
│   ├── debug.txt       # 用于debug的小语料
│   ├── synonyms.json   # 用于测试词向量的数据
│   └── treebank.txt    # 用于训练词向量的语料
├── README.md
├── w2v.ipynb
└── w2v.py
```