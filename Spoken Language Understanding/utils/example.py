import json
from utils.reload_sentence import reload_sentence
from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict):
        super(Example, self).__init__()
        self.ex = ex

        #self.utt = ex['manual_transcript'] if 'manual_transcript' in ex else ex['asr_1best']
        self.utt = ex['manual_transcript'] if 'manual_transcript' in ex else reload_sentence(ex['asr_1best'])
        self.utt.replace("(unknown)", '[UNK]')
        self.utt.replace("(noise)", '[UNK]')
        self.utt.replace("(dialect)", '[UNK]')
        self.utt.replace("(side)", '[UNK]')
        self.slot = {}
        if 'semantic' in ex:
            for label in ex['semantic']:
                act_slot = f'{label[0]}-{label[1]}'
                if len(label) == 3:
                    self.slot[act_slot] = label[2]
            self.tags = ['O'] * len(self.utt)
            for slot in self.slot:
                value = self.slot[slot]
                bidx = self.utt.find(value)
                if bidx != -1:
                    self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                    self.tags[bidx] = f'B-{slot}'
            self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
            self.input_idx = [Example.word_vocab[c] for c in self.utt]
            l = Example.label_vocab
            self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
        else:
            self.input_idx = [Example.word_vocab[c] for c in self.utt]

from transformers import AutoTokenizer, BertTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
#tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
#tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-macbert-base')
class Example_Bert():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict):
        super(Example_Bert, self).__init__()

        self.ex = ex
        max_input_len = 64
        self.utt = ex['asr_1best']
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        #self.input_idx = tokenizer(self.utt,truncation=True, padding=True, max_length=max_input_len)['input_ids']
        #self.attention_mask = tokenizer(self.utt,truncation=True, padding=True, max_length=max_input_len)['attention_mask']
        #self.token_type_ids = tokenizer(self.utt,truncation=True, padding=True, max_length=max_input_len)['token_type_ids']
        self.input_idx = tokenizer(self.utt, truncation=True, padding=True)['input_ids']
        self.attention_mask = tokenizer(self.utt, truncation=True, padding=True)[
            'attention_mask']
        self.token_type_ids = tokenizer(self.utt, truncation=True, padding=True)[
            'token_type_ids']
        l = Example_Bert.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
        self.tag_id = self.tag_id[0:len(self.input_idx)-2]


class Example_Bert_manual():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        datas = json.load(open(data_path, 'r'))
        examples = []
        for data in datas:
            for utt in data:
                ex = cls(utt)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict):
        super(Example_Bert_manual, self).__init__()

        self.ex = ex
        max_input_len = 64
        #self.utt = ex['asr_1best']
        self.utt = ex['manual_transcript'] if 'manual_transcript' in ex else ex['asr_1best']
        #self.utt = ex['manual_transcript'] if 'manual_transcript' in ex else reload_sentence(ex['asr_1best'])
        self.utt.replace("(unknown)", '[UNK]')
        self.utt.replace("(noise)", '[UNK]')
        self.utt.replace("(dialect)", '[UNK]')
        self.utt.replace("(side)", '[UNK]')
        self.slot = {}
        if 'semantic' in ex:
            for label in ex['semantic']:
                act_slot = f'{label[0]}-{label[1]}'
                if len(label) == 3:
                    self.slot[act_slot] = label[2]
            self.tags = ['O'] * len(self.utt)
            for slot in self.slot:
                value = self.slot[slot]
                bidx = self.utt.find(value)
                if bidx != -1:
                    self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                    self.tags[bidx] = f'B-{slot}'
            self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
            self.input_idx = tokenizer(self.utt, truncation=True, padding=True)['input_ids']
            self.attention_mask = tokenizer(self.utt, truncation=True, padding=True)[
                'attention_mask']
            self.token_type_ids = tokenizer(self.utt, truncation=True, padding=True)[
                'token_type_ids']
            l = Example_Bert_manual.label_vocab
            self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
            self.tag_id = self.tag_id[0:len(self.input_idx) - 2]
        else:
            self.input_idx = tokenizer(self.utt, truncation=True, padding=True)['input_ids']
            self.attention_mask = tokenizer(self.utt, truncation=True, padding=True)[
                'attention_mask']
            self.token_type_ids = tokenizer(self.utt, truncation=True, padding=True)[
                'token_type_ids']
        #self.input_idx = tokenizer(self.utt,truncation=True, padding=True, max_length=max_input_len)['input_ids']
        #self.attention_mask = tokenizer(self.utt,truncation=True, padding=True, max_length=max_input_len)['attention_mask']
        #self.token_type_ids = tokenizer(self.utt,truncation=True, padding=True, max_length=max_input_len)['token_type_ids']




