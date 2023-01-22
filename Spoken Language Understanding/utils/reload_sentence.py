from xpinyin import Pinyin 
from Pinyin2Hanzi import DefaultDagParams
from Pinyin2Hanzi import dag



def hanzi_to_pinyin(str_list):
    dagParams = DefaultDagParams()

    result = dag(dagParams,str_list,path_num=5,log=False)
    return result[0].path


def reload_sentence(sentence):
    translator = Pinyin()
    pinyin = translator.get_pinyin(sentence," ")
    pinyin_list = pinyin.split(" ")
    res = hanzi_to_pinyin(pinyin_list)
    final_res = "".join(res)
    return final_res

