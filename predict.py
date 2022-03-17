from keras.models import load_model
model = load_model('a_best_weight.h5')

from collections import Counter

import numpy as np

import utils
import string
import re
import sys
import json
import pandas as pd
from tqdm import tqdm

alphabet = set('\x00 _' + string.ascii_lowercase + string.digits + ''.join(utils.ACCENTED_TO_BASE_CHAR_MAP.keys()))

print("alphabet",alphabet)
codec = utils.CharacterCodec(alphabet, utils.MAXLEN)

def guess(ngram):
    text = ' '.join(ngram)
    text += '\x00' * (utils.MAXLEN - len(text))
    if utils.INVERT:
        text = text[::-1]
    preds = model.predict(np.array([codec.encode(text)]), verbose=0)
    rtext = codec.decode(preds[0], calc_argmax=False).strip('\x00')
    if len(rtext)>0:
        index = rtext.find('\x00')
        if index>-1:
            rtext = rtext[:index]
    return rtext


def add_accent(text):
    # lowercase the input text as we train the model on lowercase text only
    # but we keep the map of uppercase characters to restore cases in output
    is_uppercase_map = [c.isupper() for c in text]
    text = utils.remove_accent(text.lower())

    outputs = []
    words_or_symbols_list = re.findall('\w[\w ]*|\W+', text)

    # print(words_or_symbols_list)

    for words_or_symbols in words_or_symbols_list:
        if utils.is_words(words_or_symbols):
            outputs.append(_add_accent(words_or_symbols))
        else:
            outputs.append(words_or_symbols)
        # print(outputs)
    output_text = ''.join(outputs)

    # restore uppercase characters
    output_text = ''.join(c.upper() if is_upper else c
                            for c, is_upper in zip(output_text, is_uppercase_map))
    return output_text

def _add_accent(phrase):
    grams = list(utils.gen_ngram(phrase.lower(), n=utils.NGRAM, pad_words=utils.PAD_WORDS_INPUT))
    
    guessed_grams = list(guess(gram) for gram in grams)
    # print("phrase",phrase,'grams',grams,'guessed_grams',guessed_grams)
    candidates = [Counter() for _ in range(len(guessed_grams) + utils.NGRAM - 1)]
    for idx, gram in enumerate(guessed_grams):
        for wid, word in enumerate(re.split(' +', gram)):
            candidates[idx + wid].update([word])
    output = ' '.join(c.most_common(1)[0][0] for c in candidates if c)
    return output.strip('\x00 ')




# print(add_accent('do,'))
# print(add_accent('7.3 inch,'))
# print(add_accent('Truoc do, tren san khau su kien SDC 2018, giam doc cao cap mang marketing san pham di dong cua Samsung, ong Justin Denison da cam tren tay nguyen mau cua thiet bi nay. Ve co ban, no chang khac gi mot chiec may tinh bang 7.3 inch, duoc cau thanh tu nhieu lop phu khac nhau nhu polyme, lop man chong soc, lop phan cuc voi do mong gan mot nua so voi the he truoc, lop kinh linh hoat va mot tam lung da nang co the bien thanh man hinh. Tat ca se duoc ket dinh bang mot loai keo cuc ben, cho phep chiec may nay co the gap lai hang tram ngan lan ma khong bi hu hong.'))
# print(add_accent('man hinh. Tat ca se duoc ket dinh bang mot loai keo cuc ben, cho phep chiec may nay co the gap lai hang tram ngan lan ma khong bi hu hong.'))
# querys = sys.argv[1]

# # print(querys)
# with open(querys,'r',encoding='utf-8') as f:
#     test=json.load(f)

# vnm_char = 'ĂÂÁẮẤÀẰẦẢẲẨÃẴẪẠẶẬĐÊÉẾÈỀẺỂẼỄẸỆÍÌỈĨỊÔƠÓỐỚÒỒỜỎỔỞÕỖỠỌỘỢƯÚỨÙỪỦỬŨỮỤỰÝỲỶỸỴ'
# vnm_char = vnm_char.lower()
# def check_vnm(str):
#     for i in str: 
#         if i in vnm_char: 
#             return True
#     return False

# def remove_last_character(str):
#     while str[-1] in [' ', '.', '!']:
#         str = str[:-1]
#     return str


# for i in tqdm(range(len(test))):
#     previous = test[i]['query'].lower()
#     previous = remove_last_character(previous)
#     final = ''
#     try:
#         last = add_accent(previous).lower()
#         # print(last)
#         previous_list = previous.split(' ')
#         last_list = last.split(' ')
#         if len(previous_list) == len(last_list):
#             # pass
#             for element in range(len(previous_list)):
#                 if check_vnm(previous_list[element]):
#                     last_list[element] = previous_list[element]
#             for element in last_list: 
#                 final += element + ' '
#         else: 
#             final = previous            
#     # break
#     except: 
#         pass
#     test[i]['query'] = final
#     print(previous)
#     print(last)
#     print(final)
#     print()


# with open("./sample.json", "w", encoding="utf-8") as outfile:
#     json.dump(test, outfile)







