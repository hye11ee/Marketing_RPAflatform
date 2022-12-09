import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from konlpy.tag import Okt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer 
from wordcloud import WordCloud



def get_data_sent_dict(product_name='', count=0):
  """
  product_name : 검색키워드(제품명)
  count : 분석 실행 횟수 -> 추후 함수끼리 연결하면서 변경 필요 
  """
  if product_name == '신라면':
    path = '../data/제품/shin_naver_500.csv'
  elif product_name == '진라면':
    path = '../data/제품/jin_naver.csv'
  elif product_name == '푸르밀 검은콩우유':
    path = '../data/제품/purmil_naver.csv'

  total_review = pd.read_csv(path)
  sample_data=total_review['review']

  return sample_data, total_review






# 감성사전 가져오기
import json
with open('../data/SentiWord_info.json', encoding='utf-8-sig', mode='r') as f:
    data = json.load(f)



def tokenize(sample_data):
    okt=Okt()

    sample_data_normalize=sample_data.apply(okt.normalize)
    
    tokenized_sent=[]
    for i in range(len(sample_data)):
        tokenized_sent.append(okt.morphs(sample_data_normalize[i], stem=True)) 
        
    return tokenized_sent    





# 긍부정 점수계산 함수화
def sent_analyz(tokenized_sent):
    result = []
    for i in range(len(data)):
        for k in range(len(tokenized_sent)):
            if data[i]['word']==tokenized_sent[k]:
                result.append(data[i]['word_root'])
                result.append(data[i]['polarity'])	
                
    #점수계산     
    a=result[1::2]
    change_int=list(map(int, a))   


    try:
        score=(sum(change_int)/len(a))+3
    except:
        score=0
    #print(score)

    return score



# 입력받은 리뷰 데이터들 토큰화해서 점수계산
def all_process_analyze(want_analyze_col):
    token_sent=tokenize(want_analyze_col)
    
    sent_score_list=[]
    for i in range(len(token_sent)):
        sent_score_list.append(sent_analyz(token_sent[i]))
    total_score = round(sum(sent_score_list)/len(sent_score_list), 2)
    return total_score, sent_score_list






# 긍부정 문장비율 파이차트 반환
def pos_neg_ratio(sample_data):
    
    total_score, sent_score_list = all_process_analyze(sample_data)
    pos_sent=[]
    neg_sent=[]

    for score in sent_score_list:
        if score > 3.3:
            pos_sent.append(score)
        elif score < 2.7:
            neg_sent.append(score)    

    ratio = [len(pos_sent)/len(sent_score_list), len(neg_sent)/len(sent_score_list)]
    labels = ['Postive', 'Negative']
    explode = [0, 0.10]
    colors = ['skyblue', 'tomato']
    plt.title('Pos, Neg Review Ratio')
    plt.pie(ratio, explode = explode,colors=colors, labels=labels, autopct='%.1f%%', startangle=90)
    plt.show()




# 긍부정 단어 워드 클라우드 ----------------------------------------

def word_key_cloud(tokenized_sent):
    
    pos_word=[]
    neg_word=[]
    for i in range(len(data)):
        for k in range(len(tokenized_sent)):
            if data[i]['word']==tokenized_sent[k]:
                if data[i]['polarity'] in ('1', '2'):
                    pos_word.append(data[i]['word_root'])
                elif data[i]['polarity'] in ('-1', '-2'):
                    neg_word.append(data[i]['word_root'])   
                
    

    return pos_word, neg_word



def seprate_pos_neg(tokenized_sent):
    # n개문장 word_key_cloud 함수 적용하기
    word_list=[]
    for i in range(len(tokenized_sent)):
        word_list.append(word_key_cloud(tokenized_sent[i]))

    # 감성사전 통과한 단어들 리스트2개에서 빼내서 긍부정 리스트에 각각 담기
    pos_word=[]
    neg_word=[]

    for list in word_list:
        pos_word.append(list[0])
        neg_word.append(list[1])

    # 긍정)) 감성사전 통과한 단어들 리스트2개에서 빼내서 담기
    pos_filterd_word=[]
    for pos_one_word in pos_word:
        for word in pos_one_word:
            pos_filterd_word.append(word)
    

    # 부정)) 감성사전 통과한 단어들 리스트2개에서 빼내서 담기
    neg_filterd_word=[]
    for neg_one_word in neg_word:
        for word in neg_one_word:
            neg_filterd_word.append(word)
    

    return pos_filterd_word, neg_filterd_word





def make_cloud(filterd_word):
    pos_or_neg_num = Counter(filterd_word)
    # 긍정 워드클라우드
    wc = WordCloud(font_path='malgun', width=1000, height=800, scale=2.0, background_color='white', colormap='ocean').generate_from_frequencies(pos_or_neg_num)
    plt.figure(frameon=False)
    plt.axis('off')
    return plt.imshow(wc)








