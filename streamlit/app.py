import streamlit as st
import pandas as pd
import numpy as np
from pandas import DataFrame
from cluster_analysis import get_data_to_cluster, cluster_review, vizualize_review_cluster, get_topic_info_
import os
import json
import time
from bertopic import BERTopic
from PIL import Image
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# # 구글 스프레드시트 연동
# scope = ["https://spreadsheets.google.com/feeds",
#          "https://www.googleapis.com/auth/spreadsheets",
#          "https://www.googleapis.com/auth/drive.file",
#          "https://www.googleapis.com/auth/drive"]

# creds = ServiceAccountCredentials.from_json_keyfile_name("{sanggyungproject-99b6b69f33f6}.json", scope)

# # 시트 연동
# spreadsheet_name = "{productName}"
# client = gspread.authorize(creds)
# spreadsheet = client.open(spreadsheet_name)

# # 시트불러오기
# for sheet in spreadsheet.worksheets():
#     print(sheet)

scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive.file",
         "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("/Users/luvooya/Desktop/sanggyungproject-99b6b69f33f6.json", scope)

# 시트 연동
spreadsheet_name = "{productName}"
client = gspread.authorize(creds)
spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1HwP_t4z10r5otUG236AxYma8TSmfxEgafmIjA45tIVA/edit?usp=drive_web&ouid=111066283667528504586'

# 스프레스시트 문서 가져오기 
doc = client.open_by_url(spreadsheet_url)
worksheet = doc.worksheet('productName')

result=st.text_input('상품명 입력')


worksheet.append_row([result])






st.set_page_config(
    page_title="소비자 리뷰 감성 분석",
    page_icon="🐬",
    layout = 'wide'
)


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    st.image("logo.jpg", width=200)
    st.title("소비자 리뷰 감성 분석")



with st.expander(" 📃 사용 설명서", expanded=True):

    st.write(
        """     
-   원하는 제품의 시장평판을 알아보세요. 당신의 제품에 인사이트를 제공합니다. (4개 웹사이트에서 0000개의 소비자 리뷰를 모아 분석합니다)
-   사용 방법
    1. 원하는 상품을 등록해주세요. (새로운 상품은 최초등록이 필요해요.)
    2. 원하는 결과 확인시간을 정해주세요. (분석 최소 소요시간은 2시간 입니다.)
    3. 분석이 완료되면 원하는 곳으로 알람을 보내드려요.
    4. 등록없이 분석가능한 제품은 아래 선택창에서 확인 가능합니다.
	    """
    )

    st.markdown("")

st.markdown("")


product = st.selectbox(
    '원하시는 제품을 선택해주세요.',
    ('신라면', '진라면', '푸르밀 검은콩우유'))

total_review = get_data_to_cluster(product)
BERTopic_model = cluster_review(total_review)
loaded_model = BERTopic.load("./my_model")

total = len(total_review)
pos = len(total_review.loc[total_review['score'] > 3.3])
neg = len(total_review.loc[total_review['score'] < 3.3])


st.write(f'{product}의 총 리뷰 개수 : {total}개')




col1, col2, col3 = st.columns(3)
col1.metric("긍정리뷰", f"{pos}개", "-10개")
col2.metric("부정리뷰", f"{neg}개", "20개")
col3.metric("소비자마음", "😄")


st.markdown("## 📌 감성분석 결과 ")

empty1,con1,empty2 = st.columns([0.01,2.0,0.01])
empyt1,con2,con3,empty2 = st.columns([0.01,0.5,0.5,0.01])
empyt1,con4,con5,empty2 = st.columns([0.01,0.5,0.5,0.01])

# 시각화 대시보드
with con1 :
    st.markdown(f'##### <span style="color:gray">{product}의 고객 리뷰 긍부정 비율</span>', unsafe_allow_html=True)
    if product == '신라면':
        path = '../data/시각화/shin_0_ratio.png'
    elif product == '진라면':
        path = '../data/시각화/jin_0_ratio.png'
    elif product == '푸르밀 검은콩우유':
        path = '../data/시각화/purmil_0_ratio.png'        
    image = Image.open(path)
    st.image(image, width=400)

with con2 :
    st.markdown('##### <span style="color:gray"> 긍정리뷰 워드클라우드</span>', unsafe_allow_html=True)
    if product == '신라면':
        path = '../data/시각화/shin_pos_0_cloud.png'
    elif product == '진라면':
        path = '../data/시각화/jin_pos_0_cloud.png'
    elif product == '푸르밀 검은콩우유':
        path = '../data/시각화/purmil_pos_0_cloud.png'
    image = Image.open(path)
    st.image(image)

with con3 :
    st.markdown("##### <span style='color:gray'> 부정리뷰 워드클라우드</span>", unsafe_allow_html=True)
    if product == '신라면':
        path = '../data/시각화/shin_neg_0_cloud.png'
    elif product == '진라면':
        path = '../data/시각화/jin_neg_0_cloud.png'
    elif product == '푸르밀 검은콩우유':
        path = '../data/시각화/purmil_neg_0_cloud.png'        
    image = Image.open(path)
    st.image(image)

with con4 :
    st.markdown(f'##### <span style="color:gray">{product}의 고객 리뷰 분포도</span>', unsafe_allow_html=True)   
    fig = vizualize_review_cluster(loaded_model)
    fig.write_html("./file.html")
    st.write(fig)

with con5 :
    st.markdown('##### <span style="color:gray">군집(Topic)별 대표 리뷰</span>', unsafe_allow_html=True)
    
    tabs = st.tabs(['topic0', 'topic1', 'topic2', 'topic3', 'topic4', 'topic5', 'topic6', 'topic7', 'topic8', 'topic9'])

    for i in range(10):
        with tabs[i]:
            topic = get_topic_info_(loaded_model, i) 
            st.dataframe(topic)


