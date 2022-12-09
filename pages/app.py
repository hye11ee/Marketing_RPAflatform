import streamlit as st
import numpy as np 
import pandas as pd 
import time 
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from plotly import express as px
from tkinter.tix import COLUMN
from pyparsing import empty
from bertopic import BERTopic
import matplotlib.pyplot as plt
from PIL import Image



from cluster_analysis import get_data_to_cluster, cluster_review, vizualize_review_cluster, get_topic_info_

# layout -------------------------
st.set_page_config(
    page_title = "EDEN T&S's RPA platform",
    page_icon = '😃',
    layout = 'wide'
)


# sidebar ----------------------------
sidebar = st.sidebar
# header 
sidebar.write('Automating sentiment data analysis on the')
sidebar.header("EDEN T&S's RPA platform")
product=sidebar.text_input('상품명 입력')
# multiselect
options = sidebar.multiselect('옵션 설정',
    ['모두','긍정', '부정'])



empty1,con1,empty2 = st.columns([0.01,1.0,0.01])
empyt1,con2,con3,empty2 = st.columns([0.01,0.5,0.5,0.01])
empyt1,con4,con5,empty2 = st.columns([0.01,0.5,0.5,0.01])
empyt1,con6,con7,empty2 = st.columns([0.01,0.5,0.5,0.01])

if sidebar.button('검색'):
    total_review = get_data_to_cluster(product)
    BERTopic_model = cluster_review(total_review)
    loaded_model = BERTopic.load("./my_model")


    # 제품 이미지 추출
    if product == '신라면':
        sidebar.image('http://image.nongshim.com/non/pro/1647822522999.jpg', width=200)
    elif product == '진라면':
        sidebar.image('https://w.namu.la/s/7be98827b4f1737927ca16823f2136b72f163fa5d529be61fc9d18d749dfbf1d292346399de19d18825a373655bdbc3e4e0ae2c3b19c3e4ea29ed6e4ccd27c905999cb3628f9389af6b689a8f212ca18e3251235302c52950ccde1e9870ca66add9e4d19d3808c92d360d068ddf66831', width = 200)
    elif product == '푸르밀 검은콩우유':
        sidebar.image('https://shopping-phinf.pstatic.net/main_2657485/26574850523.20210330171245.jpg?type=f640', width = 200)
    else:
        sidebar.write('검색어를 정확히 입력해주세요.')

    # 시각화 대시보드
    with empty1 :
        empty() # 여백부분1
   
    with con1 :
        st.markdown(f"### 지금부터 {product}의 소비자 평가 결과를 보여드리겠습니다. ")

    with con2 :
        st.markdown(f'### {product}의 고객 리뷰 긍부정 비율')
        if product == '신라면':
            path = '../data/시각화/shin_0_ratio.png'
        elif product == '진라면':
            path = '../data/시각화/jin_0_ratio.png'
        elif product == '푸르밀 검은콩우유':
            path = '../data/시각화/purmil_0_ratio.png'        
        image = Image.open(path)
        st.image(image)

    with con3 :
        st.markdown("### Second Chart")

    with con4 :
        st.markdown("### positive wordcloud")
        if product == '신라면':
            path = '../data/시각화/shin_pos_0_cloud.png'
        elif product == '진라면':
            path = '../data/시각화/jin_pos_0_cloud.png'
        elif product == '푸르밀 검은콩우유':
            path = '../data/시각화/purmil_pos_0_cloud.png'
        image = Image.open(path)
        st.image(image)

    with con5 :
        st.markdown("### negative wordcloud")
        if product == '신라면':
            path = '../data/시각화/shin_neg_0_cloud.png'
        elif product == '진라면':
            path = '../data/시각화/jin_neg_0_cloud.png'
        elif product == '푸르밀 검은콩우유':
            path = '../data/시각화/purmil_neg_0_cloud.png'        
        image = Image.open(path)
        st.image(image)

    with con6 :
        st.markdown('### clustering plot')   
        fig = vizualize_review_cluster(loaded_model)
        fig.write_html("./file.html")
        st.write(fig)

    with con7 :
        st.markdown('### reviews')   
        topic = get_topic_info_(loaded_model)
        st.dataframe(topic)

    with empty2 :
        empty() # 여백부분2

