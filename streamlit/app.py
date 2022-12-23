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

# page setting
st.set_page_config(
    page_title="소비자 리뷰 감성 분석",
    page_icon="🐬",
    layout = 'wide'
)

# banner image
st.image("banner.jpg")

dash, info = st.tabs(['Dash Board', 'Infomation'])
# google sheet 연동
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive.file",
         "https://www.googleapis.com/auth/drive"]

#creds = ServiceAccountCredentials.from_json_keyfile_name(st.secrets["api_key"], scope)

spreadsheet_name = "productName"
client = gspread.authorize(st.secrets["api_key"])
spreadsheet_url = 'https://docs.google.com/spreadsheets/d/1HwP_t4z10r5otUG236AxYma8TSmfxEgafmIjA45tIVA/edit?usp=drive_web&ouid=111066283667528504586'

doc = client.open_by_url(spreadsheet_url)
worksheet = doc.worksheet('productName')

with dash :
    with st.expander(" 📃 사용 설명서", expanded=True):

        st.write(
            """     
    -   원하는 제품의 시장평판을 알아보세요. 당신의 제품에 인사이트를 제공합니다.
    -   사용 방법
        1. 원하는 상품을 등록해주세요. (새로운 상품은 최초등록이 필요해요.)
        2. 원하는 결과 확인시간을 정해주세요. (분석 최소 소요시간은 2시간 입니다.)
        3. 분석이 완료되면 아래의 선택 박스에 제품명이 추가됩니다.
        4. 선택 박스의 제품명 클릭시 분석된 결과를 보여줍니다.
            """
        )

        st.markdown("")

    st.markdown("")

    # 제품 추가 입력창
    result=st.text_input('상품명 입력', 
                        help = '''원하는 상품을 입력 후 Enter key를 눌러주세요.
                        \n등록된 상품은 약 2시간 이후 분석이 완료됩니다.
                        \n분석 결과는 아래의 선택박스에 저장되어 7일 단위로 상품 리뷰 결과를 보여드립니다.''',
                        max_chars = 15)
    worksheet.append_row([result]) # 구글 스프레드시트 정보 추가

    product = st.selectbox(
        '원하시는 제품을 선택해주세요.',
        ('신라면', '진라면', '푸르밀 검은콩우유'))

    total_review = get_data_to_cluster(product)
    BERTopic_model = cluster_review(total_review)
    loaded_model = BERTopic.load("./my_model")

    total = len(total_review)
    pos = len(total_review.loc[total_review['score'] > 3.3])
    neg = len(total_review.loc[total_review['score'] < 2.7])


    st.write(f'{product}의 총 리뷰 개수 : {total}개')




    col1, col2, col3 = st.columns(3)
    col1.metric("긍정리뷰", f"{pos}개")
    col2.metric("부정리뷰", f"{neg}개")
    col3.metric("소비자마음", "😄")


    st.markdown("## 📌 감성분석 결과 ")

    empyt1,con1,con2,empty2 = st.columns([0.01,0.5,0.5,0.01])
    empyt1,con3,con4,empty2 = st.columns([0.01,0.5,0.5,0.01])
    empyt1,con5,con6,empty2 = st.columns([0.01,0.5,0.5,0.01])

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
        st.image(image)

    with con2 :
        st.markdown(f'##### <span style="color:gray">{product}의 최신 뉴스 동향</span>', unsafe_allow_html=True)
        if product == '신라면':
            path = '../data/news_scraping/shin_news.xls'
        elif product == '진라면':
            path = '../data/news_scraping/jin_news.xls'
        elif product == '푸르밀 검은콩우유':
            path = '../data/news_scraping/pur_news.xls'
        news = pd.read_excel(path)
        st.dataframe(news)


    with con3 :
        st.markdown('##### <span style="color:gray"> 긍정리뷰 워드클라우드</span>', unsafe_allow_html=True)
        if product == '신라면':
            path = '../data/시각화/shin_pos_0_cloud.png'
        elif product == '진라면':
            path = '../data/시각화/jin_pos_0_cloud.png'
        elif product == '푸르밀 검은콩우유':
            path = '../data/시각화/purmil_pos_0_cloud.png'
        image = Image.open(path)
        st.image(image, width=600)

    with con4 :
        st.markdown("##### <span style='color:gray'> 부정리뷰 워드클라우드</span>", unsafe_allow_html=True)
        if product == '신라면':
            path = '../data/시각화/shin_neg_0_cloud.png'
        elif product == '진라면':
            path = '../data/시각화/jin_neg_0_cloud.png'
        elif product == '푸르밀 검은콩우유':
            path = '../data/시각화/purmil_neg_0_cloud.png'        
        image = Image.open(path)
        st.image(image, width=600)

    with con5 :
        st.markdown(f'##### <span style="color:gray">{product}의 고객 리뷰 분포도</span>', unsafe_allow_html=True)   
        fig = vizualize_review_cluster(loaded_model)
        fig.write_html("./file.html")
        st.write(fig)

    with con6 :
        st.markdown('##### <span style="color:gray">군집(Topic)별 대표 리뷰</span>', unsafe_allow_html=True)
        
        tabs = st.tabs(['topic0', 'topic1', 'topic2', 'topic3', 'topic4', 'topic5', 'topic6', 'topic7', 'topic8', 'topic9'])

        for i in range(10):
            with tabs[i]:
                topic = get_topic_info_(loaded_model, i) 
                st.dataframe(topic)


with info:
    with st.expander(" 📃 참고 페이지 ", expanded=True):

        st.write(
        """     
    -   [ 소개글 ]   
          경북대학교 K-Digital Training 교육생들 중 상경계열 학생 4인이 만든 Marketing Automation Platform 입니다!   
          소비자 리뷰를 통해 제품에 가진 감정을 긍정, 부정으로 분류 및 리뷰 클러스터링을 통해 인사이트를 도출해냅니다.   
          이는 마케터들의 반복되는 리뷰 분석 및 다채널 관리 업무의 번거로움을 줄여주며 보다 편리한 업무 환경 개선을 기대할 수 있습니다.   

    -   Notion : https://www.notion.so/aravis0309/TeamProject-8ddaebd21c414dbdae9fcc14f93e1813   
    -   개발자 이메일   
        강혜리 : kangjeon22@naver.com   
        김재열 : jaeyeol5621@gmail.com   
        심정윤 : aravis0309@knu.ac.kr    
        이혜진 : jetlag6060@gmail.com   
        """
    )
    st.markdown("")

    em1,c1,c2,em2 = st.columns([0.01,0.5,0.5,0.01])

    with c1:
        image_1 = Image.open('intro1.jpg')
        st.image(image_1)
        
    with c2:
        image_2 = Image.open('intro2.jpg')
        st.image(image_2)



