import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer



# SBert 로딩
sentence_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")   



# 경로 불러오기 / 데이터 불러오기 (클러스터링)    
# 긍부정 데이터가 한 파일에 있다고 가정하고 진행 *********************************************

def get_data_to_cluster(product_name='', count=0):
  """
  product_name : 검색키워드(제품명)
  count : 분석 실행 횟수 -> 추후 함수끼리 연결하면서 변경 필요 
  """
  # 확인 후 수정 ****************************************
  if product_name == '신라면':
    path = '../data/점수파일/shin_naver_500_score.csv'
  elif product_name == '진라면':
    path = '../data/점수파일/jin_naver_score.csv'
  elif product_name == '푸르밀 검은콩우유':
    path = '../data/점수파일/purmil_naver_score.csv'
  else:
    pass

  total_review = pd.read_csv(path)

  return total_review



def cluster_review(total_review, sentiment='Negative'):
  # 수정 필요 *******************************************************************
  
  # 긍정 / 부정 데이터 선택
  if sentiment == 'positive': 
    positive_review = total_review[total_review['score'] > 3]
    review_lst = positive_review["review"].to_list()
  else:
    negative_review = total_review[total_review['score'] < 3]
    review_lst = negative_review["review"].to_list()
  
  # for _ in review_lst:
  #   print(_)
  #   print()

  # BERTopic 객체 생성 
  BERTopic_model = BERTopic(embedding_model=sentence_model,
                            nr_topics=10,   # Topic 개수 
                            low_memory=True,
                            calculate_probabilities=True)
  
  # 임베딩
  embeddings = sentence_model.encode(review_lst, show_progress_bar=True)

  # fit_transform
  topics, probs = BERTopic_model.fit_transform(review_lst, embeddings)
  BERTopic_model.save("./my_model")
  return BERTopic_model




# 시각화 (plotly)
def vizualize_review_cluster(BERTopic_model):
  return BERTopic_model.visualize_topics()


# 시각화 (Dataframe)
def get_topic_info_(BERTopic_model, topic_num = 0):

  # 키워드
  keyword = BERTopic_model.topic_labels_
  keyword = keyword[topic_num]
  keyword = keyword.split('_')[1:]

  b = ['']
  keyword = b + keyword

  keyword = '   # '.join (keyword).strip()

  # 대표 리뷰 
  rep_review = BERTopic_model.representative_docs_
  rep_review = rep_review[topic_num]

  review_info = [keyword]
  for _ in rep_review:
      # print(_)
      review_info.append(f'- {_}')

  insert_topic_num = f'Topic {topic_num}'
  topic_info_df = pd.DataFrame({insert_topic_num:review_info})

  return topic_info_df


