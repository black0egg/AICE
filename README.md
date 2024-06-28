# AIcode

## [0-1.기본 명령어]
>
> ## 주피터노트북 명령어
> Ctrl + Enter : 셀실행  
> Shift + Enter : 셀실행 후, 아래셀 선택  
> Alt + Enter : 셀실행 후, 아래 빈쉘 생성  
> A : 바깥에서 위쪽 빈쉘 생성  
> B : 바깥에서 아래 빈쉘 생성  
> dd : 바깥에서 해당쉘 삭제  
> spacebar 2번 : 줄 바꿈

## [0-2.도구 불러오기]
> ```python
> import pandas as pd  ## pandas 불러오고, pd로 정의
> import numpy as np  ## numpy 불러오고, np로 정의
> !pip install seaborn  # (!: 리눅스 프롬프트 명령어)
> import seaborn as sns   ## seaborn 설치 및 불러오고, sns로 정의
> import matplotlib.pylot as plt   ## matplot 불러오고, plt로 정의
> %matplotlib inline  # (%: 주피터랩 명령어)
> ```

## [1-1.빅데이터 수집]
>
> ## 시리즈 및 데이터프레임 만들기
> ```python
> a1 = pd.Series(['사과','오렌지','딸기'], index=[1,2,3], name=name)  # 시리즈 생성 (Series: 대소문자 구분 필요)
> a2 = pd.Dataframe({"a" : [1,2,3], "b" : [4,5,6], "c" : [7,8,9]}  # 딕셔너리 데이터프레임 생성, 열단위생성
> a3 = pd.Dataframe([[1,2,3],[4,5,6],[7,8,9]], ["a","b","c"])  # 리스트 데이터프레임 생성, index명=["a","b","c"], 행단위생성
> ```
>
> ## 디렉토리 확인
> ```python
> !pwd  # (Print Working Directory : 현재 디렉토리)
> %cd ..  # (Change Directory : 상위 디렉토리)
> %cd ~  # (Change Directory : 홈 디렉토리)
> cd /root  # (Change Directory /root : 루트 디렉토리)
> ls  # (List : 리스트 보기)
> ```
> 
> ## 디렉토리 확인
> ```python
> custom_framework.config.data_dir
> # 파일위치 환경변수
> # data 경로 : custom_framework.config.data_dir 
> # workspace 경로 : custom_framework.config.workspace_dir  
> # model 경로 : custom_framework.config.model_dir  
> # log 경로 : custom_framework.config.workspace_logs
> ```
> 
> ## “000.csv” 데이터 로드
> ```python
> df = pd.read_csv ('./000.csv', encoding = 'cp949')
> # (cp949는 MS office에서 인코딩할때 쓰임)
> ```
> 
> ## 커스텀 프레임워크에서 “000.csv” 데이터 로드
> ```python
> df = pd.read_csv (custom_framework.config.data_dir + './000.csv', encoding = 'cp949')
> # (custom_framework.config.data_dir 폴더에서 불러오기)
> ```
>
> ## 파일형식에 맞게 데이터 저장
> ```python
> df.to_csv ("000.csv", index = false)  # “000.csv” csv파일로 저장
> df.to_excel ("000.xlsx")   # “000.xlsx” 엑셀파일로 저장
> ```

## [1-2.빅데이터 분석]
>
> Column Names = 열  
> index = 행  
> value = 값(행들의 데이터)
>
> ## df데이터 / 처음 위(head)아래(tail) 10개행을 보여주기
> ```python
> df  # 데이터 구조 보기
> df.head()  # 앞에서 5줄 보여주기
> df.head(10)  # 앞에서 10줄 
> df.tail(10)  # 뒤에서 10줄
> df.info()  # df데이터 자료구조파악
> # Rangeindex(행수),datacolumns(열수),null데이터확인,dtype(설명)
> # int64(정수형),float64(실수형),bool(부울형),datetime64(날짜표현),category(카테고리),object(문자열or복합형)
> df.index  # df데이터 Rangindex 출력 (행,가로)
> df.columns  # df데이터 컬럼내역 출력 (열,세로)
> df.values  # df데이터 값 출력 (컬럼 별 값)
> df.shape  # df데이터 형태(column/row수) 확인
> df.dtypes  # df데이터 타입 확인
> ```
> ## 데이터 뽑아오기
> ```python
> df[0]  # x의 0번째 데이터 뽑아오기
> df[-1]  # x의 뒤에서 1번째 데이터 뽑아오기
> df[0:4]  # x의 0~4번째까지 데이터 뽑아오기
> df[:]  # x의 전체 데이터 뽑아오기
> df['a']  # 'a'컬럼 데이터 확인하기
> ```
> ## df데이터 / 통계정보
> ```python
> df.describe()  # count(컬럼별개수),mean(평균값),std(표준편차),min(최소값),25%,50%,75%(4분위수),max(최대값)
> df['abc']  # df데이터 / 'abc'컬럼의 데이터 확인
> df['abc'].value_counts()  # df데이터 / 'abc'컬럼의 값분포 확인
> df['abc'].value_counts(normalize=True)  # df데이터 / 'abc'칼럼 값분포비율(normalize=True) 확인
> ```
> ```python
> for c in df : print(c)  # df데이터의 모든컬럼 프린트
> [df[c].value_counts() for c in df]  # df데이터의 모든컬럼 값분포 확인
> ```
> ## df데이터 / 상관관계 분석
> ```python
> df.corr()
> ```
> ```python
> sns.set(rc={'figure.figsize':(20:20})
> sns.heatmap(df.corr(), annot=True)
> ```
> ## df데이터 / Z-score기준 신뢰수준 99%인 데이터 확인하기
> ```python
> df[(abs((df['abc']-df['abc'].mean())/df['abc'].std()))>2.58]  # 95%(1.96), 98%(2.33). 99%(2.58)
> ```
> ## 결측치 확인
> ```python
> df.isnull().sum()
> ```
> ## 'abc'컬럼에서 '_'값을 가지고 있는 값들 찾기
> ```python 
> df[df['abc'] == '_']
> ```

## [1-3.빅데이터 전처리]
> 최고빈번값(Most frequent),중앙값(Median),평균값(Mean),상수값(Constant)
> ## 데이터프레임 복사
> ```python
> df_origin = df.copy()  # df 데이터프레임을 df_origin으로 복사
> df = df_origin.copy()  # df_origin 데이터프레임을 df로 복사
> ```
> ## 데이터프레임 병합
> ```python
> df1_1 = df1.reset_index(drop=True)  # df1~2 인덱스 리셋
> df2_1 = df2.reset_index(drop=True)
> for col in df1_1.columns :
>   df1_1.rename(coloumns={col:'a_'+col}.inplac=True)  # df1_1의 각 컬럼에 a_이름추가해서 붙이기
> df3 = pd.concat([df1_1, df2_1],axis=1]  # concat함수로 병합
> df3.shape  # 확인
> ``` 
> ## 입력데이터에서 제외
> ※ axis=0(행), axis=1(열)
> ```python
> df.drop(['abc','def'], axis=1, inplace=True)  # abc, def 컬럼 삭제
> df.drop(index=0, axis=0, inplace=True)  # 첫번째 행 삭제
> df = df.drop('abc', axis=1)  # abc, def 컬럼(axis=1(열)) 삭제
> df1 = df.drop(columns=['abc','def'])  # abc, def 컬럼 삭제후 df1로 저장
> ```
> ## 조건 지정 데이터 추출
> ```python
> df1 = df['a']>100  # a컬럼이 100보다 큰경우의 데이터만, df1으로 저장
> df2 = df1['b']=='가'  # b컬럼의 데이터값이 '가'인경우 데이터만, df2로 저장
> ```
> ```python
> # 0의 비중이 95%넘는 컬럼찾기 (100은 인덱스수)
> for column in num_cols : 
>   if(((df[column]==0).sum()/100) >0.95:
>     print(column+':'+(str)((df[column]==0.sum()/100))))
> ```
> ```python
> # 1개범주 비중이 95%넘는 컬럼찾기 (100은 인덱스수)
> for column in obj_cols :
>   if((df[column].value_count().iloc[0]/100 > 0.95):
>     print(column+':'+(str)((df[column].value_count().iloc[0]/100))))
> ```
> ## 컬럼 생성/변경
> ```python
> df['가나'] = df['가'] + df['나']
> df.insert(loc, column, value)  # loc(삽입될열위치 ex.3 : 3번째), column(삽입될 열이름), value(삽입될 열값)
> df1 = df.rename(columns = {"a" : 'a_1'})
> ```
> ## 행열 전환
> ```python
> df=df.transpose()  # 행열 전환
> ```
> ## 결측치
>> ## df데이터 / 칼럼마다 결측치 여부 확인
>> ```python
>> df.isnull().sum(axis=0)
>> ```
>> ## 결측치 처리
>> missing(결측값수)  
>> '_'값을 numpy의 null값(결측치)으로 변경
>> ```python
>> df = df.replace('_', np.NaN)
>> df.replace('_', np.NaN, inplace=True)  # df데이터프레임 내에서, '_'값이 있는 값을 모두 np.nana(null)값으로 자체데이터(inplace=True)에 변경
>> df.info()
>> ```
>> ## 결측치(Null데이터) 처리
>> ```python
>> df['float_A'] = df['float_A'].fillna(0)  ## fillna() 메서드: 이 메서드는 결측치(null 값)를 다른 값으로 채우는 역할
>> df['abc'].fillna('A', inplace=True)  ## 데이터프레임 df, abc열 내의 null값을 'A'로 채움
>> ```
>> ## 'Class' 열의 결측치값 제외시키기
>> ```python
>> df.dropna(subset=['class'])
>> ```
>> ## 결측치 Listwise방식으로 행 제외시키기 (행 내에서 1개값이라도 NaN이면 제외)
>> ```python
>> df.dropna()
>> ```
>> ## 결측치 Pairwise방식으로 행 제외시키기 (행 내에서 모든값이 NaN일때 제외)
>> ```python
>> df.dropna(how="all")
>> ```
>> ## 결측치 Most frequent(최빈)값 대체하여 채우기
>> (범주형데이터 주로사용)  
>> df데이터 / 모두
>> ```python
>> df.fillna(df.mode().iloc[0])
>> ```
>> df데이터 / “a”칼럼 결측치를 해당칼럼 최빈값으로 채우기
>> ```python
>> df['a'].valuecounts()
>> df['a'].mode().iloc[0]  # 데이터프레임 df의 a열에서 최빈값'L'을 찾아 결측치 채우기
>> 'L'  # 결과값
>> df['a'].fillna('L', inplace=True)
>> ```
>> ```python
>> df['a'].fillna(df['a'].mode()[0], inplace=True)  # 데이터프레임 df의 열 a에서 최빈값을 찾아 결측치 채우기
>> ```
>> ## mean(평균), median(중간)값 대체하여 채우기
>> (범주형데이터 주로사용)
>> ```python
>> df.fillna(df.mean()["C1":"C2"])
>> ```
>> ## 결측치(Null데이터)를 숫자형태 데이터의 경우, 중간값으로 대체하여 채우기
>> ```python
>> df['abc'].median()  # 가.중간값 찾기
>> 77.0  # 결과값:77.0
>> df['abc'].replace(np.nan, 77.0, inplace=True)  # 나.abc컬럼 결측치 77.0으로 변경
>> df['abc'] = df['abc'].astype(int)  # 다.abc컬럼 타입 정수형숫자(int)로 변경
>> df['abc'].isnull().sum()  # 확인
>> ```
>> ## 앞값(ffill), 뒷값(backfill) 대체하여 채우기
>> ```python
>> df = df.fillna(method="ffill")
>> ```
>> ## 주변값과 상관관계로 선형 채우기
>> (선형관계형 데이터에서 주로사용)
>> ```python
>> df = df.interpolate()
>> ```
>
> ## 아웃라이어
>> ## 아웃라이어 제외
>> Class열의 H값 제외후 변경
>> ```python
>> df = df [(df["class"]! = "H")]
>> ```
>> 
>> ## 아웃라이어 변경
>> Class열의 H값을 F값으로 변경
>> ```python
>> df["class"] = df["class"].replace("H", "F")
>> ```
>> 제거기준 = (Q3 + IQR \* 1.5 보다 큰 값) & (Q1 - IQR \* 1.5 보다 작은 값)
>> 가.Q1, Q3, IQR 정의 IQR = Q3(3사분위수)-Q1(1사분위수)
>> ```python
>> Q1 = df[["Dividend","PBR"]].quantile(q=0.25)
>> Q3 = df[["Dividend","PBR"]].quantile(q=0.75)
>> IQR = Q3-Q1
>> ```
>> 나.변경
>> ```python
>> IQR_df = df[(df["Dividend"] <= Q3["Dividend"]+1.5*IQR["Dividend"]) & (df["Dividend"] >= Q1["Dividend"]-1.5*IQR["Dividend"])]
>> IQR_df = IQR_df[(IQR_df["PBR"] <= Q3["PBR"]+1.5*IQR["PBR"]) & (IQR_df["PBR"] >= Q1["PBR"]-1.5*IQR["PBR"])]
>> IQR_df = IQR_df[["Dividend","PBR"]]
>> ```
>> 다.확인(박스플롯)
>> ```python
>> IQR_df.boxplot()
>> IQR_df.hist(bins=20, figsize=(10,5))
>> ```
>> 
>> ## 신뢰도 99% 기준 이상치 추출후 제외하기
>> ```python
>> outlier = df[(abs((df['abc']-df['abc'].mean())/df['abc'].std()))>2.58].index
>> df = df.drop(outlier)
>> df.info()
>> ```
>> 
> ## Feature Engineering
>> ## 비닝(Binning)
>> 연속형 변수를 범주형 변수로 만드는 방법
>> 비닝 / cut : (구간값으로 나누기)
>> ```python
>> q1 = df["avg_bill"].quantile(0.25)
>> q3 = df["avg_bill"].quantile(0.75)
>> df["bill_rating"] = pd.cut(
>>                     df["avg_bill"],
>>                     bins = [0, q1, q3, df["avg_bill"].max()],
>>                     labels = ["low", "mid", "high"])
>> print (df["bill_rating"].value_counts()]
>> ```
>> 비닝 / qcut : (구간개수로 나누기)
>> ```python
>> df["bill_rating"] = pd.qcut(  # 비닝
>>                     df["avg_bill"],
>>                     3,
>>                     labels=["low", "mid", ;high"])
>> print (df["bill_rating"].value_counts()]
>> ```
>> ## 스케일링(Scaling)
>> 데이터 단위크기를 맞춤으로서 표준화/정규화
>> Standard Scaling :  
>> 평균을 0, 표준편차를 1로 맞추기 (데이터 이상치가 심할경우 사용)
>> ```python
>> df_num = df[["avg_bill", "A_bill", "B_bill"]]
>> Standardization_df = (df_num - df_num.mean()) / df_num.std()
>> ```
>> ```python
>> from sklearn.preprocessing import StandardScaler
>> scaler=StandardScaler()
>> 
>> X_train2 = scaler.fit_transform(X_train1)  # 정규분포화
>> X_train2 = scaler.transform(X_train1)  # 표준화
>> ```
>> Min-Max Scaling : 모든 데이터를 0~1사이로 맞추기
>> ```python
>> from sklearn.preprocessing import MinMaxScaler
>> scaler=MinMaxScaler()
>> nomalization_df = df_num.copy()
>> nomalization_df[:] = scaler.fit_transform(normalization_df[:])
>> ```
>> ## 라벨인코딩
>> n개의 종류를 가진 값들에 0 ~ (n-1)의 숫자를 부여하는 방법
>> 만족도조사, 성적 등 주로 순서형 자료에 적합하여 숫자들 사이에서 관계가 존재할 때 사용
>> ```python
>> from sklearn.preprocessing import LabelEncoder
>> le = LabelEncoder()
>> df['A'] = le.fit_transform(df['A'])
>> df
>> ```
>> ```python
>> le_columns = df.select_dtypes(include='object')  # int64(정수형),float64(실수형),bool(부울형),datetime64(날짜표현),category(카테고리),object(문자열or복합형)
>> le_columns.head()
>> le = LabelEncoder()
>> le_columns['a'] = le.fit_transform(le_columns['a'])
>> ```
>> ## 원핫인코딩
>>> ## 카테고리형 데이터를 원핫인코딩으로 컬럼 작성
>>> ```python
>>> cols = ["Gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
>>> dummies = pd.get_dummies(df[cols], drop_first=True)
>>> df = df.drop(cols, axis=1)
>>> df = pd.concat([df, dummies], axis=1)
>>> ```
>>> ### 카테고리형 데이터를 판다스로 쉽게 원핫인코딩
>>> ```python
>>> onehot column = ['a','b']
>>> df1 = pd.get_dummies(data=df1, columns=onehot column, drop_first=True)  # drop_first : 첫번째컬럼 제외하여 효율성 향상
>>> df1.info()
>>> ```
>> ## OrdinalEncoding
>> Categorical feature(범주형 특성)에 대한 순서형 코딩  
>> 각 범주들을 특성으로 변경하지 않고, 그 안에서 1,2,3 등의 숫자로 변경  
>> 범주가 너무 많아 one hot encoding을 하기 애매한 상황에서 이용
>> ```python
>> from category_encoders import OrdinalEncoder
>> enc1 = OrdinalEncoder(cols = "color")
>> df2 = enc1.fit_transform(df2)
>> df2 
>> ```
>> 
> ## 기타 주요작업
>> ## 토탈차지 공백을 0으로 변경후, 소수점 숫자형(float타입)으로 변경
>> ```python
>> df["TotalCharge"].replace([" "], ["0"], inplace=True)
>> df["TotalCharge"] = df["TotalCharge"].astype(float)
>> ```
>> ## 해지여부 Yes/No를 1,0의 정수형으로 변경
>> ```python
>> df["Churn"].replace(["Yes", "No"], [1, 0], inplace=True)
>> ```
>> ## 새로운 뉴피처 추가
>> ```python
>> df["new_feature"] = df["f_1"]/df["f_2"]
>> ```
>> ## distinct 피처 제외 (값종류수)  
>> distinct=1인 경우, 모든컬럼이 동일하므로 피처에서 제외  
>> ## 편향값  
>> 순서(인덱스)는 의미의 유무에 따라 제외  
>> ## 다중공선성 처리하기
>> 회귀분석에서 독립변수들 간에 강한 상관관계가 나타나는 것  
>> 다중공선성 확인하는 방법 : 상관계수(0.9이상), z-score(3.29이상), VIF계수(10이상) 컬럼 선택제외  
 
## [1-4.빅데이터 시각화]
>
> ## Matplotlib 시각화 (스캐터,바챠트)
> 영역 지정 : plt.figure()  
> 차트/값 지정 : plt.plot()  
> 시각화 출력 : plt.show()  
>> ### df데이터 / “00000”칼럼, 바차트 시각화 1 (이산)
>> ```python
>> df["00000"].value_counts( ).plot(kind="bar")
>> plt.show( )
>> ```
>> ### df데이터 / “00000”칼럼, 바차트 시각화 2 (Pairplot)
>> ```python
>> df.corr( )["00000"][:-1].sort_values( ).plot(kind="bar")
>> sns.pairplot(df)
>> ```
>> ### df데이터 / “A.B”칼럼, 히스토그램 시각화 (연속)
>> ```python
>> df["A.B"].plot(kind="hist")
>> plt.show( )
>> ```
>> ```python
>> df["00000"].plot(kind="hist")
>> ```
>> ### 바 플롯
>> ```python
>> vc = df['a'].value_counts()
>> plt.bar(x, height=vc, width=0.1)
>> ```
>> ### 히스토그램
>> ```python
>> plt.hist(df['a'])
>> ```
>> ```python
>> plt.figure(figsize=(10,4))
>> df['a'].plot(kind='hist')
>> ```
>> ### 산점도
>> ```python
>> plt.scatter(x, y)
>> ```
>> ### 색깔별 산점도 시각화
>> ```python
>> groups = df.groupby('variety')
>> groups
>> for name,group in groups :
>>  plt.scatter(x = 'A', y = 'B', data = group, label = name)
>> plt.legend()  ## 범례 추가
>> plt.show()
>> ```
>> ### 선 그래프
>> ```python
>> plt.plot(data)
>> ```
> ## Seaborn 시각화 (히트맵, 통계)
>> ### 카운트 플롯
>> ```python
>> sns.countplot(x="A", data=df)
>> ```
>> ### 박스 플롯
>> ```python
>> sns.boxplot(x="A", y="B", data=df)
>> ```
>> ### 조인트 플롯
>> ```python
>> sns.jointplot(x="A", y="B", data=df, kind="hex")
>> ```
>> ### 스캐터 플롯
>> ```python
>> fig, (ax1,ax2,ax3) = plt.subplot(1,3)
>> fig.set_size(14,4)
>> sns.scatterplot(data=df, x='a', y='y', ax=ax1)
>> sns.scatterplot(data=df, x='b', y='y', ax=ax2)
>> sns.scatterplot(data=df, x='c', y='y', ax=ax3)
>> ```
>> ### 상관관계 히트맵
>> ```python
>> plt.rcParams['figure.figsize'](20,20)
>> sns.heatmap(df.corr(),annot=True, cmap='RDYlBu_r', vmin=-1, vmax=1)  ## annotation 포함, corr함수로 상관계수 구하기
>> plt.rcParams['figure.figsize']=(5,5)
>> ```

## [1-5.세트 구성]
> 
> ## X,y데이터 설정하기
> ‘Answer’ 칼럼을 y값/타겟/레이블로 설정하기
> ```python
> X = df.drop('Answer',axis=1).values  # y컬럼만 제외하고 X값으로 저장
> X.shape  # 확인
> (10000, 37)  # 10000개의 행과 37개의 열
> y = df['Answer'].values  # Answer컬럼을 y값으로 저장
> y.shape  # 확인
> (10000,)  # 10000개 데이터
> ```
> ## X,y데이터 불러오기
> reshape(-1,1) 2차원배열 디자인포맷(reshape) 확장(-1은 알아서 넣으라는 뜻)
> ```python
> X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).resharpe(-1,1)
> y = np.array([13, 25, 34, 47, 59, 62, 79, 88, 90, 100])
> ```
> ## 트레이닝/테스트 세트 나누기
> Train : 모델을 학습하기 위한 데이터셋으로 이때 학습은 최적의 파라미터를 찾는 것, Train data는 오직 학습을 위한 데이터셋 
> Validation : 학습이 이미 완료된 모델을 검증하기 위한 데이터, 학습이 된 여러가지 모델 중 가장좋은 하나의 모델을 고르기 위한 데이터셋, 학습과정에 어느정도 관여하나, Validation 데이터자체가 학습에 직접적으로 관여하는 것은 아님 
> test : 모델의 '최종 성능'을 평가하기 위한 데이터셋, 학습과정에 관여를 하지 않음
> Train으로 학습하고, Validation으로 검증, Test로 최종성능을 평가 
> ```python
> from sklearn.model_selection import train_test_split 
> ```
> ## 테스트세트를 30%로 분류하고, 50번 랜덤하게 섞기
> (stratify : y값이 골고루 분할되도록 분할하는 메소드)
> ```python
> X_train, X_test, y_train, y_test =
> train_test_split
> (X, y, test_size=0.30, random_state=50, stratify=y)  # 70:30(test size=0.3), stratify 메소드: 분류모델에서 필요, 회귀모델에서는 불필요
> ```
> ## 데이터 정규화/스케일링 
> ```python
> from sklearn.preprocessing import MinMaxScaler  # 데이터를 0~1사이 숫자로 변경하여 머신러닝 알고리즘성능 향상
> from sklearn.preprocessing import StandardScaler
> help(MinMaxScaler)
> df1[['a']].head()  # df1 a칼럼 확인
> scaler = MinMaxScaler( )  # scaler 정의
> X_train = scaler.transform(X_train)
> X_test = scaler.transform(X_test)
> pd.DataFrame(X_train[:, 0], columnns=['a']).head()  # 확인
> ```

## [2.학습모델] ~ [3.최적화]
>
> ## 텐서플로 불러오고, tf로 정의
> ```python
> !pip install tensorflow
> import tensorflow as tf
> ```
> ## 각모델의 하이퍼파라미터 알아보기
> ```python
> model = RandomForestClassifier()
> params = model.get_params()
> print(params)
> ```
> ## 각모델의 학습클래스, 메소스사용법 출력
> ```python
> help(model.fit)  # model : 각모델 이름 또는 model로 정의
> ```
>
> ## 텐서플로 케라스모델 및 기능 불러오기
> (시퀀셜:히든레이어개수/덴스:노드개수/액티베이션/과적합방지기능 불러오기)
> ```python
> from tensorflow.keras.models import Sequential, load_model
> from tensorflow.keras.layers import Dense, Activation, Dropout
> from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
> from tensorflow.keras.utils import to_categorical
> ```
>
> ## [모델] LinearRegression 모델 (선형회귀)
> sklearn에서, 선형회귀모델(LinearRegression) 불러오기
> ```python
> from sklearn.family import model
> from sklearn.linear_model import LinearRegression
> ```
> ```python
> LinearR_model = LinearRegression()  # 가. 모델 선정
> LinearR_model.fit(X_train, y_train)  # 나. 테스트 핏
> score = LinearR_model.score(X_test, y_test)  # 다. 스코어
> LinearR_pred = model.predict(X_test)  # 라. 예측
> LinearR_model.summary( )  # 마. 확인
> ```
>
> 회귀예측 주요 성과지표
> ```python
> import numpy as np
> np.mean((y_pred - y_test) ** 2) ** 0.5
> ```
>
> ## [모델] Logistic Regression 모델 (분류회귀)
> sklearn에서, 분류회귀모델(Logistic Regression) 불러오기
> (설명: 분류모델 주로 활용)
> ```python
> from sklearn.linear_model import LogisticRegression
> from sklearn.model_selection import train_test_split
> from sklearn.metrics import classification_report 
> ```
> 가. 라이브러리 불러오기
> ```python
> from sklearn.linear_model import LogisticRegression
> from sklearn.model_selection import train_test_split
> from sklearn.metrics import confusion_matrix
> from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
> from sklearn.metrics import classification_report
> ```
> 나. 데이터 불러오기
> ```python
> train = pd.read_csv(custom_framework.config.data_dir + "/train.csv")
> ```
> 다. 세트 나누기
> ```python
> X_train, X_test, y_train, y_test, = train_test_split(
>                  train.drop("OOO", axis=1),
>                  train["OOO"], test_size=0.30, random_state=42)
> ```
> 라. 모델링
> ```python
> LogisticR_model = LogisticRegression(c=1.0,max_iter=5000)
> LogisticR_model.fit(x_train, y_train)
> LogisticR_model.score(X_test, y_test)
> LogisticR_pred = model.predict(x_test)
> confusion_matrix(y_test, LogisticR_pred) 
> print(classification_report(y_test, LogisticR_pred)
> ```
> ## [모델] 의사결정나무(Decision Tree) (선형회귀)
> 분류/회귀가능한 다재다능, 다소복잡한 데이터셋도 학습가능
> 가. 의사결정나무 라이브러리 불러오기
> ```python
> from sklearn.tree import DecisionTreeClassifier 
> ```
> 나. 데이터셋(df) 불러오기
> ```python
> DT_model = DecisionTreeClassifier(max_depth=10,random_state=42)
> DT_model.fit(X_train,y_train)
> DT_pred = DT_model.predict(X_test)
> accuracy_eval('DecisionTree',DT_pred,y_test)
> score = DT_model.score(X_test, y_test)
> ```
> 
> ## [모델] Random Forest
> 동일한 알고리즘으로 여러 분류기를 만든 후 예측 결과를 보팅으로 최종 결정하는 알고리즘
> 선형회귀모델 중 하나로, 의사결정나무(2개)에서 여러개를 더해 예측율을 높임
> sklearn에서, 랜덤포레스트 불러오기
> ```python
> from sklearn.tree import DecisionTreeClassifier
> ```
> 가. 랜덤포레스트 불러오기
> ```python
> from sklearn.ensemble import RandomForestRegressor
> ```
> 나. model 랜덤포레스트 선정
> ```python
> RF_model = RandomForestRegressor
>         (n_estimators=50,  # 학습시 생성할 트리갯수
>         max_depth=20,  # 트리의 최대 깊이
>         random_state=42, # 난수 seed 설정
>         ...,
>         criterion="gini",  # 분할 품질을 측정하는 기능(디폴트:gini)
>         min_samples_split=2,  # 내부노드를 분할하는데 필요한 최소샘플수
>         min_samples_leaf=1,  # 리프노드에 있어야할 최소샘플수
>         min_weight_fraction_leaf=0.0,  # 가중치 부여된 min_samples_leaf에서의 샘플수 비율
>         max_feature="auto")  # 각노드에서 분할에 사용할 특징의 최대수
> ```
> 다. 테스트 핏
> ```python
> RF_model.fit(x_train, y_train)
> ```
> 라. 스코어
> ```python
> RF_model.score(x_test, y_test)
> ```
> 마. 예측
> ```python
> RF_pred = model.predict(x_test)
> ```
> 바. RMSE값 구하기
> ```python
> np.mean((RF_pred - y_test) ** 2) ** 0.5  
> ```
> ## [모델] Ensemble 기법
> 1) Bagging : 동일한 알고리즘으로 여러 분류기를 만든 후 예측 결과를 보팅으로 최종 결정하는 알고리즘
> 2) Boosting : 이전학습 잘못예측한 데이터에 가중치부여해 오차보완  
> 3) Stacking : 여러개 모델이 예측한 결과데이터 기반, final\_estimator모델로 종합 예측수행
> 4) Weighted Blending : 각모델 예측값에 대해 weight 곱하여 최종 아웃풋계산
>
> ## [모델] GBM (Gradient Boost)
> 
> ## [모델] XGBoost (잘못예측한 것들을 좀더 집중등, 예측/분류성능을 높인 앙상블기법)
> (설명: GBM의 느림, 과적합 방지를 위해 Regulation만 추가, 리소스를 적게 먹으며 조기종료 제공) 
> ```python
> !pip install xgboost  # (!는 리눅스 명령어)
> 
> from xgboost import XGBClassfier
> xgb_model = XGBClassifier(n_estimators=50)  # n_estimators : 결정트리의 개수
> xgb_model.fit(X_train, y_train)
> xgb_pred = xgb_model.predict(X_test)
> accuracy_eval(xgb_model, xgb_pred, y_test)  # accuracy_eval : 정확도를 계산
> print(confusion_matrix(y_test, xgb_pred))
> xgb_model.score(X_test, y_test)
> ```
>
> ## [모델] LightGBM (결정트리기반 앙상블기법, 메모리사용량 적고 학습시간 짦음)
>
> ```python
> !pip install lightGBM
>
> from xgboost import LGBMClassfier
> lgbm_model = LGBMClassifier(n_estimators=3, random_state=42)
> lgbm_model.fit(X_train,y_train)
> lgbm_pred = lgbm_model.predict(X_test)
> accuracy_eval('lgbm',lgbm_pred,y_test)
> ```
>
> ## [모델] KNN (K-Nearest Neighbor)
>
> ```python
> from sklearn.neighbors import KNeighborsClassifier
> KNN_model = KNeighborsClassifier(n_neighbors=5)
> KNN_model.fit(X_train,y_train)
> knn_pred = KNN_model.predict(X_test)
> accuracy_eval('K-Nearest Neighbor',knn_pred,y_test)
> ```
> 
> ## [모델] 딥러닝 모델
>
> 가. 케라스 초기화 및 모델과 기능 불러오기
> ```python
> keras.backend.clear_session()
> ```
> ```python
> from tensorflow.keras.models import Sequential, load_model
> from tensorflow.keras.layers import Dense, Activation, Dropout
> from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
> from tensorflow.keras.utils import to_categorical
> ```
> 나. 모델 작성 30개의 features, 보통 연산효율을 위해 relu활용
> -   Batchnormalization 활용
> -   과적합 방지
> -   input layer(30features), 2 hidden layer, output layer(이진분류)
> ```python
> batch_size = 1024  # 하이퍼파라미터 설정
> epochs = 30
> X_train.shape
> (10000,30)
> y_tran.shape
> (10000,)
> ```
> ```python
> model = Sequential()
> model.add(Dense(64, activation="relu", input_shape=(30,)))  # 인풋데이터30(컬럼갯수), 히든레이더64개
> model.add(BatchNormalization( ))
> model.add(dropout(0.5))
> model.add(Dense(64, activation="relu"))  # 히든레이더64개
> model.add(BatchNormalization( ))
> model.add(dropout(0.5))
> model.add(Dense(32, activation="relu"))  # 히든레이더32개
> model.add(dropout(0.5))
> ```
> (※ 아웃풋1(이진분류) = sigmoid, 아웃풋3(다중분류), softmax)
> ```python
> model.add(Dense(1, activation="sigmoid"))  # 이진분류 : 덴스1,시그모이드
> # 또는 output layer ()
> ```
> (※ 아웃풋1(이진분류) = sigmoid, 아웃풋3(다중분류), softmax)
> ```python
> model.add(Dense(3, activation="softmax"))  # 다중분류 : 덴스2~,소프트맥스
> # 또는 output layer ()
> ```
> 
> 다.컴파일
>> ##### 이진분류 모델 (binary\_crossentropy)
>> ```python
>> model.compile(optimizer="adam",
>>               loss="binary_crossentropy",
>>               metrics=["accuracy"])
>> ```
>> ##### 다중분류 모델 (categorical\_crossentropy) (원핫인코딩 된 경우)
>> ```python
>> model.compile(optimizer="adam",
>>               loss="categorical_crossentropy",
>>               metrics=["accuracy"])
>> ```
>> ##### 다중분류 모델 (sparse\_categorical\_crossentropy) (원핫인코딩 안된 경우)
>> ```python
>> model.compile(optimizer="adam",
>>               loss="sparse_categorical_crossentropy",
>>               metrics=["accuracy"])
>> ```
>> ##### 예측 모델
>> ```python
>> model.compile(optimizer="adam",
>>               loss="mse")
>> ```
>
> 라. 딥러닝 테스트 핏
> ```python
> model.fit(x=X_train, y=y_train,
>           epochs=epochs, batch_size=batch_size,  # 하이퍼파라미터 설정
>           validation_data=(X_test, y_test),
>           verbose=1,
>           callbacks=[early_stop, check_point])  # 하이퍼파라미터 설정
> ```
>> ##### 조기종료 옵션 (케라스 조기종료&체크포인트 불러오기)
>> ```python
>> from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
>> ```
>> ##### (조기종료 : 로스값이 올라가면(5번까지는 괜찮음) 조기종료하기)
>> ```python
>> early_stop = EarlyStopping(monitor="val_loss", mode="min",
>>              verbose=1, patience=5)
>> ```
>> ##### (체크포인트 : 최적 로스값을 기억(best\_model.h5)하여 불러오기)
>> ```python
>> check_point = ModelCheckpoint("best_model.h5", verbose=1,
>>               monitor="val_loss", mode="min", save_best_only=True)
>> ```
>
> 마. 학습과정 로그(loss,accuracy) history에 선언하여 남기기
> ```python
> history = model.fit(x=X_train, y=y_train,
>           epochs=50, batch_size=20,
>           validation_data=(X_test, y_test),
>           verbose=1,
>           callbacks=[early_stop, check_point])
> ```
> ```python
> model.save('my_model1.h5')  # 모델 my_model1.h5로 저장
> !ls -l my_model1.h5
> ```
>
> 바. 학습로그 시각화 확인
> ```python
> import matplotlib.pyplot as plt  # Accuracy 그래프
> plt.figure(figsize=(10,5))
> plt.plot(history.history["accuracy"])
> plt.plot(history.history["val_accuracy"])
> plt.title("Model Accuracy")
> plt.xlabel("Epochs")
> plt.ylabel("Accuracy")
> plt.legend(["train_acc", "val_acc"], loc='lower right')
> plt.show( )
> ```
> ```python
> import matplotlib.pyplot as plt  # Loss 그래프
> plt.figure(figsize=(10,5))
> plt.plot(history.history['loss'], 'b', label='Train Loss')
> plt.plot(history.history['val_loss'], 'y', label='Validation Loss')
> plt.title("Model Loss")
> plt.xlabel("Epochs")
> plt.ylabel("Loss")
> plt.legend()
> plt.show()
> ```
> 
> 사. 딥러닝 성능평가
> ```python
> losses = pd.DataFrame(model.history.history)
> losses[["loss", "val_loss"]].plot( )
>
> from sklearn.metrics import classification_report, confusion_matrix
> predictions = model.predict_classes(X_test)
> print(classification_report(y_test, predictions))
> print(confustion_matrix(y_test,predictions))
> ```
>
> ## RNN
> RNN 모델링
> ```python
> import tensorflow as tf
> from tensorflow.keras.models import Sequential
> from tensorflow.keras.layers import Dense, Flatten
> from tensorflow.keras.layers import LSTM
> 
> X_train.shape, X_test.shape
> 
> X_train = X_train.reshape(-1,18,1)
> X_test = X_test.reshape(-1,18,1)
> 
> X_train.shape, X_test.shape
> 
> model = Sequential()
> model.add(LSTM(32,activation='relu',return_sequences=True,input_shape=(18,1)))
> model.add(LSTM(16,activation='relu',return_sequences=True))
> model.add(Flatten)
> model.add(Dense(8,activation='relu'))
> model.add(Dense(1,activation='sigmoid'))
> 
> model.summary()
> 
> model.compile(
> optimizer='adam',
> loss='binary_crossentropy',  ## 이진분류 : binary_crossentropy
> metrics=['accuracy'])
>
> history = model.fit(x=X_train,y=y_train,
>           epochs=10,
>           batch_size=128,
>           validation_data=(X_test,y_test),
>           verbose=1)
>
> losses = pd.DataFream(model.history.history)
> losses.head()
> losses[['loss','val_loss']].plot()
> 
> losses[['loss','val_loss','accuracy','val_accuracy']].plot()
> 
> plt.plot(history.history['accuracy'])
> plt.plot(history.history['val_accuracy'])
> plt.title('Accuracy')
> plt.xlabel('Epochs')
> plt.ylabel('Acc')
> plt.legend(['acc','val_acc'])
> plt.show()
> ```
> 
> ## CNN
> 가. 이미지 불러오기
> ```python
> import os
> from glob import glob
> import tensorflow as tf
> 
> FILENAME = 'dataset-new_old.zip'
> glob(FILENAME)
> 
> if not os.path.exists('IMAGE') :
> !mkdir IMAGE
> !cp dataset-new_old.zip ./IMAGE
> !cd IMAGE ; unzip dataset-new_old.zip
>
> new_img_path = './IMAGE/new/plastic1.jpg'
> gfile = tf.io.read_file(new_img_path)
> image = tf.io.decode_image(gfile,dtype=tf.float32)
> image.shape
> plt.imshow(image)
> plt.show()
> 
> old_img_path = './IMAGE/old/old_plastic1.jpg'
> gfile = tf.io.read_file(old_img_path)
> image.shape
> plt.imshow(image)
> plt.show()
> 
> Data Preprocess
> 
> num_epochs = 50
> batch_size = 4
> learning_rate = 0.001
> 
> input_shape = (384,512,3)  ## size
> num_classes = 2  ## new & old
> 
> from tensorflow.keras.preprocessing.image import ImageDataGenerator
> 
> training_datagen = ImageDataGenerator(
>                    rescale = 1./255,
>                    validation_split=0.2  # train set : 435*(1-0.2)=348)
> 
> test_datagen = ImageDataGenerator(
>                rescale = 1./255,
>                validation_split=0.2  # test set : 435*0.2 = 87)
> ```
> 나. 이미지 읽기, 배치, 셔플, 레이블링
> ```python
> !rm -rf ./IMAGE/.ipynb_checkpoints 
> 
> training_generator
> training_datagen.flow_from_directory(
>  ',/IMAGE/',
>  batch_size = batch_size,
>  target_size = (384,512),  # size
>  class_mode = 'catrgorical',  # binary, categorical
>  shuffle = True,
>  subset ='training'  # training, validation, validation_split 사용하므로 subset 지정
> )
> 
> test_generator
> test_datagen.flow_from_directory(
> ',/IMAGE/',
> batch_size = batch_size,
> target_size = (384,512),  # size
> class_mode = 'catrgorical',  # binary, categorical
> shuffle = True,
> subset ='validation'  # training, validation, validation_split 사용하므로 subset 지정
> )
> 
> print(training_generator.class_indices)
> 
> batch_samples = next(iter(traning_generator))
> 
> print('True Value : 'batch_sample[1][0])
> plt.imshow(batch_sample[0][0])
> plt.show()
> ```
> 다. CNN 모델링
> ```python
> import tensorflow as tf
> from tensorflow.keras.models import Sequential
> from tensorflow.keras.layers import Dense, Flatten, Dropout
> from tensorflow.keras.layers import Conv2D, MaxPooling2D
> 
> model = Sequential()  # Feature extraction
> model.add(Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=input_shape))
> model.add(MaxPooling2D(pool_size=2))
> model.add(Conv2D(filters=16,kernel_size=3,activation='relu'))
> model.add(MaxPooling2D(pool_size=2))
> 
> model.add(Flatten())  # Classfication
> model.add(Dense(50, activation='relu'))
> model.add(Dense(2, activation='softmax))
>
> model.summary()
>
> model.compile(
> optimizer='adam',
> loss='categorical_crossentropy',  # 이진분류
> metrics=['accuracy'])
>
> history = model.fit(training_generator,
> epochs=3,
> steps_per_epoch = len(training_generator) / batch_size,
> validation_steps = len(test_generator) / batch_size,
> validation_data = test_generator,
> vervose = 1
> )
> ```
> 라. 성능평가/시각화
> ```python
> losses = pd.Dataframe(model.history.history)
> losses = head()
> 
> losses[['loss','val_loss']].plot()
> 
> losses[['loss','val_loss','accuracy','val_accuracy']].plot()
> 
> plt.plot(history.history['accuracy'])
> plt.plot(history.history['val_accuracy'])
> plt.title('Accuracy')
> plt.xlabel('Epochs')
> plt.ylable('Acc')
> plt.legend(['acc','val_acc'])
> plt.show()
> ```
> 마. 예측하기
> ```python
>  # test_generator 샘플데이터 가져오기
>  # 배치사이즈 32 확인
> batch_img, batch_label = next(iter(test_generator))
> print(batch_img.shape)
> print(batch_label.shape)
> 
>  # 4개 test 샘플이지미 그려보고 예측해보기
> i = 1
> plt.figure(figsize=(16,30))
> for img, label in list(zip(batch_img, batch_label)):
> pred = model.predict(img,reshape(-1,384,512,3))
> pred_t = np.argmax(pred)
> plt.subplot(8,4,i)
> plt.title(f'True Value:{np.argmax(label)}, Pred Value:{pred_t})
> plt.imshow(img)
> i = i + 1
> ```
> 
> ## [모델] 비지도학습
> 가.주성분 분석
> ```python
> from sklearn.datasets import make_blobs  # 합성데이터 생성
> import matplotlib.pylot as plt
> x, y = make_blobs(n_features=10,  # (10차원)
>   n_samples-1000,
>   centers=5,  # (클러스트(모임) : 5개)
>   random_state-2023,
>   cluster_std=1)
> plt.scatter(x[:0], x[:1], c=y)
>
> from sklearn.preprocessing import StandardScaler  # 데이터세트 표준화하기
> scaler = StandardScaler()
> scaler.fit(x)
> std_data = scaler.transform(x)
> print(std_data)
>
> import pandas as pd  # 주성분분석 수행하기
> from sklearn.decompostion import PCA
> pca = PCA(n_components=10)
> reduced_data = pca.fit_transform(std_data)
> pca_df = pd.DataFrame(reduced_data)
> pca_df.head()
> print(pca.explained_variance_)  # 설명된 분산값 확인
> print(pca.explained_variance_ratio)  # 설명된 분산비율 확인
> # 이어서...
> ```
>
> ## [모델] SVM (Support Vector Machine)
> 
> ## [모델] Auto Encoder
> 
> ## 별도모델
> [모델] AdaBoost
> [모델] LSTM
> [모델] Transformer
> [모델] SES (Simple Exponential Smoothing)
> [모델] YOLO
> [모델] VGG
>
> ## Stacking
> 개별모델이 예측한 데이터를 기반한 종합예측
> ```python
> from sklearn.ensemle import StackingRegressor, StackingClassifier
> 
> stack_models =
> [('LogisticRegression',lg),('KNN',knn),('DecisionTree',dt)]
> 
> stacking = StackingClassifier(
>            stack_models, final_estimator=rfc,n_jobs=-1)
> 
> stacking.fit(X_train,y_train)
> 
> stacking_pred = stacking.predict(X_test)
> 
> accuracy_eval('Stacking Ensemble', stacking_pred, y_test)
> ```
>
> ## Weighted Blending
> 각 모델 예측값에 대하여 weight를 곱하여 최종계산
> ```python
> final_output = {
>  'DecisionTree':dt_pred,
>  'randomforest':rf_pred,
>  'xgb':xgb_pred,
>  'lgbm':lgbm_pred,
>  'stacking':stacking_pred
> }
>
> final_prediction=\
> final_outputs['DecisionTree']*0.1\
> +final_outputs['randomforest']*0.2\
> +final_outputs['xgb']*0.25\
> +final_outputs['lgbm']*0.15\
> +final_outputs['stacking']*0.3\
> 
> final_prediction = np.where(final_prediction>0.5,1,0) 
> # 가중치값이 0.5 초과하면 1, 그렇지 않으면 0
> 
> accuracy_eval('Weighted Blending', final_prediction, y_test)
> ```
> 
> ## GridSearchCV
> GridSearchCV는 모델과 하이퍼파라미터 값범위를 지정하면 교차검증을 사용하여 하이퍼파라미터값의 가능한 모든조합을 수행하여 최적값 도출
> ```python
> from sklearn.model_selection import GridSearchCV, train_test_split  # 불러오기
> dt_clf = DecisionTreeClassifier(random_state=1)  # 튜닝전 디시전트리 모델 예측정확도 확인
> dt_clf.fit(X_train, y_train)
> pred = dt_clf.predict(X_test)
> accuracy = accuracy_score(y_test, pred)
> print('예측 정확도 : {0:.4f}'.format(accuracy))  # 결과값 확인
> print('\nDecisionTreeClassifier 하이퍼 파라미터:\n', dt_clf.get_params())  # 디시전트리 모델 하이퍼 파라미터 확인
> DecisionTreeClassifier 하이퍼 파라미터:
> {'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'random_state': 1, 'splitter': 'best'
> ```
>> criterion : 분할 성능 측정 기능
>> min_samples_split : 노드를 분할하기 위한 최소한의 샘플 데이터수로, 과적합을 제어하는데 주로 사용함. 작게 설정할 수록 분할 노드가 많아져 과적합 가능성이 높아짐
>> max_depth : 트리의 최대 깊이, 깊이가 깊어지면 과적합될 수 있음
>> max_features : 최적의 분할을 위해 고려할 최대 feature 개수 (default = None : 데이터 세트의 모든 피처를 사용)
>> min_samples_leaf : 리프노드가 되기 위해 필요한 최소한의 샘플 데이터수 (과적합 제어 용도), 작게 설정 필요
>> max_leaf_nodes : 리프노드의 최대 개수
> ```python
> param_grid = {  # 하이퍼파라미터 실험영역 설정
>    'criterion':['gini','entropy'], 
>    'max_depth':[None,2,3,4,5,6], 
>    'max_leaf_nodes':[None,2,3,4,5,6,7], 
>    'min_samples_split':[2,3,4,5,6], 
>    'min_samples_leaf':[1,2,3], 
>    'max_features':[None,'sqrt','log2',3,4,5]
>    }
> ```
>> GridSearchCV 의 인자들
>> estimator : 보통 알고리즘을 객체로 만들어 넣어준다.
>> param_grid : 튜닝을 위한 대상 파라미터, 사용될 파라미터를 딕셔너리 형태로 넣어준다.
>> scoring : 예측 성능을 측정할 평가 방법을 넣는다. 분류 알고리즘일 때는, 'accuracy', 'f1', 회귀 알고리즘일 때는 'neg_mean_squared_error', 'r2' 등을 넣을 수 있다.
>> cv : 교차 검증에서 몇개로 분할되는지 지정한다.(정수로 넣어주면 K겹 교차검증이 되고, KFold(k) 이런식으로 넣어주어도 무방 // default 값은 cv=3)
>> refit : True로 하면 최적의 하이퍼 파라미터를 찾아서 estimator를 재학습시킨다. (default 값이 True임)
> ```python
> grid_search = GridSearchCV(dt_clf, param_grid = param_grid, cv = 5, scoring = 'accuracy', refit=True)
> grid_search.fit(X_train, y_train)
> print('best parameters : ', grid_search.best_params_)
> print('best score : ', round(grid_search.best_score_, 4))
> df = pd.DataFrame(grid_search.cv_results_)
> df
> estimator = grid_search.best_estimator_
> pred = estimator.predict(X_test)
> print('score: ', round(accuracy_score(y_test,pred), 4))
> ```
>
> ## RandomSearchCV 
> 
## [4.성능평가]
>
> ## 손실함수(신경망 학습의 목적으로 출력값,정답 차이계산)
>> 회귀모델 손실함수(Loss Function)
>> -   MSE(Mean Squared Error) : 실제에서 예측값 차이를 제곱, 합하여 평균 (예측) (작을수록 좋음)
>> -   MAE(Mean Absolute Error) : 실제값 빼기 예측값 절댓값의 평균 (작을수록 좋음)
>> 
>> 분류모델 손실함수
>> -   CEE(Cross Entropy Error) : 예측결과가 빗나갈수록 더큰패널티 부여 (분류)
>> -   Binary Cross Entropy (이진분류)
>> -   Categorical Cross Entropy (다중분류)
>> -   Sparse Categorical Cross Entropy
>> -   Multi Class Classfication (다중분류)
>>
>> 주요 지표
>> -   loss = MSE (학습시 사용한 loss function종류에 의해 결정) (작을수록 좋음)
>> -   error = 실제값 빼기 예측값의 평균 (작을수록 좋음)
>> -   R2(결정계수) = 독립변수가 종속변수를 얼마나 잘설명하는지 (클수록 좋음)
>>
> ## 옵티마이저 (딥러닝 모델의 매개변수(w,b)를 조절하여 손실함수값을 최저로 만드는 과정
>> 경사하강법(Gradient Descent) : 손실함수의 현가중치에서 기울기를 구해서 loss를 줄이는 방향으로 업데이트해나가는 방법
>> 순전파 : 딥러닝 모델에 값을 입력해서 출력을 얻는 과정
>> 오차역전파 : 실제값과 결과겂 오차를 구한 후, 해당오차를 다시 앞으로 보내 가중치를 재업데이트하는 과정
>> ref) GD(Gradient Descent), SGD, RMSProp, Adam...
>>
> ## RMSE값 확인하기
>
> ## 목표
>> Loss(오차율) 낮추고, Accuracy(정확도) 높이기  
>> Error -> Epochs이 많아질수록 줄어들어야 함  
>> Epoch 많아질수록, 오히려 TestSet Error 올라가는경우 생길때, 직전Stop  
>> 학습시 조기종료(early stop) 적용되지 않았을 때는 개선여지가 있기에,  
>> 배치사이즈나 에포크를 수정하여 개선할 수 있음
> 
> ## 좋은 모델
>> 과적합(overfitting) : 선이 너무 복잡  
>> 트레인 어큐러시만 높아지고, 벨리드 어큐러시는 높아지지 않을때 (트레인어큐러시에 맞춰짐)  
>> 과소적합(underfitting) : 선이 너무 단순  
>> 트레인/벨리드 어큐러시가 교차되지 않고 아직 수평선을 향해갈때  
>> 좋은모델 : 어느정도 따라가는 적당선  
>> 트레인/벨리드 어큐러시가 수평선을 이어 서로 교차될때
>
> ## 분류모델 성능지표
>
>> 1.오차행렬(Confusion Matrix) (분류모델에 주로 쓰임)
>> - TP (True Positive) : 포지티브 중 맞춤 
>> - TN (True Negative) : 네거티브 중 맞춤
>> - FP (False Positive) : 포지티브 중 틀림
>> - FN (False Negative) : 네거티브 중 틀림
>
>> 오차행렬 지표
>> - 정확도(Accuracy) = TP + TN (맞춤) / 전체(total)
>> - 정밀도(Precision) = TP / TP + FP (예측한 클래스 중, 실제로 해당 클래스인 데이터 비율)
>> - 재현율(Recall) = TP = TP + FN (실제 클래스 중, 예측한 클래스와 일치한 데이터 비율) 
>> - F1점수(F1-score) = 2 \* \[1/{(1/Precision)+(1/Recall)}\] (Precision과 Recall의 조화평균)
>> - Support = 각 클래스 실제 데이터수
>
>> 2.Classification_report
>    
> 오차행렬 성능지표 쉽게확인 
> ```python
> from sklearn.metrics import classification_report
> print(classification_report(y_test, y_pred))
> ```
> 
> 오차행렬 성능지표 확인
> ```python
> import seaborn as sns  # 불러오기
> from sklearn.metrics import confusion_matrix
> from sklearn.metrics import ConfusionMatrixDisplay
> from sklearn.metrics import precision_score, recall_score
> 
> y_pred = model.predict(X_test)
> cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
> disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
> disp.plot()
> plt.show()
> print(classification_report(y_test, y_pred))
> 
> sns.heatmap(cm, annot=True)
> plt.show()
> 
> print(classification_report(y_test, y_pred))
>
> confusion_matrix(y_true, y_pred)
> accuracy_score(y_true, y_pred)
> precision_score(y_true, y_pred)
> recall_score(y_true, y_pred)
> f1_score(y_true, y_pred)
> ```
>

## [5.적용]

