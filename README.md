___

## [0-1.기본 명령어]

> ## Jupyter Notebook 명령어
> Shift + Enter : 셀실행 후, 아래셀 선택  
> Alt + Enter : 셀실행 후, 아래 빈쉘 생성  
> Ctrl + Enter : 셀실행  
> A : 바깥에서 위쪽 빈쉘 생성  
> B : 바깥에서 아래 빈쉘 생성  
> dd : 바깥에서 해당쉘 삭제

___

## [0-2.도구 불러오기]

> ## pandas 불러오고, pd로 정의
> ```python
> import pandas as pd  # 판다스 불러오기
> ```
> ## numpy 불러오고, np로 정의
> ```python
> import numpy as np
> ```
> ## seaborn 설치 및 불러오고, sns로 정의
>(!: 리눅스 프롬프트 명령어)
> ```python
> !pip install seaborn
> import seaborn as sns
> ```
> ## matplot 불러오고, plt로 정의
>(%: 주피터랩 명령어)
> ```python
> %matplotlib inline
> import matplotlib.pylot as plt
> ```
> ## 텐서플로 불러오고, tf로 정의
> ```python
> import tensorflow as tf
> ```
> ## 텐서플로 케라스모델 및 기능 불러오기
> (시퀀스(히든레이어개수)/덴스(노드개수)/액티베이션/과적합방지기능 불러오기)
> ```python
> from tensorflow.keras.models import Sequential
> from tensorflow.keras.layers import Dense, Activation, Dropout
> ```
> ## [모델] sklearn에서, 선형회귀모델(LinearRegression) 불러오기
> ```python
> from sklearn.family import model
> from sklearn.linear_model import LinearRegression
> ```
> ## [모델] sklearn에서, 분류회귀모델(Logistic Regression) 불러오기
> (설명: 분류모델 주로 활용)
> ```python
> from sklearn.linear_model import LogisticRegression
> from sklearn.model_selection import train_test_split
> from sklearn.metrics import classification_report 
> ```
> ## [모델] sklearn에서, 랜덤포레스트 불러오기
> (설명: 의사결정나무 2개에서, 여러개를 더해 예측율을 높임)
> ```python
> from sklearn.tree import DecisionTreeClassifier
> ```
> ## [모델] sklearn에서, 의사결정나무 불러오기
> (설명: 분류/회귀가능한 다재다능, 다소복잡한 데이터셋도 학습가능)
> ```python
> from sklearn.tree import DecisionTreeClassifier
> ```
> ## [모델] AdaBoost
> ## [모델] GBM (Gradient Boost)
> ## [모델] XGBoost
> (설명: GBM의 느림, 과적합 방지를 위해 Regulation만 추가, 리소스를 적게 먹으며 조기종료 제공)
> ## [모델] SVM (Support Vector Machine)
> ## [모델] Auto Encoder
> ## [모델] CNN
> ## [모델] RNN
> ## [모델] LSTM
> ## [모델] Transformer
> ## [모델] SES (Simple Exponential Smoothing)
> ## [모델] YOLO
> ## [모델] VGG

___

## [1-1.빅데이터 수집]
> ## “00000.csv” 데이터 로드
> (cp949는 MS office에서 인코딩할때 쓰임)
> ```python
> df = pd.read_csv ("./00000.csv", encoding = "cp949")
> ```
> ## 커스텀 프레임웍에서 “00000.csv” 데이터 로드 2
> (custom\_framework.config.data\_dir 폴더에서 불러옴)
> ```python
> df = pd.read_csv (custom_framework.config.data_dir + "/00000.csv")
> ```
> ## 파일위치 환경변수
> data 경로: custom\_framework.config.data\_dir  
> workspace 경로: custom\_framework.config.workspace\_dir  
> model 경로: custom\_framework.config.model\_dir  
> log 경로: custom\_framework.config.workspace\_logs
> ## “00000\_final.csv” 데이터 저장 1
> ```python
> df.to_csv ("00000_final.csv", index = false)
> ```
> ## “00000\_final.xlsx” 엑셀로 저장 2
> ```python
> df.to_excel ("00000.xlsx")
> ```

___

## [1-2.빅데이터 분석]
Column Names = 열  
index = 행  
value = 값(행들의 데이터)
> ## df데이터 / 처음 위(head)아래(tail) 10개행을 보여주기
> ```python
> df.head( )
> df.head(10)
> df.tail(10)
> ```
> ## df데이터 / 형태(column/row수) 확인
> ```python
> df.shape
> ```
> ## df데이터 / 컬럼내역 출력 (열,세로)
> ```python
> df.columns
> ```
> ## df데이터 / 로우내역 출력 (행,가로)
> ```python
> df.values
> ```
> ## df데이터 / 자료구조 파악
> ```python
> df.info( )
> ```
> ## df데이터 / 타입 확인
> ```python
> df.dtypes 
> ```
> ## df데이터 / 통계정보
> mean(평균), std(분산), min/max(최소/최대값)  
> ※ df.describe( ).transpose( )
> ```python
> df.describe( )
> ```
> ## df데이터 / 상관관계 분석
> ```python
> df.corr( )
> ```
> ## 데이터 뽑아오기
> ```python
> x[0]  ## x의 0번째 데이터 뽑아오기
> x[-1]  ## x의 뒤에서 1번째 데이터 뽑아오기
> x[0:4]  ## x의 0~4번째까지 데이터 뽑아오기
> x[:]  ## x의 전체 데이터 뽑아오기
> ```
> ## df데이터 / “00000”컬럼의 데이터 확인
> ```python
> df["00000"]
> ```
> ## df데이터 / “00000”컬럼의 값분포 확인
> ```python
> df["00000"].value_counts()
> ```
> ## df데이터 / “00000”칼럼의 값비율 확인
> ```python
> df["00000"].value_counts(normalize=True)
> ```

___

## [1-3.빅데이터 시각화]

> ## [Matplotlib] 시각화 (스캐터,바챠트)
> 영역 지정 : plt.figure()  
> 차트/값 지정 : plt.plot()  
> 시각화 출력 : plt.show()  
>> ### df데이터 / “00000”칼럼, 바차트 시각화 1 (이산)
>> ```python
>> df["00000"].value_counts( ).plot(kind="bar")
>> plt.show( )
>> ```
>> ### df데이터 / “00000”칼럼, 바차트 시각화 2
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
>> plt.bar(x, height)
>> ```
>> ### 히스토그램
>> ```python
>> plt.hist(x)
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
> ## [Seaborn] 시각화 (히트맵, 통계)
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
>> ### 상관관계 히트맵
>> ```python
>> sns.heatmap(df.corr( ), annot=True)
>> ```
>> ```python
>> corr = jiro_df.corr()  ## corr함수로 상관계수 구하기
>> sns.heatmap(corr,annot=True)  ## annotation 포함
>> ```

___

## [1-4.빅데이터 전처리]
최고빈번값(Most frequent), 중앙값(Median), 평균값(Mean), 상수값(Constant)

> ## 입력데이터에서 제외
> ※ axis=0(행), axis=1(열)
> ```python
> drop( )
> ```
> ```python
> df = df.drop('A', axis=1)
> ```
> ## 누락데이터 처리
> ※ axis=0(행), axis=1(열)
> ```python
> replace( )
> ```
> ## 결측치
>> ## df데이터 / 칼럼마다 결측치 여부 확인
>> ```python
>> df.isnull().sum()
>> ```
>> ## 결측치(Null데이터) 처리
>> ```python
>> dropna( ), fillna( ) 
>> ```
>> ```python
>> df['float_A'] = df['float_A'].fillna(0)  ##float_A열 결측치 0으로 채우기
>> ```
>> ## 결측치 처리
>> missing(결측값수)  
>> “_“를 numpy의 null값(결측치)으로 변경
>> ```python
>> df = df.replace("_", np.NaN)
>> ```
>> ## “Class” 열의 결측치값 제외시키기
>> ```python
>> df.dropna(subset=["class"])
>> ```
>> ## Listwise 결측치 행 제외시키기
>> (행의 1개값이라도 NaN이면 제외)
>> ```python
>> df.dropna()
>> ```
>> ## Pairwise 결측치 행 제외시키기
>> (행의 모든값이 NaN일때 제외)
>> ```python
>> df.dropna(how="all")
>> ```
>> ## Most frequent(최빈)값 대체하여 채우기
>> (범주형데이터 주로사용)  
>> df데이터 / 모두
>> ```python
>> df.fillna(df.mode().iloc[0])
>> ```
>> df데이터 / “A”칼럼 결측치를 해당칼럼 최빈값으로 채우기
>> ```python
>> df["A"].fillna(df["A"].mode()[0])
>> ```
>> ## mean(평균), median(중간)값 대체하여 채우기
>> (범주형데이터 주로사용)
>> ```python
>> df.fillna(df.mean()["C1":"C2"])
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
> ## 아웃라이어
>> ## 아웃라이어 제외
>> Class열의 H값 제외후 변경
>> ```python
>> df = df [(df["class"]! = "H")]
>> ```
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
>> lb = LabelEncoder()
>> df['A'] = lb.fit_transform(df['A']
>> df
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
>>> data = df[["AA","BB"]]
>>> one_hot_df = pd.get_dummies(data, columns=["class"])
>>> one_hot_df
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

___

## [1-5.세트 구성]

> ## 트레이닝/테스트 세트 나누기
> ```python
> from sklearn.model_selection import train_test_split 
> ```

> ## X,y데이터 설정하기
> ‘Answer’ 칼럼이 y값/타겟/레이블
> ```python
> X = df.drop('Answer',axis=1).values
> y = df['Answer'].values
> ```

> ## X,y데이터 불러오기
> reshape(-1,1) 2차원배열 디자인포맷(reshape) 확장(-1은 알아서 넣으라는 뜻)
> ```python
> X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).resharpe(-1,1)
> y = np.array([13, 25, 34, 47, 59, 62, 79, 88, 90, 100])
> ```

> ## 테스트세트를 30%로 분류하고, 50번 랜덤하게 섞기
> (y값이 골고루 분할되도록 stratify하게 분할)
> ```python
> X_train, X_test, y_test =
> train_test_split
> (X, y, test_size=0.30, random_state=50, stratify = y)
> ```
> (데이터 정규화/스케일링)
> ```python
> from sklearn.preprocessing import MinMaxScaler
> help(MinMaxScaler)
> scaler = MinMaxScaler( )
> scaler.fit(X_train)
> X_train = scaler.transform(X_train)
> X_test = scaler.transform(X_test)
> ```

___

## [2.학습모델] ~ [3.최적화]
>
> ## LinearRegression 모델 (선형회귀)
> 가. 모델 선정
> ```python
> model = LinearRegression( )
> ```
> 나. 테스트 핏
> ```python
> model.fit(X_train, y_train)
> ```
> 다. 예측
> ```python
> linearR_pred = model.predict(X_test)
> ```
> 라. 확인
> ```python
> model.summary( )
> ```
> 회귀예측 주요 성과지표
> ```python
> import numpy as np
> np.mean((y_pred - y_test) ** 2) ** 0.5
> ```
>
> ## Logistic Regression 모델 (분류회귀)
> 가. 라이브러리 불러오기
> ```python
> from sklearn.linear_model import LogisticRegression
> from sklearn.model_selection import train_test_split
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
> model = LogisticRegression( )
> model.fit(x_train, y_train)
> LogisticR_pred = model.predict(x_test)
> print(classification_report(y_test, LogisticR_pred)
> ```
>
> ## 의사결정나무(Decision Tree) (선형회귀)
> 분류/회귀가능한 다재다능, 다소복잡한 데이터셋도 학습가능
> 가. 의사결정나무 라이브러리 불러오기
> ```python
> from sklearn.tree import DecisionTreeClassifier 
> ```
> 나. 데이터셋(df) 불러오기
> ```python
> model = DecisionTreeClassifier(max_depth=10,random_state=42)
> model.fit(X_train,y_train)
> dt_pred = model.predict(X_test)
> accuracy_eval('DecisionTree',dt_pred,y_test) 
> ```
>
> ## Ensemble 기법
> 1) Bagging  
> 2) Boosting : 이전학습 잘못예측한 데이터에 가중치부여해 오차보완  
> 3) Stacking : 여러개 모델이 예측한 결과데이터 기반, final\_estimator모델로 종합 예측수행
> 4) Weighted Blending : 각모델 예측값에 대해 weight 곱하여 최종 아웃풋계산
>
> ## XGBoost
> (!는 리눅스 명령어)
> ```python
> !pip install xgboost
> 
> from xgboost import XGBClassfier
> model = XGBClassifier(n_estimators=50)
> model.fit(X_train,y_train)
> xgb_pred = model.predict(X_test)
> accuracy_eval('XGBoost',xgb_pred, y_test)
> ```
>
> ## LightGBM
>
> ```python
> !pip install lightGBM
>
> from xgboost import GBMClassfier
> model = LGBMClassifier(n_estimators=3, random_state=42)
> model.fit(X_train,y_train)
> lgbm_pred = model.predict(X_test)
> accuracy_eval('lgbm',lgbm_pred,y_test)
> ```
>
> ## KNN (K-Nearest Neighbor)
>
> ```python
> from sklearn.neighbors import KNeighborsClassifier
> model = KneighborsClassifier(n_neighbors=5)
> model.fit(X_train,y_train)
> knn_pred = model.predict(X_test)
> accuracy_eval('K-Nearest Neighbor',knn_pred,y_test)
> ```
> 
> ## Random Forest
> 선형회귀모델 중 하나  
> 의사결정나무 2개에서, 여러개를 더해 예측율을 높임
>
> 가. 랜덤포레스트 불러오기
> ```python
> from sklearn.ensemble import RandomForestRegressor
> ```
> 나. model 랜덤포레스트 선정
> ```python
> model = RandomForestRegressor
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
> model.fit(x_train, y_train)
> ```
> 라. 스코어
> ```python
> model.score(x_test, y_test)
> ```
> 마. 예측
> ```python
> rf_pred = model.predict(x_test)
> ```
> 바. RMSE값 구하기
> ```python
> np.mean((y_pred - y_test) ** 2) ** 0.5  
> ```
>
> ## 딥러닝 모델
> 가. 케라스 초기화
> ```python
> keras.backend.clear_session( )
> ```
> 나. 모델 작성 30개의 features, 보통 연산효율을 위해 relu활용  
>    Batchnormalization 활용  
>    과적합 방지  
>    input layer(30features), 2 hidden layer, output layer(이진분류)
> ```python
> model = Sequential( )
> model.add(Dense(64, activation="relu", input_shape=(30,)))
> model.add(BatchNormalization( ))
> model.add(dropout(0.5))
> model.add(Dense(64, activation="relu"))
> model.add(BatchNormalization( ))
> model.add(dropout(0.5))
> model.add(Dense(32, activation="relu"))
> model.add(dropout(0.5))
> model.add(Dense(1, activation="sigmoid"))
> # 또는 output layer ()
> ```
> (※ 아웃풋1(이진분류) = sigmoid, 아웃풋3(다중분류), softmax)
> ```python
> model.add(Dense(3, activation="softmax"))
> ```
> 다.컴파일
> 이진분류 모델 (binary\_crossentropy)
> ```python
> model.compile(optimizer="adam",
>               loss="binary_crossentropy",
>               metrics=["accuracy"])
> ```
> 다중분류 모델 (categorical\_crossentropy) (원핫인코딩 된 경우)
> ```python
> model.compile(optimizer="adam",
>               loss="categorical_crossentropy",
>               metrics=["accuracy"])
> ```
> 다중분류 모델 (sparse\_categorical\_crossentropy) (원핫인코딩 안된 경우)
> ```python
> model.compile(optimizer="adam",
>               loss="sparse_categorical_crossentropy",
>               metrics=["accuracy"])
> ```
> 예측 모델
> ```python
> model.compile(optimizer="adam",
>               loss="mse")
> ```
> 마. 딥러닝 테스트 핏
> ```python
> model.fit(x=X_train, y=y_train,
>           epochs=50, batch_size=20,
>           validation_data=(X_test, y_test),
>           verbose=1,
>           callbacks=[early_stop, check_point])
> ```
>> 조기종료 옵션 (케라스 조기종료&체크포인트 불러오기)
>> ```python
>> from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
>> ```
>> (조기종료 : 로스값이 올라가면(5번까지는 괜찮음) 조기종료하기)
>> ```python
>> early_stop = EarlyStopping(monitor="val_loss", mode="min",
>>              verbose=1, patience=5)
>> ```
>> (체크포인트 : 최적 로스값을 기억(best\_model.h5)하여 불러오기)
>> ```python
>> check_point = ModelCheckpoint("best_model.h5", verbose=1,
>>               monitor="val_loss", mode="min", save_best_only=True)
>> ```
> 바. 학습과정 로그(loss,accuracy) history에 선언하여 남기기
> ```python
> history = model.fit(x=X_train, y=y_train,
>           epochs=50, batch_size=20,
>           validation_data=(X_test, y_test),
>           verbose=1,
>           callbacks=[early_stop, check_point])
> ```
> 사. 학습로그 시각화 확인
> ```python
> import matplotlib.pyplot as plt
> plt.plot(history.history["accuracy"])
> plt.plot(history.history["val_accuracy"])
> plt.title("Accuracy")
> plt.xlabel("Epochs")
> plt.ylabel("Accuracy")
> plt.legend(["train_acc", "val_acc"])
> plt.show( )
> ```
> 아. 딥러닝 성능평가
> ```python
> losses = pd.DataFrame(model.history.history)
> losses[["loss", "val_loss"]].plot( )
>
> frome sklearn.metrics import classification_report, confusion_matrix
> predictions = model.predict_classes(X_test)
> print(classification_report(y_test, predictions))
> print(confustion_matrix(y_test,predictions))
> ```

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

## [4.성능평가]
> ## 목표
> Loss(오차율) 낮추고, Accuracy(정확도) 높이기  
> Error -> Epochs이 많아질수록 줄어들어야 함  
> Epoch 많아질수록, 오히려 TestSet Error 올라가는경우 생길때, 직전Stop  
> 학습시 조기종료(early stop) 적용되지 않았을 때는 개선여지가 있기에,  
> 배치사이즈나 에포크를 수정하여 개선할 수 있음
> 
> ## 좋은 모델
> 과적합(overfitting) : 선이 너무 복잡  
> 트레인 어큐러시만 높아지고, 벨리드 어큐러시는 높아지지 않을때 (트레인어큐러시에 맞춰짐)  
> 과소적합(underfitting) : 선이 너무 단순  
> 트레인/벨리드 어큐러시가 교차되지 않고 아직 수평선을 향해갈때  
> 좋은모델 : 어느정도 따라가는 적당선  
> 트레인/벨리드 어큐러시가 수평선을 이어 서로 교차될때
>
> ## 성능지표
> 오차행렬(Confusion Matrix) (분류모델에 주로 쓰임)
> -   TP (True Positive)
> -   TN (True Negative)
> -   FP (False Positive)
> -   FN (False Negative)
> 오차행렬 지표
> -   정확도(Accuracy) = 맞춤(TP&TN) / 전체(total)
> -   정밀도(Precision) = TP / TP + FP (예측한 클래스 중, 실제로 해당 클래스인 데이터 비율)
> -   재현율(Recall) = TP = TP + FN (실제 클래스 중, 예측한 클래스와 일치한 데이터 비율)
> -   F1점수(F1-score) = 2 \* \[1/{(1/Precision)+(1/Recall)}\] (Precision과 Recall의 조화평균)
> -   Support = 각 클래스 실제 데이터수
> 오차행렬 성능지표 쉽게확인
> ```python
> from sklearn.metrics import classification_report
> print(classification_report(y_test, y_pred))
> ```
> 오차행렬 성능지표 확인
>```python
> import seaborn as sns
> from sklearn.metrics import confusion_matrix
> from sklearn.metrics import precision_score, recall_score
> 
> y_pred = model.predict(X_test)
> cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
>
> sns.heatmap(cm, annot=True)
> plt.show()
> 
> print(classification_report(y_test, y_pred))
> 
> precision_score(y_true, y_pred)
> recall_score(y_true, y_pred)
>```
> 
> ## 손실함수
> 회귀모델 손실함수(Loss Function)
> -   MSE(Mean Squared Error) : 실제에서 예측값 차이를 제곱, 합하여 평균 (예측)
> -   MAE(Mean Absolute Error) : 실제값 빼기 예측값 절댓값의 평균
> -   CEE(Cross Entropy Error) : 예측결과가 빗나갈수록 더큰패널티 부여 (분류)
> 분류모델 손실함수
> -   Binary Cross Entropy (이진분류)
> -   Multi Class Classfication (다중분류)
> ## 주요 지표
> -   loss = MSE (학습시 사용한 loss function종류에 의해 결정) (작을수록 좋음)
> -   error = 실제값 빼기 예측값의 평균 (작을수록 좋음)
> -   MSE = 실제값 빼기 예측값 제곱의 평균 (작을수록 좋음)
> -   MAE = 실제값 빼기 예측값 절댓값의 평균 (작을수록 좋음)
> -   R2(결정계수) = 독립변수가 종속변수를 얼마나 잘설명하는지 (클수록 좋음)
> ## RMSE값 확인하기

## [5.적용]

