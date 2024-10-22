# 完全雲端運行:使用Google CoLaboratory訓練神經網路。
南華大學跨領域-人工智慧期中報告

11215027劉語涵
# CoLaboratory 訓練神經網路

本文旨在展示如何使用CoLaboratory 訓練神經網路。我們將展示一個在威斯康辛乳癌資料集上訓練神經網路的範例，資料集可在UCI Machine Learning Repository（http://archive.ics.uci.edu/ml/datasets）取得。本文的範例相對比較簡單。

本文所使用的CoLaboratory notebook 連結：https://colab.research.google.com/notebook#fileId=1aQGl_sH4TVehK8PDBRspwI4pD16xIR0r
# 深度學習

深度學習（英語：deep learning）是機器學習的分支，是一種以人工神經網路為架構，對資料進行表徵學習的演算法。
# 程式碼
問題：研究者取得乳房腫塊的細針穿刺（FNA），然後產生數位影像。此資料集包含描述影像中細胞核特徵的實例。每個實例包括診斷結果：M（惡性）或B（良性）。我們的任務是在該數據上訓練神經網路根據上述特徵診斷乳癌。

開啟CoLaboratory，出現一個新的untitled.ipynb 檔案供你使用。

Google有一台linux 虛擬機，這樣就可以存取終端為專案安裝特定套件。如果在程式碼單元中輸入(!ls )指令，那麼你的虛擬機器中會出現一個simple_data 資料夾。

![image](https://github.com/Han931026/report1/blob/main/!ls(%E6%88%AA%E5%9C%96).png)

首先將資料集放置到該機器上，這樣我們的notebook 就可以存取它。你可以使用以下程式碼：

```
#Uploading the Dataset

from google.colab import files

uploaded = files.upload()

with open("breast_cancer.csv", 'wb') as f:
    f.write(uploaded[list(uploaded.keys())[0]])
```
輸入!ls 指令，查看機器上是否有該檔案。你將會看到datalab 資料夾和breast_cancer_data.csv 檔案。

![image](https://github.com/Han931026/report1/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202024-10-21%20092315.png)

資料預處理:

現在數據已經在機器上了,我們使用pandas將其輸入到檔案中。

```

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing dataset
dataset = pd.read_csv('breast_cancer.csv')



#Check the first 5 rows of the dataset. 

    dataset.head(5)

```

![image](https://github.com/Han931026/report1/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202024-10-23%20002945.png)

CoLaboratory 上的輸出結果圖示。

現在，分割因變數（Dependent Variables）和自變數（Independent Variables）。

```

#Seperating dependent and independent variables. 



X = dataset.iloc[:, 2:32].values  #Note: Exclude Last column with all NaN values.
y = dataset.iloc[:, 1].values

```

Y 包含一列，其中的「M」和「B」分別代表「是」（惡性）和「否」（良性）。我們需要將其編碼成數學形式，即“1”和“0”。可以使用Label Encoder 類別完成此任務。

```
#Encoding Categorical Data

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()



y = labelencoder.fit_transform(y)

```

現在資料已經準備好，我們將其分割成訓練集和測試集。在Scikit-Learn 中使用train_test_split 可以輕鬆完成這項工作。

```

#Splitting into Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

```
參數test_size = 0.2 定義測試集比例。這裡，我們將訓練集設定為資料集的80%，測試集佔資料集的20%。

#Keras:

Keras 是一種建構人工神經網路的高階API。它使用TensorFlow 或Theano 後端執行內部運作。要安裝Keras，必須先安裝TensorFlow。CoLaboratory 已經在虛擬機器上安裝了TensorFlow。使用以下指令可以檢查是否安裝TensorFlow：

!pip show tensorflow

你也可以使用!pip install tensorflow==1.2，安裝特定版本的TensorFlow。

另外，如果你更喜歡用Theano 後端，可以閱讀該文件：https://keras.io/backend/。

# 安裝Keras：

!pip install -q keras

```

# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense

```

使用Sequential 和Dense 類別指定神經網路的節點、連接和規格。如上所示，我們將使用這些自訂網路的參數並進行調整。

為了初始化神經網絡，我們將建立一個Sequential 類別的物件。

```

# Initialising the ANN

classifier = Sequential()

```

#設計網路

對於每個隱藏層，我們需要定義三個基本參數：units、kernel_initializer 和activation。 units 參數定義每層包含的神經元數量。 Kernel_initializer 定義神經元在輸入資料上執行時的初始權重（詳見https://faroit.github.io/keras-docs/1.2.2/initializations/）。 activation 定義資料的激活函數。
