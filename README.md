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
