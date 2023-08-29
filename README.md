# unbiased-learning-to-rank

# 実験概要 (まだ未完成)
UnbiasedなPointwise損失関数とListwise損失関数の比較を行う

## 目的
推薦の文脈でunbiasedなListwise損失関数が機能し、Pointwise損失関数よりもランク指標が高くなることを確かめたい。

## 懸念事項 
論理的にはunbiasedなPointwise, Listwise損失関数は機能するが、完全にImplicitなデータしかない場合、biasedな推定量と有意な差が見られない

# 実験設定
## 真の分布を仮定した人工的な評価値行列を生成
How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility (https://arxiv.org/abs/1710.11214) と同じ生成方法  

### 真の嗜好度合い行列
$V_{u,i} \in [0,1]$

## クリックモデルに従い、Implicitフィードバックを生成
### クリック変数 
$Y_{u,i}$  

### 嗜好度合い変数
$R_{u,i} \sim \text{Bernoulli}(V_{u,i})$  

### 露出変数 (今回は人気なアイテムほどログに残りやすい設定)
  
$\theta_{u,i} = \frac{\sum_{u \in D}V_{u,i}}{\max i \in D \sum_{u \in D}V_{u,i}}$  
$O_{u,i} \sim \text{Bernoulli}(\theta_{u,i})$


### biased click model (train, valデータ)
$P(Y_{u,i} = 1) = P(R_{u,i} = 1) P(O_{u,i} = 1)$

### unbiased click model (test データ)
$P(Y_{u,i} = 1) = P(R_{u,i} = 1)$

### ログデータ
$D = \(u, i, Y_{u,i}) \sim \pi_{b}$  
$\pi_{b}$: 過去の推薦方策はランダムpolicyとする。(分布が乖離している原因が露出変数だけであることを仮定するため)

# unbiased pointwise loss (Relevance-MF (https://arxiv.org/pdf/1909.03601.pdf)) 

$$J_{unbiased} = \sum_{u \in D}\sum_{i \in D}\frac{Y_{u,i}}{\theta_{u,i}}\log\sigma(\hat{R_{u,i}}) + (1 - \frac{Y_{u,i}}{\theta_{u,i}})\log\sigma(1 - \hat{R_{u,i}})$$

$$\hat{R_{u,i}} = \mathbf{q_i}^T\mathbf{p_u} + b + \mathbf{b_u} + \mathbf{b_i}$$

$$ \sigma(x) = \frac{1}{1 + e^{-x}}$$

# unbiased Listwise loss (unbiased ListRank-MF)
ListRank-MF (https://dl.acm.org/doi/abs/10.1145/1864708.1864764?casa_token=MUgYve_rEOoAAAAA:j1ljuuHOeQ3ic8s_dtv5xSA2SLZQbQUio74JCfCoob5YDSdPCxQkANgLY2RRiqOF0xzYHcQghUR5) をunbiasedに拡張した新たな推定量  

$$ J_{unbiased} = \sum_{u \in D}\sum_{i \in D_u}\frac{Y_{u,i}}{\theta_{u,i}}\frac{\exp(\mathbf{q_i}^T\mathbf{p_u})}{\sum_{i' \in D_u} \exp(\mathbf{q_i}^T\mathbf{p_u})}$$


