# Record what was tested in each experiment

## experiment_log-2025-07-29-0149-53

### Objective
- moon baseline
- mu = 5
- 原封不動的 code

## experiment_log-2025-08-04-0055-32

### Objective
- fedavg baseline
- mu = 5
- 原封不動的 code

## experiment_log-2025-08-06-0101-36

### Objective
- fedprox baseline
- 這邊的 mu 要記得改成 0.01
- 原封不動的 code


## experiment_log-2025-08-22-0042-48

### Objective
- 把 global model contrastive learning 的部分先註解掉
- local training 的部分先照搬 fedavg，然後在加 global model test accuracy on local test data 上去
- 理論上要跟 fedavg 一樣

### Result
![result](result\comment_contrastive_learning_part.png)
- 目前還是不知道為甚麼只有在 train_net_my (我的 algorithm 的 local training function) 比 train_net (fedavg algorithm 的 local training function) 多加 global model 來計算 global model test accuracy on local test set，還有回傳 test accuracy 的部分
- 實驗結果是差不多，但是理論上應該要一樣 (epoch loss ...)
    - 沒有動到 local training 的部分但是結果卻不完全一樣
- 確定不是 data loader 的問題，有測過改過 data loader 的部分跑 fedavg，跟作者最原始的 code 跑出來是一樣的
- 確定是 utils.py 的 compute_accuracy 在 evaluation 跟 training mode 之間的轉換造成的

## experiment_log-2025-08-24-0129-37

### Objective
- 加上 contrastive learning 的部分 (讓 global model 靠近 global test accuracy above average 的 models)
- positive samples 的部分沒有取 mean
- contrastive learning rate 的部分是用 base learning rate * (current round/ max round)

### Result
![result](result\pos_accuracy_above_average_model_not_mean.png)
- 初期的上升幅度感覺比 baseline 高，後面又掉下來

## experiment_log-2025-08-26-0028-30

### Objective
- contrastive learning rate 用 cosine delay
- base contrastive learning rate = 0.005
```
scaled_lr = 0.5 * base_lr * (1 + math.cos(math.pi * t))
```

### Result
![result](result\cosine_delay_contrastive_learning_rate.png)
- 最尾端有跟上來

## experiment_log-2025-08-27-0030-44

### Objective
- 跟 experiment_log-2025-08-26-0028-30 一樣，只是 contrastive base learning rate 調成 0.002

### Result
![result](result\cosine_delay_0.002.png)
- 最尾端有跟上來

## experiment_log-2025-08-28-0045-57

### Objective
- 跟 experiment_log-2025-08-26-0028-30 一樣，只是 contrastive base learning rate 調成 0.001

### Result
![result](result\cosine_decay_0.001.png)
- 最尾端有跟上來
- 又比 0.002 好一些

## experiment_log-2025-08-29-0045-57

### Objective
- 跟 experiment_log-2025-08-26-0028-30 一樣，只是 contrastive base learning rate 調成 0.0005

### Result
![result](result\contrastive_learning_rate_0.0005.png)

## experiment_log-2025-09-05-0000-25

### Objective
- contrastive base learning rate = 0.001
- contrastive learning rate -> -cosine delay (cosine warmup) ，從 0 以 -cos 慢慢增加到 0.001
```
scaled_lr = 0.5 * base_lr * (1 - math.cos(math.pi * t))
```

### Result
![result](result\cosine_warmup_0.001.png)
- 在 10 round 以前，after contrastive learning 都上升的比 baseline 快，後面掉下來
- 最後沒收斂
- 比較 
    - cosine decay (以 cosine 函數從 base learning rate 降到 0) -> experiment_log-2025-08-28-0045-57
        - 初期上升的比 baseline 差，但是最後面有跟上來
    - -cosine warmmup (以 cosine 函數從 0 上升到 base learning rate) -> experiment_log-2025-09-05-0000-25
        - 初期 after contrastive learning 上升的比 baseline 快，10 round 以後變得很不好
- 結論
    - cosine decay 後期好， -cosine decay 初期好
    - 有可能是 learning rate 還太大
    - 有可能直接維持 0.001 就好

## experiment_log-2025-09-06-0055-22

### Objective
- contrastive base learning rate = 0.0005
- contrastive learning rate -> -cosine delay，從 0 以 -cos 慢慢增加到 0.0005

### Result
![result](result\cosine_warmup_0.0005.png)
- 比 0.001 稍微好一點，但是最後還是沒收斂

## experiment_log-2025-09-06-1900-33

### Objective
- 固定 contrastive learning rate = 0.0005

### Result
![result](result\fixed_learning_rate_0.0005.png)

## experiment_log-2025-09-07-1447-02

### Objective
- contrastive base learning rate = 0.0005
- 根據前面的實驗發現開頭和結尾都是接近 0 表現比較好，所以把 learning rate 改成從 0 上升到 base learning rate 再下降回 0 (以 cos 函數移動)
    - round = 100
    - round = 0 -> learning rate = 0
    - round = 49 -> learning rate = base learning rate
    - round = 99 -> learning rate = 0

### Result
![result](result\cosine_warmup_decay_0.0005.png)
- 開頭和結尾確實不錯，但中間 after contrastive learning 還是有點不好，有可能是 learning rate 還太大
- 最後 accuracy 介於 fedavg 和 fedprox 中間

## experiment_log-2025-09-12-1108-46

### Objective
- contrastive base learning rate = 0.00025
- 跟上一個實驗一樣 (experiment_log-2025-09-07-1447-02)，先 cosine warmup 再 cosine decay

### Result
![result](result\cosine_warmup_decay_0.00025.png)

## experiment_log-2025-09-14-0122-31

### Objective
- contrastive base learning rate = 0.000125
- 跟上一個實驗一樣 (experiment_log-2025-09-07-1447-02)，先 cosine warmup 再 cosine decay

### Result
![result](result\cosine_warmup_decay_0.000125.png)

## experiment_log-2025-09-14-1914-30

### Objective
- contrastive base learning rate = 0.000075
- 跟上一個實驗一樣 (experiment_log-2025-09-07-1447-02)，先 cosine warmup 再 cosine decay

### Result
![result](result\cosine_warmup_decay_0.000075.png)

## experiment_log-2025-09-19-0028-48

### Objective
- moon 演算法的 local contrastive loss 的 coefficient 改成 cosine warmup + cosine decay 
- base coefficient = 0.01

### Result
![result](result\moon_cosine_warmup_decay_coefficient_contrastive_loss.png)


## experiment_log-2025-09-19-1507-55

### Objective
- global contrastive learning rate -> cosine decay
- base learning rate = 0.0005

### Result
![result](result\cosine_decay_0.0005.png)

## experiment_log-2025-09-20-1113-51

### Objective
- average positive -> only one positive sample
- base learning rate = 0.0005

### Result
![result](result\average_positive_sample_0.0005.png)

## experiment_log-2025-09-27-0124-44

### Objective
- Add server-side contrastive learning to MOON (client-side) contrastive learning
- Use multi-positive instead of averaging positive samples for server contrastive learning
- base learning rate = 0.0005

### Result
![result](result\server-side_moon_0.0005.png)

## experiment_log-2025-09-27-2148-19

### Objective
- Same as experiment_log-2025-09-27-0124-44
- base learning rate = 0.00025

### Result
![result](result\server-side_moon_0.00025.png)

## experiment_log-2025-09-29-0113-30

### Objective
- Same as experiment_log-2025-09-27-0124-44
- base learning rate = 0.000125

### Result
![result](result\server-side_moon_0.000125.png)

---

# experiment_log-2025-10-27-2241-12

## Objective
- Test run FedAvg algorithm

## Result
- About 7 hour

# experiment_log-2025-10-28-2022-13

## Objective
- Test run
    - positive sample: mean global test accuracy on local test set > average
    - negative sample: else
- Cosine warmup + Cosine decay

## Result
![result](result\test.png)

# experiment_log-2025-10-29-1010-16

## Objective
- Test run MOON

# experiment_log-2025-10-29-1804-34

## Objective
- Test run FedProx

# experiment_log-2025-10-30-0141-07

## Objective
- Use f1-score on training data for differentiating positive and negative samples

## Result
![result](result\f1-score_on_training_data.png)
- 10 round 以前上升的比 MOON 好，但是最後沒有收斂，需要調 learning rate 的感覺

