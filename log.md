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

## experiment_log-2025-08-21-0018-26

### Objective
- 把 global model contrastive learning 的部分先註解掉，只跑 fedavg 在 local training 的部分多跑一個 global model test accuracy on local test data 並回傳給 server，觀察是否合理
- 理論上要跟 fedavg 一樣

### Result
![result](result\fedavg_plus_global_test_accuracy_on_local_test_set.png)
- 不知道為甚麼跑到 200 round，結果跟 fedavg 不一樣
- 原因:不小心印兩次 global model test accuracy

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
- contrastive learning rate = 0.005

### Result
![result](result\cosine_delay_contrastive_learning_rate.png)
- 最尾端有跟上來

## experiment_log-2025-08-27-0030-44

### Objective
- 跟 experiment_log-2025-08-26-0028-30 一樣，只是 contrastive learning rate 調成 0.002

### Result
![result](result\contrastive_learning_rate_0.002.png)
- 最尾端有跟上來

## experiment_log-2025-08-28-0045-57

### Objective
- 跟 experiment_log-2025-08-26-0028-30 一樣，只是 contrastive learning rate 調成 0.001

### Result
![result](result\contrastive_learning_rate_0.001.png)
- 最尾端有跟上來
- 又比 0.002 好一些

## experiment_log-2025-08-29-0045-57

### Objective
- 跟 experiment_log-2025-08-26-0028-30 一樣，只是 contrastive learning rate 調成 0.0005

### Result
![result](result\contrastive_learning_rate_0.0005.png)
- 

