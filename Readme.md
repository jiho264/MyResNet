# Create ResNet
##### LEE, JIHO
> Dept. of Embedded Systems Engineering, Incheon National University

> jiho264@inu.ac.kr /  jiho264@naver.com
 
- The purpose of this project is to create a ResNet34 model using Pytorch.
- The Goal of this project is that to get the accuracy of near original paper's accuracy.
- The Origin ResNet32 have 7.51% top-1 error rate in CIFAR-10 dataset.
- The Origin ResNet34 have 21.53% top-1 error rate in ImageNet2012 dataset.
---
# 1. Usage
## 1.1. Requierments
  - ```Ubuntu 20.04 LTS```
  - ```Python 3.11.5```
  - ```Pytorch 2.1.1```
  - ```CUDA 11.8```
  - ```pip [sklearn, copy, time, tqdm, matplotlib]```
  - ```/data/ImageNet2012/train```
  - ```/data/ImageNet2012/val```
## 1.2. How to run 
  - Run ```........ipynb```
  - Options
    - ```BATCH = 256```
    - ```DATASET = {"CIFAR10", "CIFAR100", "ImageNet2012"}```
  - The trained model is ```............pth```
## 1.3. The Manual from Original Paper
### 1.3.1. Implementation about training process :
  - [x] We initialize the weights as on **He initialization**
  - [x] We adopt **batch normalization** after each convolutional and before activation
  - [x] We use **SGD** with a **mini-batch size of 256**
  - [x] The learning rate starts from **0.1** and is **divided by 10** when the error plateaus
  - [x] We use a **weight decay of 0.0001** and a **momentum of 0.9**
  - [x] We **do not use** dropout
  
### 1.3.2. ```MyResNet34``` preprocessing for ImageNet2012 :
  - [x] The image is resized with its shorter side randomly sampled in [256, 480] for scale augmentation [41]. 
  - [x] A 224×224 crop is randomly sampled from an image or its horizontal flip, with the per-pixel mean subtracted [21]. 
  - [x] The standard color augmentation in [21] is used.
    > torchvision.transform.AutoAugment() 적용함.
  - [x] In testing, for comparison studies we adopt the standard 10-crop testing [21]. For best results, we adopt the fully- convolutional form as in [41, 13], and average the scores at multiple scales (images are resized such that the shorter side is in {224, 256, 384, 480, 640}).
    > src/Prediction_for_MultiScaleTest.ipynb

### 1.3.3. ```MyResNet_CIFAR``` preprocessing for CIFAR10 :
  - [x] 45k/5k train/valid split from origin train set(50k)
  - [x] 4 pixels are padded on each side, and a 32 x 32 crop is randomly sampled from the padded image or its horizontal flip.
  - [x] For testing, use original images
---

# 2. Development Log
- Jan 10    
  - 거의 모든 사항 구현.
  - BN 
    - affine = True가 기본인데, 이래야 gamma and beta가 학습됨.
    - https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

- Jan 11
  - >재실험으로 실험내용 삭제
  - Origin과의 비교 결과
    - 같은 학습 method에서는 거의 동일한 Convergence 보임.
    - 500 epochs에서도 valid loss 줄지 않으면, 그간 최소 loss였던 weight를 저장함. -> 579에서 종료면 실제 학습은 79에서 제일 잘 나왔던 것.
      > 학습 방법의 차이에서 80%대 달성이 좌우되는 듯 하다.
  - ImageNet2012 학습 :
    - AMP를 train, valid, test 중 train의 forward pass에만 적용
      - https://tutorials.pytorch.kr/recipes/recipes/amp_recipe.html
    - Result
      ```
      [Epoch 23/5000] :
      100%|██████████| 5005/5005 [32:00<00:00,  2.61it/s]  
      Training time: 1920.08 seconds
      Train Loss: 1.5245 | Train Acc: 64.11%
      Valid Loss: 1.9787 | Valid Acc: 56.24%
       ```
- Jan 12 : 
  - >재실험으로 실험내용 삭제
  - ResNet 32 추가 (n에 따라 가변적으로 ResNet 생성 가능.) 
  - amp on/off 추가. ImageNet2012 학습하는 ResNet34일 때만 적용하도록 바꿈.
- Jan 13 : ResNet32 for CIFAR10
  - >재실험으로 실험내용 삭제
    - ~~train만 전처리 하고, valid, test에 ToTensor()만 적용시 507 epoch에서 stop되었고 acc 각각 100%, 80%, 58%로 나타남.~~
    - ~~on CIFAR10에서 testing시, origin image 32x32x3 썼다고했는데, submean을 하지 않고서는 도저히 이렇게 나오지 않는다.~~
    - ~~Submean하는게 맞는 것 같다.~~
    - [Jan 17] CIFAR10에서 Submean을 valid, test set에 적용하지 않아도 학습이 잘 되었음. 
- Jan 15 : 
  - build training logging process 
  - Model, Dataloader 둘 다 별도 py파일로 분리시킴.
  - case별 실험 및 비교위한 코드 정리 및 재정의 수행.
- Jan 16 : ResNet32 in CIFAR10
  - >재실험으로 실험내용 삭제
  - 하나 알게된 것 : 동일 모델을 test할 때마다 loss가 소숫점 2자리대까지 바뀌는 것을 확인함. 
    - 동일 weights이어도, 컴퓨터 계산의 한계 때문에 오차 발생하는 것으로 보임  
  - Q1 : Adam 논문에서는 Learning Rate alpha가 어떻게 변화하는가? 왜 lr의 재정의가 필요없다고 했는가?
  - Q2 : 왜 Adam보다 SGD가 더 학습이 잘 되었는가?
- Jan 17 : 
  - 아차..
    - **이틀 간 진행한 실험은 Adam과 SGD가 CIFAR10 & ResNet 구조에서 다른 성능을 낸다는 결론 이외 학습 결과는 중요하지 않음.**
    - 구현 실수로 첫 conv3x3 layer의 BN과 Relu를 빼먹었음.
    - Dataloader에서 training set 전처리도 오류있었음.
    - 이하 (commit **BUG FIX**) 수정 후 재실험 :
  - **MyResNet32_CIFAR_128_SGD** [End at Jan 17 23:22]
    ```py
    batch = 128
    split_ratio = 0    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = MultiStepLR(optimizer, milestones=[82, 123], gamma=0.1)
    EarlyStopCounter = 500
    train.transforms = Compose(
        ToTensor()
        Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[1, 1, 1], inplace=True)
        AutoAugment(interpolation=InterpolationMode.NEAREST, policy=AutoAugmentPolicy.CIFAR10)
        RandomCrop(size=(32, 32), padding=[4, 4, 4, 4], pad_if_needed=False, fill=0, padding_mode=constant)
        RandomHorizontalFlip(p=0.5)
    ) 
    test.transforms = ToTensor() 
    ```
    ```
    [Epoch 239/500] :
    100%|██████████| 391/391 [00:09<00:00, 43.04it/s]
    Train Loss: 0.0011 | Train Acc: 87.50%
    Test  Loss: 0.2361 | Test Acc: 92.82%
    Early stop!! best_eval_loss = 0.230629503420448
    ``` 
    > test_loss: 0.2305202476232301
    > test_acc: 92.63%
    > test_error: 7.37%


  - **MyResNet32_CIFAR_256_SGD** 
    ```py
    batch = 256
    split_ratio = 0.95    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(patiance=50, factor=0.1, cooldown=50)
    EarlyStopCounter = 150
    train.transforms = Compose(
        ToTensor()
        Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[1, 1, 1], inplace=True)
        AutoAugment(interpolation=InterpolationMode.NEAREST, policy=AutoAugmentPolicy.CIFAR10)
        RandomCrop(size=(32, 32), padding=[4, 4, 4, 4], pad_if_needed=False, fill=0, padding_mode=constant)
        RandomHorizontalFlip(p=0.5)
    ) 
    test.transforms = ToTensor() 
    ``` 
    ``` 
    [Epoch 850/1000] :
    100%|██████████| 196/196 [00:09<00:00, 20.04it/s]
    Train Loss: 0.0007 | Train Acc: 96.25%
    Test  Loss: 0.2224 | Test Acc: 93.72%
    Early stop!! best_eval_loss = 0.208574132155627
    ```  
    > test_loss: 0.20969283301383257
    > test_acc: 93.30%
    > test_error: 6.70%
    
  - **MyResNet34_ImageNet_256_SGD** - [case1] [End at Jan 19]
    ```py
    batch = 256
    split_ratio = 0    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(patiance=5, factor=0.1, cooldown=5)
    EarlyStopCounter = 25 # MyResNet32_CIFAR_256_SGD의 결과가 좋아서, 동일한 공식으로 sch, ealry 설정함.
    train = Compose(
        RandomShortestSize(min_size=range(256, 480), antialias=True),
        RandomCrop(size=224),
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        RandomHorizontalFlip(self.Randp),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1], inplace=True),
    )
    # center croped valid set
    valid = Compose(
        RandomShortestSize(min_size=range(256, 480), antialias=True),
        CenterCrop(size=368),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1], inplace=True),
    )
    # 10-croped valid set
    scales = [224, 256, 384, 480, 640]
    valid  = Compose(
        RandomShortestSize(min_size=scale[i], antialias=True)
        TenCrop(size=scale[i])
        ToTensor()
        Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1], inplace=True)
    )
    ``` 
    ``` 
    58 epoch: train_loss=0.0003, train_acc=0.6573, valid_loss=1.3160, valid_acc=0.7199, lr=0.0010
    59 epoch: train_loss=0.0003, train_acc=0.6224, valid_loss=1.3129, valid_acc=0.7204, lr=0.0010
    60 epoch: train_loss=0.0003, train_acc=0.6853, valid_loss=1.3108, valid_acc=0.7221, lr=0.0010
    61 epoch: train_loss=0.0003, train_acc=0.6993, valid_loss=1.2985, valid_acc=0.7242, lr=0.0010
    ```
  - **MyResNet34_ImageNet_256_SGD** - [case2]
    - paper의 figure 4와 비슷하게 스케쥴링함.
    ```
    scheduler = MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    EarlyStopCounter = 30
    ```
    ```
    
    ```
# 3. Conclusion
- Best ResNet32 Model on CIFAR10 
  - **<MyResNet32_CIFAR_128_SGD>**
    > test_loss: 0.2305202476232301
    > test_acc: 92.63%
    > test_error: 7.37%
    - test loss에 기반해 스케쥴링하지 않고, MultiStepLR로 명시적인 Learning rate들을 적용함. 
    - Validation set 따로 만들지 않고, 50K의 Training set 다 학습 시킨 것의 결과가 좋았음.
      - **<MyResNet32_CIFAR_128_SGD_90>에 비해 2.93%의 Acc 향상있음.**
    - 한정된 Training set의 환경에서 5k개의 추가 데이터는 원활한 학습에 큰 도움이 되었음.
- Best ResNet34 model on ImageNet2012
  - 연구 진행 중..
     