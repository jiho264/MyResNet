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

# 2. Training Results
## 2.1. CIFAR10
```py 
train.transforms = Compose(
    ToTensor()
    Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[1, 1, 1], inplace=True)
    AutoAugment(interpolation=InterpolationMode.NEAREST, policy=AutoAugmentPolicy.CIFAR10)
    RandomCrop(size=(32, 32), padding=[4, 4, 4, 4], pad_if_needed=False, fill=0, padding_mode=constant)
    RandomHorizontalFlip(p=0.5)
) 
test.transforms = ToTensor() 
```
### 2.1.1. MyResNet32_CIFAR_128_SGD [End at Jan 17]
- batch = 128
- split_ratio = 0    
- optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
- scheduler = MultiStepLR(optimizer, milestones=[82, 123], gamma=0.1)
- EarlyStopCounter = 500
> test_loss: 0.2305202476232301
> test_acc: 92.63%
> test_error: 7.37%
> 100 epochs 넘게 converge했기에, 239epochs에서 종료. 

### 2.1.2. MyResNet32_CIFAR_128_SGD_90 [Training now..]
- batch = 128
- split_ratio = 0.9    
- optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
- scheduler = ReduceLROnPlateau(patiance=10, factor=0.1, cooldown=50)
- EarlyStopCounter = 70
> test_loss: 0.5532871841252605
> test_acc: 83.39%
> test_error: 16.61%
>> 232 epochs에서 lr 1e-5로 감소해서 종료.

### 2.1.3. MyResNet32_CIFAR_128_SGD_95 [End at Jan 22]
- batch = 128
- split_ratio = 0.95
- optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
- scheduler = ReduceLROnPlateau(patiance=10, factor=0.1, cooldown=50)
- EarlyStopCounter = 70 
> test_loss: 0.33611102500321355
> test_acc: 89.40%
> test_error: 10.60%
>> 205 epochs에서 Early stop

---
## 2.2. ImageNet2012
```py
# Training set
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
    RandomShortestSize(min_size=scale[i]+1, antialias=True)
    TenCrop(size=scale[i])
    ToTensor()
    Normalize(mean=[0.485, 0.456, 0.406], std=[1, 1, 1], inplace=True)
)
```
### 2.2.1. MyResNet34_ImageNet_256_SGD_1 [End at Jan 19]
- batch = 256
- optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
- scheduler = ReduceLROnPlateau(patiance=5, factor=0.1, cooldown=5)
- EarlyStopCounter = 25 

``` 
[Epoch 68/500] :
100%|██████████| 5005/5005 
Train Loss: 0.0003 | Train Acc: 62.24%
Valid Loss: 1.2975 | Valid Acc: 72.39%
```
> Avg Loss: 26.3212
> Avg Top-1 Acc: 47.93%
> Avg Top-5 Acc: 71.18%
>> Train set에서 acc가 낮은 현상 때문에, Test(10-crop)에서도 47%의 Top-1 Acc나옴. 
>> Train set도 acc올라올 때 까지 다시 학습시켜야 할 것 같음.

### 2.2.2. MyResNet34_ImageNet_256_SGD_2 [Training now...]
- case1보다 cooldown을 5에서 25로 늘림. 얼리스탑 카운터도 25에서 40으로.
- batch = 256  
- optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
- scheduler = ReduceLROnPlateau(patiance=10, factor=0.1, cooldown=25)
- EarlyStopCounter = 40

> test_loss: 0.20692915320396424
> test_acc: 92.92%
> test_error: 7.08%

# 3. Conclusion
## Best ResNet32 Model on CIFAR10 
  - MyResNet32_CIFAR_128_SGD
    > test_loss: 0.2305202476232301
    > test_acc: 92.63%
    > test_error: 7.37%
    - test loss에 기반해 스케쥴링하지 않고, MultiStepLR로 명시적인 Learning rate들을 적용함. 
      - 명시적인 lr 감소는 경험에 기반한 것인데, 이를 알아내기 위해선 시행착오가 필요함.
    - split한 것과의 비교
## Best ResNet34 model on ImageNet2012
  - MyResNet34_ImageNet_256_SGD (120m 소요)
    >Dataset 1: Loss: 25.1631, Top-1 Acc: 0.4722, Top-5 Acc: 0.7023
    >Dataset 2: Loss: 23.1898, Top-1 Acc: 0.50176, Top-5 Acc: 0.730776
    >Dataset 3: Loss: 23.3688, Top-1 Acc: 0.527296, Top-5 Acc: 0.75704
    >Dataset 4: Loss: 26.6102, Top-1 Acc: 0.496662, Top-5 Acc: 0.731202
    >Dataset 5: Loss: 33.2743, Top-1 Acc: 0.398678, Top-5 Acc: 0.637878
    >Avg Loss: 26.3212, Avg Top-1 Acc: 0.4793, Avg Top-5 Acc: 0.7118
    >> Top-1 47.93%, Top-5 71.18%
     