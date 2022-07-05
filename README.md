- huggingface를 이용하여 네이버 댓글 긍부정 분류

## Data

- 네이버댓글

```
head -5 data/review.sorted.uniq.refined.shuf.train.tsv

positive        나름 괜찬항요 막 엄청 좋은건 아님 그냥 그럭저럭임... 아직 까지 인생 디퓨져는 못찾은느낌
negative        재질은플라스틱부분이많고요...금방깨질거같아요..당장 물은나오게해야하기에..그냥설치했어요..지금도 조금은후회중.....
positive        평소 신던 신발보다 크긴하지만 운동화라 끈 조절해서 신으려구요 신발 이쁘고 편하네요
positive        두개사서 직장에 구비해두고 먹고있어요 양 많아서 오래쓸듯
positive        생일선물로 샀는데 받으시는 분도 만족하시구 배송도 빨라서 좋았네요
```

## code structure

```
.
├── data
│   ├── review.sorted.uniq.refined.shuf.test.tsv
│   ├── review.sorted.uniq.refined.shuf.train.tsv
│   ├── review.sorted.uniq.refined.shuf.tsv
│   └── review.sorted.uniq.refined.tsv
├── hf_trainer_ouput
├── log
│   └── bert_logs
│       └── version_0
│           ├── events.out.tfevents.1656736525.minsoo.31481.0
│           └── hparams.yaml
├── Makefile
├── mypy.ini
├── notebook
│   └── eda.ipynb
├── README.md
├── requirements
│   ├── requirements.txt
│   └── test_requirements.txt
├── save_model
│   └── model.pth
├── src
│   ├── classify_plm.py
│   ├── data_load
│   │   ├── dataset.py
│   │   └── __init__.py
│   ├── finetune_plm_hf.py
│   ├── finetune_plm_native.py
│   ├── __init__.py
│   ├── lit_model
│   │   ├── __init__.py
│   │   └── lit_bert.py
│   └── utils
│       ├── __init__.py
│       └── utils.py
├── test
│   ├── __init__.py
│   └── test.py
└── tox.ini
```

## Dataset

- 미니배치에서는 문장의 길이가 다르면 안되니까 collate function을 이용하여 padding을 넣는다.
- `TextClssificationCollator` class로 collate function 구현
- `torch`에서는 아래처럼 `DataLoader`에서 사용된다.

```python
train_loader = DataLoader(
    TextClassificationDataset(texts[:idx], labels[:idx]),
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=TextClssificationCollator(tokenizer, max_length),
)
```

- `transformers`에서는 아래와 같이 `Trainer`에서 사용된다.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=TextClssificationCollator(
        tokenizer, config.max_length, with_text=False
    ),
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)
```

## Tokenize

- BPE 압축 알고리즘 사용
  - subword segmentation
  - data-driven 통계 방식
  - train set만 이용하여 만듬
  - 형태소 분석기 쓰고 BPE해도 됨
- `transformers`에서 제공하는 & 우리가 사용할 pretrained model을 만들 때 사용했던 tokenizer를 함께 다운받아서 사용한다.

## tensorboard

- tensorboard 사용
  - 아래와 같이 logger를 만들고 `tensorboard --logdir=log/bert_logs` 명령어를 실행하면 된다.

```python
logger = TensorBoardLogger(save_dir="../log", name="bert_logs")

trainer = pl.Trainer(
    gpus=1,
    max_epochs=config.n_epochs,
    logger=logger,
    callbacks=[checkpoint_callback],
    precision=16,
)
```

## callback 함수 구현

- pl에서 제공하는 callback 함수 중에서 `ModelCheckpoint`를 이용했다.
- callback 함수란 어떤 이벤트에 의해 호출되어지는 함수라고 이해할 수 있다.

```python
checkpoint_callback = ModelCheckpoint(
    dirpath="../save_model",
    filename="{epoch}-{val_loss:.2f}",
    save_top_k=1,
    every_n_epochs=1,
    monitor="val_loss",
    mode="min",
)

trainer = pl.Trainer(
    gpus=1,
    max_epochs=config.n_epochs,
    logger=logger,
    callbacks=[checkpoint_callback],
    precision=16,
)
```

## AMP (Automatic Mixed Precision, 2018)

- weight가 FP32이라고 해도 적절한 스케일링을 통해서 학습과정(forward, backward)에서는 FP16b를 사용
  - 적절한 스케일링을 하지 않으면 FP16bit가 표현하지 못해서 0이 되는 경우가 발생
- 속도는 더 빠르면서 메모리도 아끼고 성능은 비슷하거나 뛰어난 결과
- 모든 GPU에서 제공하는 것은 아니고 예외처리해야할 상황이 생길 수도 있음

## lightning-transformers

- https://github.com/Lightning-AI/lightning-transformers
  - 더 쉽게 사용할 수도 있다.

## Reference

- https://github.com/kh-kim/simple-ntc
- 패스트캠퍼스 김기현님 강의
