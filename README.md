## Dataset

- 미니배치에서는 문장의 길이가 다르면 안되닌까 collate function을 이용하여 padding을 넣는다.

## Tokenize

- BPE 압축 알고리즘 사용
  - subword segmentation
  - data-driven 통계 방식
  - train set만 이용하여 만듬
  - 형태소 분석기 쓰고 BPE해도 됨
