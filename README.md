# kakr

```
kakr-4th-competition
```

## 데이터 세부 설명
### train/test는 14개의 columns으로 구성되어 있고, train은 예측해야 하는 target 값 feature까지 1개가 추가로 있습니다. 각 데이터는 다음을 의미합니다.

+ id
+ age : 나이
+ workclass : 고용 형태
+ fnlwgt : 사람 대표성을 나타내는 가중치 (final weight의 약자)
+ education : 교육 수준
+ education_num : 교육 수준 수치
+ marital_status: 결혼 상태
+ occupation : 업종
+ relationship : 가족 관계
+ race : 인종
+ sex : 성별
+ capital_gain : 양도 소득
+ capital_loss : 양도 손실
+ hours_per_week : 주당 근무 시간
+ native_country : 국적
+ income : 수익 (예측해야 하는 값)

  '>50K : 1' 
  
  '<=50K : 0'
