## [Incontext Learning](https://github.com/KJ-Min/Natural-Language-Processing/blob/master/Incontext_Learning.ipynb)

### 설명
이 코드에서는 In-context Learning 기법을 활용하여 언어 모델의 추론 능력을 테스트했습니다. 
In-context Learning은 모델의 파라미터를 직접 업데이트하지 않고도, 프롬프트 엔지니어링만으로 모델이 특정 작업을 수행할 수 있도록 하는 기법입니다. 
실제로 모델을 재학습시키는 것이 아님에도 "Learning"이라고 부르는 이유는 모델이 주어진 맥락(context)에서 패턴을 파악하고 이를 통해 새로운 입력에 대응하는 방식이 학습과 유사하기 때문입니다.

### 사용 모델
- [meta-llama/Llama-3.2-3B-Instruct (HuggingFace)](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) 

### 구현 내용
- **모델 및 토크나이저 초기화**
   - HuggingFace의 `transformers` 라이브러리를 사용하여 모델과 토크나이저를 초기화

-  **Task 1: 간단한 추론 테스트**
   - 예시 프롬프트: "Which is a fruit? (a) apple, (b) car, (c) dog"
   - 모델이 논리적으로 올바른 대답을 반환하는지 확인

-  **Task 2: 상식 추론 문제 테스트**
   - [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa) 데이터셋에서 10개의 질문을 선별하여 사용
   - 다양한 상식 질문에 대한 모델의 응답 정확도 측정
   - **결과**: 10개 질문 중 7개 정답 (정확도: 70%)
