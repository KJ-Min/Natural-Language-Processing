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

## [Q-Learning for CartPole](https://github.com/KJ-Min/Natural-Language-Processing/blob/master/Q-Learning.ipynb)

### 설명
이 코드에서는 OpenAI Gym 라이브러리의 `CartPole-v1` 환경을 이용하여 Q-러닝(Q-Learning) 강화학습 알고리즘을 구현하고 테스트했습니다. 
Q-러닝은 에이전트가 특정 상태에서 어떤 행동을 취했을 때 받을 미래 보상의 기댓값(Q-값)을 학습하여 최적의 행동 정책을 찾아나가는 알고리즘입니다. 
CartPole 환경의 목표는 카트를 좌우로 움직여 카트 위에 붙어있는 막대가 쓰러지지 않도록 균형을 최대한 오래 유지하는 것입니다.

### 사용 환경 및 라이브러리
- **환경:** `CartPole-v1` (OpenAI Gym)
- **주요 라이브러리:** `gym`, `numpy`, `matplotlib`

### 구현 내용
- **상태 이산화 (State Discretization)**
   - CartPole 환경의 연속적인 상태 공간(카트 위치, 카트 속도, 막대 각도, 막대 각속도)을 Q-테이블에서 사용 가능한 이산적인 상태로 변환하기 위해 버킷팅(Bucketing) 기법을 사용했습니다.
   - 상태 변수별 경계값(`STATE_BOUNDS`)과 버킷 수(`NUM_BUCKETS`)를 정의하고, `state_to_bucket` 함수를 구현하여 연속 상태를 이산적인 버킷 인덱스 튜플로 매핑했습니다.

- **Q-테이블 기반 학습**
   - 상태 버킷과 행동(좌/우 이동)을 차원으로 가지는 Q-테이블을 생성하고 0으로 초기화했습니다.
   - 에이전트가 환경과 상호작용하며 얻는 경험(상태, 행동, 보상, 다음 상태)을 바탕으로 Q-러닝 업데이트 공식을 사용하여 Q-테이블의 값을 반복적으로 갱신했습니다.

     $$ 
     Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right] 
     $$

- **행동 선택 정책 (Epsilon-Greedy)**
   - 학습 중 탐험(Exploration)과 활용(Exploitation)의 균형을 맞추기 위해 엡실론-그리디(Epsilon-Greedy) 정책을 사용했습니다.
   - 학습 초반에는 높은 확률(엡실론)로 무작위 행동을 선택하여 다양한 상태를 탐험하고, 학습이 진행됨에 따라 엡실론 값을 점차 감소시켜 학습된 Q-값을 활용하는 비율을 높였습니다.

- **동적 하이퍼파라미터 조정**
   - 학습률(α)과 탐험률(ε)이 에피소드 진행에 따라 동적으로 감소하도록 `get_learning_rate` 와 `get_explore_rate` 함수를 구현하여 적용했습니다. 이는 학습 안정성과 수렴 속도를 개선하는 데 도움이 됩니다.

- **학습 결과 시각화**
   - `matplotlib` 라이브러리를 사용하여 에피소드별 총 보상과 이동 평균선을 그래프로 시각화하여 학습 과정을 모니터링하고 결과를 평가했습니다.

### 결과
- 구현된 Q-러닝 에이전트는 학습을 통해 CartPole 환경에서 막대의 균형을 성공적으로 유지하는 정책을 학습했습니다.
- 학습 결과 그래프에서 총 보상이 환경의 최대 타임스텝인 500에 도달하고 안정적으로 유지되는 것을 확인하여, 에이전트가 목표를 효과적으로 달성했음을 보여줍니다.

## [Policy Gradient (REINFORCE) for CartPole](https://github.com/KJ-Min/Natural-Language-Processing/blob/master/Policy-Gradient.ipynb)

### 설명
이 코드에서는 OpenAI Gym 라이브러리의 `CartPole-v1` 환경을 이용하여 Monte-Carlo Policy Gradient, 흔히 REINFORCE라고 불리는 강화학습 알고리즘을 구현하고 테스트했습니다. 
Policy Gradient 방법은 보상을 최대화하는 방향으로 정책(Policy) 자체를 직접 학습하는 접근 방식입니다. 
에이전트는 신경망으로 구현된 정책을 통해 현재 상태에서 어떤 행동을 취할지에 대한 확률 분포를 학습하고, 이 분포에 따라 행동을 샘플링하여 환경과 상호작용합니다. 
REINFORCE는 에피소드 전체를 경험한 후, 각 타임스텝에서 얻은 할인된 누적 보상(Discounted Return)을 사용하여 정책을 업데이트하는 Monte-Carlo 방식의 알고리즘입니다.

### 사용 환경 및 라이브러리
- **환경:** `CartPole-v1` (OpenAI Gym)
- **주요 라이브러리:** `gym`, `numpy`, `torch`, `torch.nn`, `torch.optim`, `torch.distributions`, `matplotlib`

### 구현 내용
- **정책 신경망 (Policy Network)**
   - `torch.nn.Module`을 사용하여 상태를 입력받아 행동 확률 분포를 출력하는 신경망을 정의했습니다. 이 신경망은 에이전트의 정책 $\pi_\theta(a|s)$를 근사합니다.
   - Softmax 활성화 함수를 사용하여 출력값이 행동 확률 분포를 나타내도록 했습니다.

- **Monte-Carlo 기반 학습**
   - 에이전트는 현재 정책에 따라 에피소드를 끝까지 실행하여 상태, 행동, 보상의 궤적(trajectory)을 생성합니다.
   - 에피소드가 종료된 후, 각 타임스텝 $t$에서의 할인된 누적 보상 $G_t = \sum_{k=t}^{T} \gamma^{k-t} R_k$를 계산합니다. 보상 정규화를 통해 학습 안정성을 높였습니다.

- **정책 경사 업데이트 (Policy Gradient Update)**
   - 계산된 $G_t$와 행동의 로그 확률 $\log \pi_\theta(a_t|s_t)$를 사용하여 정책 경사를 추정하고, 이를 통해 정책 신경망의 파라미터 $\theta$를 업데이트합니다.
   - 목표 함수 $J(\theta)$를 최대화하기 위해 경사 상승법(Gradient Ascent)을 사용합니다. 코드에서는 일반적으로 손실 함수 $L(\theta) = -\sum_t G_t \log \pi_\theta(a_t|s_t)$ 를 정의하고 경사 하강법(Gradient Descent)을 적용하여 구현합니다.
   - `torch.optim.Adam` 옵티마이저를 사용하여 파라미터를 업데이트했습니다.

- **학습 결과 시각화**
   - `matplotlib` 라이브러리를 사용하여 에피소드별 총 보상과 이동 평균선을 그래프로 시각화하여 학습 과정을 모니터링하고 알고리즘의 성능을 평가했습니다.

### 결과
- 구현된 REINFORCE 에이전트는 학습을 통해 CartPole 환경에서 막대의 균형을 성공적으로 유지하는 정책을 점진적으로 학습했습니다.
- 학습 결과 그래프에서 이동 평균 보상이 꾸준히 증가하여 환경 해결 기준(예: 최근 100개 에피소드 평균 보상 475점)에 도달하는 것을 확인할 수 있습니다. Policy Gradient 방법의 특성상 에피소드별 보상의 분산이 크게 나타날 수 있지만, 이동 평균을 통해 전반적인 성능 향상 추세를 확인할 수 있었습니다.
