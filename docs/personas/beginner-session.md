# mi-copilot persona session — `beginner`

_Generated: 2026-06-02T12:06:43_

## Q1

> 신경망이 '갑자기 일반화' 한다는 grokking 현상이 뭔가요? 그리고 그게 왜 흥미로운가요?

### mi-copilot answer

Grokking 현상은 신경망이 오랫동안 학습을 하면서 처음에는 학습 데이터만 '기억(memorize)'하여 훈련 손실만 낮추다가, 훨씬 더 오랜 학습 뒤 갑자기 일반화(generalization)를 해서 테스트 손실이 급격히 떨어지는 현상입니다. 즉, 모델이 오랜 시간 '외우기'만 하다가 어느 순간 빠른 전환을 통해 '일반화하는 법'을 터득합니다 ("abrupt shift during training, when the model switches from memorization to generalization" — Nanda 2023). 이때 훈련 정확도와 테스트 정확도의 차이가 오랫동안 지속되다가, 'phase transition'처럼 갑자기 테스트 성능이 좋아집니다.

이 현상이 흥미로운 이유는, 보통 신경망의 능력이 점진적으로 늘어난다고 생각하지만, 실제로는 아주 짧은 기간에 '질적으로' 새로운 능력을 갑자기 획득한다는 점입니다. 예를 들어, induction heads가 언어 모델에서 갑자기 등장하는 현상도 이런 급격한 변화의 일례입니다 (Nanda 2023 Appendix D.3). 이런 현상은 일반화 메커니즘이 Gradual하게 강화되다가, 나중에 '외우는' 부분이 제거되면서(test loss가 급격히 좋아짐) 완전히 새로운 단계로 모델이 넘어가는 것으로 분석됩니다 ("the sudden transition to perfect test accuracy in grokking occurs during cleanup, after the generalizing mechanism is learned" — Nanda 2023).

참고 출처:  
- Nanda 2023 Progress Measures Grokking Modular Addition (본문, Appendix D.3, §2)  
- TransformerLens docs (Grokking demo 설명)

---

## Q2

> Nanda 2023 라는 논문이 modular addition 으로 grokking 회로를 분석했다고 들었는데, 모델이 어떤 알고리즘을 학습한 거죠?

### mi-copilot answer

Nanda 2023에서는 소형 transformer 모델을 (a + b) mod P, 즉 modular addition 문제에 학습시켰습니다. 학습 과정에서, 모델은 처음엔 과적합하다가 나중에 일반화된 알고리즘(“grokking”)으로 급격히 전환하는 현상을 보였습니다. 논문에 따르면, 모델이 학습한 알고리즘은 입력 a, b에 대해 c = (a + b) mod P를 예측하는 동작을 수행하며, 이 과정에서 회로가 주파수 기반의 해법(푸리에 기저 기반 회로)을 형성한 것이 확인되었습니다.

즉, 모델 내부에서는 "푸리에 변환" 구조를 사용해 modular addition을 처리하는 서킷이 구현됩니다. 학습이 진행됨에 따라 네트워크가 서서히 이 일반화 회로 구조로 발전해가며, 이 회로는 단순한 테이블 룩업이 아니라 연산 그 자체를 내재화한 형태임이 밝혀졌습니다.

출처: Nanda 2023, Progress Measures for Grokking via Mechanistic Interpretability §3 및 Figure 2, 4. (Source: papers · Nanda_2023_Progress_Measures_Grokking_Modular_Addition.pdf)

---

## Q3

> 그 알고리즘을 처음부터 검증하려면 어떤 코드를 짜야 하나요? 어떤 라이브러리가 필요한지도 알려주세요.

### mi-copilot answer

Charton 2024의 GCD sieve 알고리즘을 처음부터 검증하려면 다음과 같은 접근이 가장 적합합니다:

1. mi-copilot 라이브러리 설치  
mi-copilot은 이미 Charton 2024 GCD 논문의 검증용 코드를 제공합니다. 먼저 라이브러리를 설치합니다.

```bash
pip install -e .
cp .env.example .env  # OPENAI_API_KEY 추가
mi-copilot ingest     # 로컬 벡터스토어 빌드 (최초 1회)
```
(Source: mi_copilot_repo · README.md)

2. 검증 코드 활용  
저장소의 reproductions/charton2024_gcd.py 파일이 LLM 가이드에 따라 논문을 실제로 재현하는 코드입니다. 이 코드를 실행하거나, 필요하다면 내용을 참고하여 직접 코드를 재작성할 수 있습니다.
- 위치: reproductions/charton2024_gcd.py
- 이 파일은 Charton GCD 모델의 sieve 현상(특정 divisor에 대해서만 학습이 잘 되는 현상)을 실험적으로 검증하는 코드를 포함합니다.

3. 필요한 라이브러리  
- mi-copilot (LLM 어시스턴트 및 reproductions 코드)
- 추가적으로 numpy, pytorch 등 일반적 머신러닝/딥러닝 실험에 필요한 패키지가 필요할 수 있으나, 핵심 코드는 위 reproductions 파일에 포함되어 있습니다.

4. 검증 절차(논문 기준)
- GCD 모델을 base B (예: base 1000=2³·5³)에서 학습
- 모델이 학습한 divisor에 대해서만 GCD 값을 맞추고, 비-divisor(예: 3,7,11 등)에 대해선 정확도가 0%인지 실험을 통해 확인

요약: 기존 scratch 코드 작성 없이, mi-copilot을 설치하고 reproductions/charton2024_gcd.py를 그대로 사용하면 논문과 동일한 검증을 실험할 수 있습니다.

출처:
- mi_copilot_repo · README.md (라이브러리 설치법 및 파일 구조)
- mi_copilot_repo · docs\presentation\result-slides-outline.md (검증 절차 및 예시)
- Charton 2024 §3.2 Table 5 (sieve 알고리즘 현상 설명)

---

## Q4

> W_E 의 Fourier 분해를 실제로 보여주는 plot 은 어떻게 그리나요?

### mi-copilot answer

Nanda 2023의 modular addition 회로 분석을 재현하려면 W_E (embedding matrix)의 Fourier 변환을 시각화해야 합니다. 논문에 따르면 아래와 같이 진행하면 됩니다:

1. W_E (embedding matrix)를 불러옵니다.
2. 각 row(embedding vector)에 대해 FFT (Fast Fourier Transform)를 계산합니다.
3. 각 주파수별 파워(power)를 계산: power = (|fft(W_E)|²).sum(axis=1)
4. 이를 plot 하면, W_E의 Fourier basis sparsity (몇몇 주파수에만 파워가 몰림)를 시각적으로 확인할 수 있습니다.

논문에서는 예시로 Top-5 주파수만 남기고 ablate 했을 때도 모델 성능이 잘 유지된다고 밝힘 (Figure 4 참고).

즉, 다음과 같은 단계를 코드로 옮길 수 있습니다:
```python
import numpy as np
import matplotlib.pyplot as plt

# W_E: [vocab, d_model] shape의 numpy 배열
fft_w_e = np.fft.fft(W_E, axis=1)
power = np.abs(fft_w_e) ** 2
total_power = power.sum(axis=0)  # 각 frequency마다 합침

plt.plot(total_power)
plt.xlabel("Fourier frequency")
plt.ylabel("Power")
plt.title("Fourier Power Spectrum of W_E")
plt.show()
```
이렇게 하면 논문에서 확인한 Fourier sparsity 현상을 직접 볼 수 있습니다.

Source: Nanda 2023 §3.1, Figure 4, presentation/result-slides-outline.md

---

## Q5

> frequency ablation 으로 인과적 검증을 한다는데, 구체적으로 어떻게 하는 건가요?

### mi-copilot answer

Frequency ablation을 통한 인과적 검증이란, 모델의 출력(logits 또는 임베딩 등)을 주파수(특히 Fourier 변환 기준)별로 분해한 뒤, 특정 주파수 성분만 남기거나 제거(ablate)하여 성능 변화를 측정하는 방법입니다.

구체적 절차 (Nanda 2023을 예시로):

1. 모델을 (예: (a+b) mod 113 문제) 학습시킵니다.
2. 임베딩 행렬(WE)이나 logit 등을 FFT(이산 푸리에 변환)로 변환하여 각 주파수의 크기를 관찰합니다.
   - 예: power = (|fft(W_E)|²).sum(axis=1)
3. 중요한(top-k) 주파수만 남기고 나머지는 0으로 ablate(제거) 합니다.
   - “We perform a 2D DFT on the logits to write them as a linear combination of waves ... set all terms besides the constant term and ... the five key frequencies to 0.” (Source: Nanda 2023 §4.4)
4. 이 ablated 상태에서 모델의 성능(정확도 혹은 loss)을 측정합니다.
   - 중요한 주파수만 남겨도 정확도 유지 → 해당 주파수가 인과적으로 중요하다는 것.
5. 반대로 주요 주파수만 ablate(제거) 하고 나머지로 성능 측정도 가능
   - “ablating all of the key frequencies reduces performance to worse than chance, while ablating everything but the key frequencies improves test performance.” (Source: Nanda 2023 §4.4)

즉, frequency ablation은 “특정 주파수 성분이 모델 예측에 인과적(contributive)으로 중요한가?”를 실험적으로 검증하는 방법입니다.

Source: Nanda 2023 §3.1, §4.4; mi_copilot_repo docs\presentation\result-slides-outline.md

---
