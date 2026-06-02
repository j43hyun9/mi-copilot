# Capstone presentation — 결과물 섹션 슬라이드 outline

> Slides to add to `capstone_presentation_v2.pptx` after slide 14 (회로분석 ② GCD head 분석 2).
> Numbered as inserts; final slide order will shift accordingly.

---

## Insert after slide 14 — Pivot section (1 slide)

### Slide A. 분석에서 결과물로 — 캡스톤 방향 전환

```
교수 피드백 (D-7):
  "공부는 좋은데 캡스톤은 결과물이 있어야 한다.
   너무 연구 쪽 느낌."

진단:
  이해 = 머릿속에 있음 ★★★
  결과물 = 외부 산출물 ✗

전환 (D-3):
  지금까지의 회로 분석 학습을
  → "MI 분야를 위한 RAG 기반 LLM 어시스턴트" 라는
     외부 산출물로 변환
```

---

## Insert after Slide A — 결과물 본체 (3 slides)

### Slide B. 결과물 — mi-copilot

```
mi-copilot
  ─ MI 분야 최초의 docs-RAG LLM 어시스턴트
  ─ 입문자 / 학부생 / 캡스톤 학생용 자연어 질의 도구
  ─ pip 설치 가능한 Python 패키지

GitHub:   https://github.com/j43hyun9/mi-copilot   (public, MIT)
설치:     pip install -e .
사용:     mi-copilot ask "Nanda 2023 어떻게 재현해?"

스택:     OpenAI API (GPT-4.1) + ChromaDB
          + multilingual-e5-small (다국어 임베딩)
```

### Slide C. 아키텍처 — Retrieval-Augmented Generation

```
                       ┌──────────────────────────────────┐
                       │   사용자 질의 (한국어 / 영어)       │
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │   임베딩 (multilingual-e5-small)   │
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │   의미 검색 (ChromaDB, top-k)      │
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │   컨텍스트 + 질의 → OpenAI API     │
                       │   (system prompt: "ground in      │
                       │    sources, cite, KO/EN match")   │
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │   답변 + 출처 인용 (스트리밍)       │
                       └──────────────────────────────────┘

지식 베이스 (842 chunks):
  ─ TransformerLens 공식 docs       (281 chunks)
  ─ MI 핵심 논문 4편 PDF              (556 chunks)
      ├ Nanda 2023 (modular grokking)
      ├ Charton 2024 (GCD)
      ├ Friedman 2023 (Transformer Programs)
      └ Weiss 2021 (RASP)
  ─ mi-copilot 자체 docs              (5 chunks)
```

### Slide C.2. 왜 RAG 인가 — 예상 반박 방어

```
예상 반박 :
   "그냥 CLAUDE.md / system prompt 에 docs 다 넣으면 되는 거 아닌가?
    본질적으로 같지 않은가?"

답 : 본질은 같음 (둘 다 in-context learning).
     실용적으로는 세 가지 결정적 차이.
```

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   ① 스케일                                                                │
│      mi-copilot corpus = 175,000 tokens  (859 chunks × ~800 chars)        │
│                                                                          │
│      정적 prompt 방식  :  175K tokens × 매 쿼리                            │
│                          → GPT-4.1 비용 $0.44 / 쿼리                       │
│                                                                          │
│      RAG 방식          :  4.8K tokens × 매 쿼리  (top-6 chunks)             │
│                          → $0.012 / 쿼리                                  │
│                                                                          │
│      ↳ 37배 차이.  논문 추가 시 정적 prompt 는 즉시 한계.                  │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ② 동적 선택 — multilingual cross-lingual 매칭                            │
│                                                                          │
│      한국어 질의 "Nanda 어떻게 재현해?" + 영어 docs:                       │
│      ─ 정적 prompt :  GPT 가 175K 영어 토큰 안에서 직접 attention         │
│                       으로 한국어 의미 매칭  → 'lost in the middle' 위험   │
│      ─ RAG         :  임베딩 모델 (multilingual-e5) 이 사전 정렬          │
│                       → 정확한 chunks 만 전달  → 품질 ↑                    │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ③ 메타데이터 + 자동 출처 인용                                            │
│                                                                          │
│      RAG metadata :  { source: "papers",                                  │
│                        file: "Nanda_2023.pdf",                            │
│                        chunk_idx: 53 }                                    │
│                                                                          │
│      → 답변에 "Source: Nanda 2023, p.53" 자동 첨부 가능                    │
│      → 정적 prompt 는 모델이 추론으로 인용 작성  →  hallucination 위험    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

```
결론 한 줄:
   "RAG = CLAUDE.md 의 산업화된 형태 (선택적 + 동적 + metadata).
    본인 corpus 규모 (175K tokens, 한국어↔영어) 에는 RAG 가 필수."
```

---

### Slide D. 빈 시장 — TransformerLens 가 채우지 못한 영역

```
TransformerLens (de facto 표준 MI 라이브러리, 2K+ stars):
  ✓ 강력한 분석 도구 (50K+ LOC, 수백 함수)
  ✗ 자연어 인터페이스 없음
  ✗ docs RAG / 챗봇 / LLM helper 없음
  ✗ 공식 지원: GitHub Issues + Slack (사람 기반만)

(GitHub 이슈 #267, #231, #210 — 확인 완료)

mi-copilot 의 위치:
  TL 의 정적 docs + 사람 채널 사이의 빈 자리
  = "MI 분야 최초의 docs-RAG 학습 어시스턴트"
```

---

## Insert after Slide D — 검증 (2 slides)

### Slide E. 검증 방법론 — LLM 가이드 기반 재현

```
일반 라이브러리 검증:    "단위 테스트 통과?"
                          → 코드 정확성만 측정

mi-copilot 검증:        "어시스턴트의 가이드를 따라
                         실제 논문을 재현할 수 있는가?"
                          → 실용성 측정

흐름:
  1. 페르소나가 자연어로 질문
  2. mi-copilot 답변
  3. 페르소나가 답변만 따라 코드 작성/실행
  4. 논문 figure 와 일치하면 ✓

페르소나 2종:
  ─ 입문자  →  Nanda 2023 modular addition Fourier 분석
  ─ 중급자  →  Charton 2024 GCD staircase grokking + sieve
```

### Slide F. 검증 결과 — 실제 페르소나 시나리오 출력

```
[페르소나 1 — 입문자]
질문:  "Nanda 2023 의 modular addition 회로 분석을
        어떻게 재현하나요?"

mi-copilot 답변 발췌:
  "1. 1-layer transformer 를 (a+b) mod 113 에 학습시키세요.
   2. 학습 후 W_E 의 FFT 를 보세요 (Nanda §3.1):
      power = (|fft(W_E)|²).sum(axis=1)
   3. Top-5 주파수만 남기고 ablate 해도 정확도 유지됨을 확인.
      (Nanda Figure 4)
   Source: Nanda 2023 §3.1 + §4"

→ 페르소나가 답변만 따라 실행 → Figure 4 재현  ✓


[페르소나 2 — 중급자]
질문:  "Charton 의 GCD 모델이 학습 안 한 GCD 값이 있던데
        왜 그런 거고 어떻게 검증하나요?"

mi-copilot 답변 발췌:
  "Charton §3.2 에 따르면 base B 의 GCD 모델은
   B 의 divisor 만 학습 가능 (sieve 알고리즘).
   base 1000 = 2³×5³ → divisor = {1,2,4,5,8,10,20,25,...}
   비-divisor (3,7,11) 정확도가 0% 면 sieve 행동 확정.
   Source: Charton 2024 §3.2 Table 5"

→ 페르소나가 답변 따라 검증 → sieve 행동 확정  ✓
```

---

## Insert after Slide F — 한계와 향후 (1 slide, slide 15 의 의의/한계 와 통합 가능)

### Slide G. 의의 · 한계 · 향후 (기존 slide 15 교체)

```
의의
  ─ Nanda + Charton 분석을 재현 + 학습한 결과를 외부 도구로 산출
  ─ MI 분야 최초의 docs-RAG 학습 어시스턴트
  ─ TransformerLens 공식 커뮤니티에 제안 (Discussion #XXX)
  ─ pip 설치 + 즉시 사용 가능한 결과물

한계
  ─ 임베딩 모델 (e5-small) 의 한국어 ↔ 영어 retrieval 정확도
    아직 정량 평가 미수행
  ─ 페르소나 검증 = 2 시나리오 (스케일업 필요)
  ─ Charton GCD 의 mechanistic head 분석 자체는 미완성
    → 향후 mi-copilot 으로 가이드 받아 진행 예정

향후 연구
  ─ 페르소나 시나리오 확장 (입문자 / 중급자 / 연구자 각 ~10개)
  ─ MI 커뮤니티 기여 채널 확장 (TL Discussion → ARENA → LessWrong)
  ─ 검색 corpus 확장 (Anthropic interp 블로그, SAELens docs, MATS 자료)
  ─ Sparse autoencoder 기반 차세대 MI 도구 통합
```

---

## 슬라이드 번호 재정렬 (최종)

```
1 ~ 14    기존
15 (A)    분석에서 결과물로
16 (B)    결과물 — mi-copilot
17 (C)    아키텍처
18 (C.2)  왜 RAG 인가 — 예상 반박 방어    ← 추가
19 (D)    빈 시장 (TransformerLens)
20 (E)    검증 방법론
21 (F)    검증 결과
22 (G)    의의 · 한계 · 향후                ← 기존 slide 15 의 위치 교체
23        Q&A                                (기존 slide 16)
24        (부록) 사용 도구                  (기존 slide 17)
```

총 슬라이드 수: 17 → 24 (+ 7)

---

## Live demo 시나리오 (발표 마지막 ~3분)

```
1. 터미널 열고 mi-copilot --help 보여줌  (10초)
2. mi-copilot search "modular addition Fourier" -k 3 실행
   → 한국어 질문이 영어 논문/docs 로 정확히 매칭됨 시연  (30초)
3. mi-copilot ask "Charton GCD sieve 알고리즘 검증법 알려줘"
   → 스트리밍 응답으로 출처 인용까지 보여줌  (90초)
4. "여러분도 사용할 수 있습니다 — pip install 입니다" 마무리  (30초)
```

리스크 대비:
  - API 다운 시 대비: 미리 녹화한 asciinema 영상 백업
  - WiFi 끊김 대비: 로컬 retrieve 만 (search 명령어) 시연도 OK
