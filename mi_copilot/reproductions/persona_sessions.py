"""Persona validation sessions — run real queries through mi-copilot.

This script captures the questions that the two validation personas
(beginner / intermediate) would ask, and runs them end-to-end through
the assistant. The output is saved as Markdown transcripts that serve as
evidence in the capstone presentation.

Run:
    py -m mi_copilot.reproductions.persona_sessions

Requires:
    - ANTHROPIC_API_KEY set in .env
    - Ingest run at least once (`mi-copilot ingest`)
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 on Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


PERSONAS: dict[str, list[str]] = {
    "beginner": [
        # The beginner doesn't know MI vocabulary yet — uses everyday language.
        "신경망이 '갑자기 일반화' 한다는 grokking 현상이 뭔가요? 그리고 그게 왜 흥미로운가요?",
        "Nanda 2023 라는 논문이 modular addition 으로 grokking 회로를 분석했다고 들었는데, 모델이 어떤 알고리즘을 학습한 거죠?",
        "그 알고리즘을 처음부터 검증하려면 어떤 코드를 짜야 하나요? 어떤 라이브러리가 필요한지도 알려주세요.",
        "W_E 의 Fourier 분해를 실제로 보여주는 plot 은 어떻게 그리나요?",
        "frequency ablation 으로 인과적 검증을 한다는데, 구체적으로 어떻게 하는 건가요?",
    ],
    "intermediate": [
        # The intermediate persona has MI background and asks deeper, code-specific questions.
        "TransformerLens 의 HookedTransformer 와 비교했을 때, Nanda 의 raw 1-layer transformer 구현을 그대로 쓰는 게 더 나은 case 는 어떤 case 인가요?",
        "Charton 2024 의 GCD 모델이 base 의 divisor 만 학습한다는 sieve 행동을 mechanistic level 에서 검증하려면 어떤 분석을 추가해야 하나요?",
        "per-head Fourier 분해를 GCD 모델에 적용했을 때 어떤 결과를 기대할 수 있나요? Nanda 의 modular addition 과 어떻게 다를 것 같나요?",
        "Charton 의 staircase grokking 을 progress measure 로 정량화하려면 Nanda 의 restricted/excluded loss 를 어떻게 일반화해야 하나요?",
        "이 두 분석 (Nanda Fourier + Charton sieve) 을 통합한 universality 가설을 세운다면 어떤 형태가 되어야 하나요?",
    ],
}


def run_session(persona: str, queries: list[str], out_path: Path) -> None:
    from mi_copilot.llm_assistant.chat import answer

    transcript: list[str] = []
    transcript.append(f"# mi-copilot persona session — `{persona}`")
    transcript.append("")
    transcript.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    transcript.append("")

    for i, q in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"[{persona}] Q{i}: {q}")
        print(f"{'='*80}\n")
        try:
            a = answer(q, k=6, stream=True)
        except Exception as e:
            a = f"[ERROR: {e}]"
            print(a)
        transcript.append(f"## Q{i}")
        transcript.append("")
        transcript.append(f"> {q}")
        transcript.append("")
        transcript.append("### mi-copilot answer")
        transcript.append("")
        transcript.append(a)
        transcript.append("")
        transcript.append("---")
        transcript.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(transcript), encoding="utf-8")
    print(f"\n[saved] {out_path}")


def main():
    out_root = Path(__file__).resolve().parents[2] / "docs" / "personas"
    for persona, queries in PERSONAS.items():
        out_path = out_root / f"{persona}-session.md"
        run_session(persona, queries, out_path)


if __name__ == "__main__":
    main()
