from __future__ import annotations

from typing import Dict, List, Optional

from .llm import GeminiClient


class DictionaryEnricher:
    def __init__(self, kid_safe: bool = False):
        self.kid_safe = kid_safe

    def enrich(self, base: Dict[str, str], context_rows: Optional[List[Dict[str, str]]] = None, level: str = "A1/A2") -> Dict[str, object]:
        si = (base.get("sinhala") or "").strip()
        en = (base.get("english") or "").strip()
        translit = (base.get("transliteration") or "").strip()
        pos = (base.get("pos") or "").strip()
        ex_si = (base.get("example_si") or "").strip()
        ex_en = (base.get("example_en") or "").strip()

        # Build a compact corpus block
        corpus_lines: List[str] = []
        if context_rows:
            for it in context_rows[:10]:
                _si = (it.get("sinhala") or "").strip()
                _en = (it.get("english") or "").strip()
                if _si or _en:
                    corpus_lines.append(f"- Sinhala: {_si}\n  English: {_en}")
        corpus_block = "\n".join(corpus_lines)

        # Prompt for learner's dictionary entry
        guidance = (
            "Create a learner's dictionary entry for Sinhala learners of English. "
            "Focus on a single English headword (not a sentence). "
            "Use simple English at the specified CEFR level and keep it kid-friendly if requested. "
            "Return STRICT JSON (no markdown) with keys: "
            "{\"english\", \"sinhala\", \"transliteration\", \"pos\", \"definition_en\", \"examples_en\": [2], \"explanation_si\", \"examples_si\": [2], \"synonyms_en\": [<=5], \"notes_si\"}. "
            "Make examples short and clear. If inputs are sentences, extract the headword."
        )

        pieces: List[str] = [guidance, f"\nLevel: {level}"]
        if corpus_block:
            pieces.append("\nUse this parallel corpus to stay consistent:\n" + corpus_block)
        pieces.append("\nInput fields (may be noisy):\n" + str({
            "sinhala": si,
            "english": en,
            "transliteration": translit,
            "pos": pos,
            "example_si": ex_si,
            "example_en": ex_en,
        }))
        prompt = "\n".join(pieces)

        gem = GeminiClient()
        if getattr(gem, "_kid_guidelines", None) and self.kid_safe:
            prompt = gem._kid_guidelines() + prompt  # reuse shared kid guidelines

        if getattr(gem, "_provider", "v2") == "v2":
            resp = gem._client.models.generate_content(model=gem.model_name, contents=prompt)
            raw = (getattr(resp, "text", "") or "").strip()
        else:
            resp = gem._client.generate_content(prompt)
            raw = (getattr(resp, "text", "") or "").strip()

        import json
        try:
            data = json.loads(raw)
            # Ensure required keys exist
            cleaned = {
                "english": str(data.get("english", en)).strip(),
                "sinhala": str(data.get("sinhala", si)).strip(),
                "transliteration": str(data.get("transliteration", translit)).strip(),
                "pos": str(data.get("pos", pos)).strip(),
                "definition_en": str(data.get("definition_en", "")).strip(),
                "examples_en": list(data.get("examples_en", []))[:2],
                "explanation_si": str(data.get("explanation_si", "")).strip(),
                "examples_si": list(data.get("examples_si", []))[:2],
                "synonyms_en": list(data.get("synonyms_en", []))[:5],
                "notes_si": str(data.get("notes_si", "")).strip(),
            }
            # Coerce list items to strings
            cleaned["examples_en"] = [str(x).strip() for x in cleaned["examples_en"] if str(x).strip()]
            cleaned["examples_si"] = [str(x).strip() for x in cleaned["examples_si"] if str(x).strip()]
            cleaned["synonyms_en"] = [str(x).strip() for x in cleaned["synonyms_en"] if str(x).strip()]
            return cleaned
        except Exception:
            # Fallback minimal entry
            return {
                "english": en,
                "sinhala": si,
                "transliteration": translit,
                "pos": pos,
                "definition_en": raw,
                "examples_en": [ex_en] if ex_en else [],
                "explanation_si": "",
                "examples_si": [ex_si] if ex_si else [],
                "synonyms_en": [],
                "notes_si": "",
            }
