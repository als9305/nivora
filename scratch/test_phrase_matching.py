import re
from rapidfuzz import fuzz

def evaluate_challenge_phrase(transcribed_text: str, challenge_phrase: str):
    """
    Evaluates whether transcribed speech matches the challenge phrase with tolerance
    for Indian English accents, minor phoneme variations, and ASR compounding.
    """
    if not challenge_phrase:
        return {
            "match": "UNVALIDATED",
            "score": 0.0,
            "transcribed_text": transcribed_text or "",
            "challenge_phrase": "",
            "reason": "No challenge phrase specified."
        }
        
    if not transcribed_text or not transcribed_text.strip():
        return {
            "match": "INCONCLUSIVE",
            "score": 0.0,
            "transcribed_text": "",
            "challenge_phrase": challenge_phrase,
            "reason": "No intelligible speech transcribed."
        }

    # Normalize
    clean_target = re.sub(r"[^\w\s]", "", challenge_phrase.lower()).strip()
    clean_spoken = re.sub(r"[^\w\s]", "", transcribed_text.lower()).strip()

    target_words = clean_target.split()
    spoken_words = clean_spoken.split()

    if not target_words or not spoken_words:
        return {
            "match": "INCONCLUSIVE",
            "score": 0.0,
            "transcribed_text": transcribed_text,
            "challenge_phrase": challenge_phrase,
            "reason": "Empty token sequence."
        }

    # Word-level fuzzy matching
    matched_words = 0
    word_scores = []
    for tw in target_words:
        best_w_score = max(fuzz.ratio(tw, sw) for sw in spoken_words)
        word_scores.append(best_w_score)
        # 70% allows for minor phonetic mishearings (e.g. 'mangoe' vs 'mango', 'colud' vs 'cloud', 'flowr' vs 'flower')
        if best_w_score >= 70:
            matched_words += 1

    # Full phrase metrics
    set_ratio = fuzz.token_set_ratio(clean_target, clean_spoken)
    sort_ratio = fuzz.token_sort_ratio(clean_target, clean_spoken)
    avg_word_score = sum(word_scores) / len(word_scores)

    combined_score = round(float(0.5 * avg_word_score + 0.5 * set_ratio) / 100.0, 4)

    # Decision criteria:
    # All target words should ideally be spoken. We require at least 2/3 words matched with high confidence,
    # or all 3 words matched with moderate confidence, and combined score >= 0.70.
    if (matched_words == len(target_words) and combined_score >= 0.68) or (matched_words >= 2 and combined_score >= 0.75):
        verdict = "MATCHED"
        reason = f"Challenge phrase confirmed ({matched_words}/{len(target_words)} words matched, score: {combined_score:.2f})."
    elif combined_score >= 0.45 or matched_words >= 1:
        verdict = "INCONCLUSIVE"
        reason = f"Partial phrase match detected ({matched_words}/{len(target_words)} words, score: {combined_score:.2f})."
    else:
        verdict = "MISMATCH"
        reason = f"Spoken words do not match challenge phrase (score: {combined_score:.2f})."

    return {
        "match": verdict,
        "score": combined_score,
        "transcribed_text": transcribed_text,
        "challenge_phrase": challenge_phrase,
        "matched_words": f"{matched_words}/{len(target_words)}",
        "reason": reason
    }

# Test cases
cases = [
    ("Apple river green", "Apple river green"),
    ("apple river green", "Apple River Green"),
    ("Apple river green", "appl rivar grin"), # accent variation
    ("Ocean window yellow", "ocean window yellow"),
    ("Ocean window yellow", "the ocean and window is yellow"),
    ("Ocean window yellow", "sunny tiger water"), # completely wrong challenge
    ("Ocean window yellow", "hello test one two three"), # generic test speech
    ("Ocean window yellow", ""), # silence / empty
]

for ch, sp in cases:
    res = evaluate_challenge_phrase(sp, ch)
    print(f"Target: '{ch}' | Spoken: '{sp}' => {res['match']} (score: {res['score']}) - {res['reason']}")
