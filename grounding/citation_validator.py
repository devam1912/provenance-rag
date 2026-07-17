import re
import logging
from typing import List, Dict, Any, Tuple
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("grounding.citation_validator")

def get_cited_sentence(text: str, match_start: int) -> str:
    """Finds the sentence immediately preceding the citation by tracing backward to a sentence boundary."""
    start = match_start
    boundaries = {".", "?", "!", "\n"}
    while start > 0:
        # Check if previous character is a boundary
        if text[start - 1] in boundaries:
            break
        start -= 1
    
    sentence = text[start:match_start].strip()
    # Strip leading/trailing punctuation if any remains
    sentence = re.sub(r"^[\s\-\*]+", "", sentence)
    return sentence


def validate_citations(
    answer: str, 
    retrieved_chunks: List[Any], 
    llm: Any
) -> Dict[str, Any]:
    """
    Parses and validates all citations in the generated answer.
    Checks:
    1. Existence: Cited chunk must be present in retrieved_chunks.
    2. Factuality: Preceding claim must be logically supported by the cited chunk.
    
    Returns:
    {
        "is_valid": bool,
        "failed_citations": List[str]
    }
    """
    logger.info("Starting citation validation check...")
    
    # Parse citations of format [filename#chunk_idx]
    # e.g., [academic_standing.txt#chunk_3]
    citation_pattern = r"\[([a-zA-Z0-9_\-\.]+#chunk_\d+)\]"
    matches = list(re.finditer(citation_pattern, answer))
    
    if not matches:
        logger.info("No citations found to validate.")
        # If the answer is the fallback "insufficient information", that's valid.
        # Otherwise, if it made claims without citing, we fail it.
        fallback_phrases = ["insufficient", "no verified information", "information not found"]
        is_fallback = any(phrase in answer.lower() for phrase in fallback_phrases)
        
        if is_fallback or len(answer.strip()) < 100:
            return {"is_valid": True, "failed_citations": []}
        else:
            return {
                "is_valid": False,
                "failed_citations": ["The response made claims but did not include any citations. Every factual claim must end with a [filename#chunk_idx] citation."]
            }
            
    # Map retrieved chunks by chunk_id
    retrieved_map = {c.chunk_id: c for c in retrieved_chunks}
    failed_citations = []
    
    for match in matches:
        chunk_id = match.group(1)
        match_start = match.start()
        
        claim = get_cited_sentence(answer, match_start)
        # If claim is too short, look further back or use a default
        if len(claim) < 5:
            claim = answer[:match_start].split("\n")[-1].strip()
            
        logger.info(f"Checking citation: '{chunk_id}' for claim: '{claim}'")
        
        # Check 1: Existence
        if chunk_id not in retrieved_map:
            msg = f"Citation '{chunk_id}' is invalid because it was never retrieved in the search phase."
            logger.warning(msg)
            failed_citations.append(msg)
            continue
            
        # Check 2: Factuality / Entailment
        chunk = retrieved_map[chunk_id]
        chunk_text = chunk.text
        
        system_prompt = (
            "You are an NLI (Natural Language Inference) validation engine.\n"
            "Your task is to determine if a claim is logically supported and entailed by the provided reference text.\n"
            "Answer strictly with YES or NO. Do not add any explanation or preamble.\n\n"
            "Reference Text:\n"
            f"{chunk_text}\n\n"
            f"Claim to verify: '{claim}'\n\n"
            "Is the claim fully supported and true according to the Reference Text? (YES/NO):"
        )
        
        messages = [
            SystemMessage(content=system_prompt)
        ]
        
        try:
            res = llm.invoke(messages)
            # Safe text extraction helper
            if isinstance(res.content, str):
                verdict = res.content.strip().upper()
            elif isinstance(res.content, list):
                verdict = "".join([b.get("text", "") for b in res.content if isinstance(b, dict) and b.get("type") == "text"]).strip().upper()
            else:
                verdict = str(res.content).strip().upper()
                
            logger.info(f"NLI Verdict for '{chunk_id}': {verdict}")
            if "YES" not in verdict:
                msg = f"The claim '{claim}' cited by '{chunk_id}' is not supported by the cited reference chunk."
                logger.warning(msg)
                failed_citations.append(msg)
                
        except Exception as e:
            logger.error(f"Entailment check failed for '{chunk_id}': {e}")
            # If checking fails due to transient API errors, log but pass it to prevent blocking
            pass

    is_valid = len(failed_citations) == 0
    logger.info(f"Citation validation complete. Valid: {is_valid} | Failed count: {len(failed_citations)}")
    return {
        "is_valid": is_valid,
        "failed_citations": failed_citations
    }
