"""
LLM-based topic curator for continual topic model.

Two-stage curation:
  Stage 1 — Prune & Refine: For each global topic, decide RETAIN or REMOVE
            based on evidence from local topics.
  Stage 2 — Detect Novel: For each local topic, decide NOVEL or COVERED
            based on similarity to retained global topics.

Supports:
  - "gemini" provider (Google Generative AI)
  - "none" provider (similarity-based fallback, no LLM needed)
"""

import json
import time
import logging
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.topic_utils import Topic, cosine_similarity_matrix, find_nearest_topics

# ─── Prompt logger (writes to outputs/prompts.log) ───────────────────────────

_prompt_logger: Optional[logging.Logger] = None

def _get_prompt_logger(log_path: str = "outputs/prompts.log") -> logging.Logger:
    global _prompt_logger
    if _prompt_logger is None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("prompt_logger")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        _prompt_logger = logger
    return _prompt_logger


def _log_prompt(stage: str, topic_id: int, prompt: str,
                response: str, log_path: str = "outputs/prompts.log"):
    logger = _get_prompt_logger(log_path)
    sep = "═" * 80
    logger.debug(
        f"\n{sep}\n"
        f"[{stage}] topic_id={topic_id}  ts={time.strftime('%H:%M:%S')}\n"
        f"{'─'*40} PROMPT {'─'*40}\n{prompt}\n"
        f"{'─'*40} RESPONSE {'─'*38}\n{response}\n"
        f"{sep}"
    )


@dataclass
class CurationDecision:
    """Decision for a single topic."""
    topic_id: int
    action: str          # "RETAIN", "REMOVE", "NOVEL", "COVERED"
    refined_words: Optional[list[str]] = None
    reason: str = ""
    confidence: float = 0.0


@dataclass
class RetainPrior:
    """Soft retain prior for a global topic."""
    topic_id: int
    probability: float
    reason: str = ""


@dataclass
class NoveltyPrior:
    """Soft novelty prior for a local topic."""
    topic_id: int
    probability: float
    reason: str = ""


# ─── LLM Interface ───────────────────────────────────────────────────────────

class LLMCurator:
    """LLM-based topic curator."""

    def __init__(self, config: dict, log_path: str = "outputs/prompts.log"):
        self.provider = config.get("provider", "none")
        self.model_name = config.get("model", "gemini-2.0-flash")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 1024)
        self.rate_limit_delay = config.get("rate_limit_delay", 0.5)
        self.max_retries = config.get("max_retries", 5)
        self.fallback_config = config.get("fallback", {})
        self.log_path = log_path
        self.call_count = 0
        self.print_frequency = config.get("print_frequency", 10)

        self._client = None
        if self.provider == "gemini":
            self._init_gemini()

    def _init_gemini(self):
        """Initialize Google Generative AI client."""
        try:
            import os
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                self._client = genai.Client(api_key=api_key)
            else:
                self._client = genai.Client()  # falls back to ADC / GOOGLE_APPLICATION_CREDENTIALS
            print(f"  LLM Curator: Using Gemini ({self.model_name})")
        except Exception as e:
            print(f"  Warning: Failed to init Gemini: {e}. Falling back to similarity-based.")
            self.provider = "none"

    def _call_llm(self, prompt: str, stage: str = "", topic_id: int = -1) -> str:
        """Call the LLM and return text response. Logs prompt and response."""
        if self.provider == "gemini":
            from google import genai
            for attempt in range(self.max_retries):
                try:
                    response = self._client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={
                            "temperature": self.temperature,
                            "max_output_tokens": self.max_tokens,
                        },
                    )
                    text = response.text
                    _log_prompt(stage, topic_id, prompt, text, self.log_path)
                    
                    self.call_count += 1
                    if self.call_count % self.print_frequency == 0:
                        print(f"\n    --- Sampled LLM Call ({stage} topic {topic_id}) ---")
                        print(f"    Prompt snippet: {prompt.strip()[:150].replace(chr(10), ' ')}...")
                        print(f"    Response: {text.strip().replace(chr(10), ' ')}")
                        print(f"    --------------------------------------------------\n")
                        
                    time.sleep(self.rate_limit_delay)
                    return text
                except Exception as e:
                    error_msg = str(e)
                    print(f"    LLM call failed (attempt {attempt+1}/{self.max_retries}): {e}")
                    if "429" in error_msg or "ResourceExhausted" in error_msg or "quota" in error_msg.lower():
                        sleep_time = (2 ** attempt) * 15 + self.rate_limit_delay
                        print(f"    Rate limit reached. Sleeping for {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        time.sleep(2 ** attempt)
            _log_prompt(stage, topic_id, prompt, "[FAILED]", self.log_path)
            return ""
        # provider == "none": log the prompt with a fallback marker
        _log_prompt(stage, topic_id, prompt, "[SKIPPED — fallback mode]", self.log_path)
        return ""

    # ─── Stage 1: Prune & Refine ─────────────────────────────────────────────

    # --- Soft priors: Retain / Novelty probabilities ------------------------

    def score_retain_priors(
        self,
        global_topics: list[Topic],
        local_topics: list[Topic],
        top_k: int = 5,
    ) -> list[RetainPrior]:
        """Return soft survival priors p_k^LLM for global topics."""
        if not global_topics:
            return []

        if self.provider == "none":
            return self._retain_prior_fallback(global_topics, local_topics)

        nearest = find_nearest_topics(global_topics, local_topics, top_k=top_k)
        priors = []
        for g_idx, g_topic in enumerate(global_topics):
            neighbor_strs = [
                f"  - {local_topics[l_idx].to_string()} (similarity={sim:.3f})"
                for l_idx, sim in nearest[g_idx]
            ]
            prompt = self._build_retain_prior_prompt(g_topic, neighbor_strs)
            response = self._call_llm(prompt, stage="Soft-Retain", topic_id=g_topic.id)
            priors.append(self._parse_retain_prior_response(response, g_topic.id))
        return priors

    def score_novelty_priors(
        self,
        local_topics: list[Topic],
        global_topics: list[Topic],
        top_k: int = 5,
    ) -> list[NoveltyPrior]:
        """Return soft novelty priors q_j^LLM for local topics."""
        if not local_topics:
            return []

        if self.provider == "none":
            return self._novelty_prior_fallback(local_topics, global_topics)

        if not global_topics:
            return [
                NoveltyPrior(topic_id=l.id, probability=1.0,
                             reason="No global topics to compare against")
                for l in local_topics
            ]

        nearest = find_nearest_topics(local_topics, global_topics, top_k=top_k)
        priors = []
        for l_idx, l_topic in enumerate(local_topics):
            neighbor_strs = [
                f"  - {global_topics[g_idx].to_string()} (similarity={sim:.3f})"
                for g_idx, sim in nearest[l_idx]
            ]
            prompt = self._build_novelty_prior_prompt(l_topic, neighbor_strs)
            response = self._call_llm(prompt, stage="Soft-Novelty", topic_id=l_topic.id)
            priors.append(self._parse_novelty_prior_response(response, l_topic.id))
        return priors

    def _build_retain_prior_prompt(self, global_topic: Topic,
                                   neighbor_strs: list[str]) -> str:
        return f"""You are a topic modeling expert analyzing research topic evolution.

GLOBAL TOPIC (from previous memory):
  {global_topic.to_string()}

NEARBY LOCAL TOPICS (from current data):
{chr(10).join(neighbor_strs)}

TASK: Estimate the probability that the global topic is still semantically alive
in the current data. Return a soft probability, not a hard decision.

Use 0.0 for obsolete/unsupported, 1.0 for clearly active, and intermediate
values for partial survival or drift.

Respond ONLY with a JSON object:
{{"retain_probability": 0.0-1.0, "reason": "brief explanation"}}
"""

    def _build_novelty_prior_prompt(self, local_topic: Topic,
                                    neighbor_strs: list[str]) -> str:
        return f"""You are a topic modeling expert analyzing research topic evolution.

LOCAL TOPIC (from current data):
  {local_topic.to_string()}

NEARBY GLOBAL TOPICS (existing memory):
{chr(10).join(neighbor_strs)}

TASK: Estimate the probability that the local topic contains genuinely new
semantic content not already represented by the global memory. Return a soft
probability, not a hard decision.

Use 0.0 for fully covered, 1.0 for clearly novel, and intermediate values for
partial novelty or semantic drift.

Respond ONLY with a JSON object:
{{"novelty_probability": 0.0-1.0, "reason": "brief explanation"}}
"""

    def _parse_retain_prior_response(self, response: str, topic_id: int) -> RetainPrior:
        try:
            data = _extract_json(response)
            prob = _clip_probability(data.get("retain_probability", 0.5))
            return RetainPrior(
                topic_id=topic_id,
                probability=prob,
                reason=data.get("reason", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        return RetainPrior(topic_id=topic_id, probability=0.5,
                           reason="LLM parse failure")

    def _parse_novelty_prior_response(self, response: str, topic_id: int) -> NoveltyPrior:
        try:
            data = _extract_json(response)
            prob = _clip_probability(data.get("novelty_probability", 0.5))
            return NoveltyPrior(
                topic_id=topic_id,
                probability=prob,
                reason=data.get("reason", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        return NoveltyPrior(topic_id=topic_id, probability=0.5,
                            reason="LLM parse failure")

    def _retain_prior_fallback(
        self,
        global_topics: list[Topic],
        local_topics: list[Topic],
    ) -> list[RetainPrior]:
        """Similarity-based soft retain priors."""
        if not local_topics:
            return [
                RetainPrior(topic_id=g.id, probability=0.0,
                            reason="No local topics to support")
                for g in global_topics
            ]

        threshold = self.fallback_config.get("prune_threshold", 0.3)
        scale = self.fallback_config.get("soft_scale", 12.0)
        sim_matrix = cosine_similarity_matrix(global_topics, local_topics)
        priors = []
        for g_idx, g_topic in enumerate(global_topics):
            max_sim = float(sim_matrix[g_idx].max())
            prob = _sigmoid(scale * (max_sim - threshold))
            priors.append(RetainPrior(
                topic_id=g_topic.id,
                probability=prob,
                reason=f"fallback max_sim={max_sim:.3f}",
            ))
        return priors

    def _novelty_prior_fallback(
        self,
        local_topics: list[Topic],
        global_topics: list[Topic],
    ) -> list[NoveltyPrior]:
        """Similarity-based soft novelty priors."""
        if not global_topics:
            return [
                NoveltyPrior(topic_id=l.id, probability=1.0,
                             reason="No global topics to compare")
                for l in local_topics
            ]

        threshold = self.fallback_config.get("novel_threshold", 0.5)
        scale = self.fallback_config.get("soft_scale", 12.0)
        sim_matrix = cosine_similarity_matrix(local_topics, global_topics)
        priors = []
        for l_idx, l_topic in enumerate(local_topics):
            max_sim = float(sim_matrix[l_idx].max())
            prob = _sigmoid(scale * (threshold - max_sim))
            priors.append(NoveltyPrior(
                topic_id=l_topic.id,
                probability=prob,
                reason=f"fallback max_sim={max_sim:.3f}",
            ))
        return priors

    def stage1_prune_and_refine(
        self,
        global_topics: list[Topic],
        local_topics: list[Topic],
        top_k: int = 5,
    ) -> list[CurationDecision]:
        """For each global topic, decide RETAIN or REMOVE.

        A global topic is RETAINED if it still has support in the current
        local topics, otherwise REMOVED (outdated/redundant).
        """
        if not global_topics:
            return []

        if self.provider == "none":
            return self._stage1_fallback(global_topics, local_topics)

        # Find nearest local topics for each global topic
        nearest = find_nearest_topics(global_topics, local_topics, top_k=top_k)

        decisions = []
        for g_idx, g_topic in enumerate(global_topics):
            neighbors = nearest[g_idx]
            neighbor_strs = []
            for l_idx, sim in neighbors:
                neighbor_strs.append(
                    f"  - {local_topics[l_idx].to_string()}"
                )

            prompt = self._build_stage1_prompt(g_topic, neighbor_strs)
            response = self._call_llm(prompt, stage="Stage1-Prune", topic_id=g_topic.id)
            decision = self._parse_stage1_response(response, g_topic.id)
            decisions.append(decision)

        return decisions

    def _build_stage1_prompt(self, global_topic: Topic, neighbor_strs: list[str]) -> str:
        return f"""You are a topic modeling expert analyzing research topic evolution.

GLOBAL TOPIC (from previous time period):
  {global_topic.to_string()}

TOP-K MOST SIMILAR LOCAL TOPICS (from current data):
{chr(10).join(neighbor_strs)}

TASK: Decide whether this global topic should be RETAINED or REMOVED.

DECISION CRITERIA:
- REMOVE if none of the local topics are semantically related to this global topic,
  meaning the theme has become outdated, irrelevant, or absorbed into other topics.
- RETAIN if at least one local topic shares a clearly related research theme,
  meaning the topic is still active in the current data.

If RETAIN, optionally refine the topic's representative words based on current local usage.

Respond ONLY with a JSON object:
{{"action": "RETAIN" or "REMOVE", "refined_words": ["word1", "word2", ...] or null, "reason": "brief explanation of decision"}}
"""

    def _parse_stage1_response(self, response: str, topic_id: int) -> CurationDecision:
        try:
            data = _extract_json(response)
            return CurationDecision(
                topic_id=topic_id,
                action=data.get("action", "RETAIN").upper(),
                refined_words=data.get("refined_words"),
                reason=data.get("reason", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        # Default to RETAIN on parse failure
        return CurationDecision(topic_id=topic_id, action="RETAIN", reason="LLM parse failure")

    def _stage1_fallback(
        self,
        global_topics: list[Topic],
        local_topics: list[Topic],
    ) -> list[CurationDecision]:
        """Similarity-based fallback for Stage 1.

        REMOVE if even the most similar local topic has cosine sim < threshold.
        """
        threshold = self.fallback_config.get("prune_threshold", 0.3)

        if not local_topics:
            # No local topics → remove all globals
            return [
                CurationDecision(topic_id=g.id, action="REMOVE",
                                 reason="No local topics to support")
                for g in global_topics
            ]

        sim_matrix = cosine_similarity_matrix(global_topics, local_topics)
        decisions = []

        for g_idx, g_topic in enumerate(global_topics):
            max_sim = float(sim_matrix[g_idx].max())
            if max_sim >= threshold:
                # Find the closest local topic to potentially refine
                best_local_idx = int(sim_matrix[g_idx].argmax())
                best_local = local_topics[best_local_idx]
                refined = _merge_topic_words(g_topic, best_local)
                decisions.append(CurationDecision(
                    topic_id=g_topic.id,
                    action="RETAIN",
                    refined_words=refined,
                    reason=f"max_sim={max_sim:.3f} >= {threshold}",
                    confidence=max_sim,
                ))
            else:
                decisions.append(CurationDecision(
                    topic_id=g_topic.id,
                    action="REMOVE",
                    reason=f"max_sim={max_sim:.3f} < {threshold} (not similar to any local topic)",
                    confidence=1.0 - max_sim,
                ))

        return decisions

    # ─── Stage 2: Detect Novel ───────────────────────────────────────────────

    def stage2_detect_novel(
        self,
        local_topics: list[Topic],
        retained_global_topics: list[Topic],
        top_k: int = 5,
    ) -> list[CurationDecision]:
        """For each local topic, decide NOVEL or COVERED.

        A local topic is NOVEL if it represents a genuinely new theme
        not covered by any retained global topic.
        """
        if not local_topics:
            return []

        if self.provider == "none":
            return self._stage2_fallback(local_topics, retained_global_topics)

        if not retained_global_topics:
            # All local topics are novel if no global topics exist
            return [
                CurationDecision(
                    topic_id=l.id, action="NOVEL",
                    refined_words=l.words,
                    reason="No global topics to compare against",
                )
                for l in local_topics
            ]

        nearest = find_nearest_topics(local_topics, retained_global_topics, top_k=top_k)

        decisions = []
        for l_idx, l_topic in enumerate(local_topics):
            neighbors = nearest[l_idx]
            neighbor_strs = []
            for g_idx, sim in neighbors:
                neighbor_strs.append(
                    f"  - {retained_global_topics[g_idx].to_string()}"
                )

            prompt = self._build_stage2_prompt(l_topic, neighbor_strs)
            response = self._call_llm(prompt, stage="Stage2-Novel", topic_id=l_topic.id)
            decision = self._parse_stage2_response(response, l_topic.id, l_topic.words)
            decisions.append(decision)

        return decisions

    def _build_stage2_prompt(self, local_topic: Topic, neighbor_strs: list[str]) -> str:
        return f"""You are a topic modeling expert analyzing research topic evolution.

LOCAL TOPIC (from current data):
  {local_topic.to_string()}

TOP-K MOST SIMILAR GLOBAL TOPICS (existing topics):
{chr(10).join(neighbor_strs)}

TASK: Decide whether this local topic is NOVEL or already COVERED by existing global topics.

DECISION CRITERIA:
- COVERED if at least one global topic shares a clearly related research theme,
  meaning this topic is already represented in the global memory.
- NOVEL if none of the global topics are semantically related, meaning this is a
  genuinely new or emerging research theme not yet in the global memory.

If NOVEL, provide refined representative words for the new topic.

Respond ONLY with a JSON object:
{{"action": "NOVEL" or "COVERED", "refined_words": ["word1", "word2", ...] or null, "reason": "brief explanation"}}
"""

    def _parse_stage2_response(self, response: str, topic_id: int,
                                default_words: list[str]) -> CurationDecision:
        try:
            data = _extract_json(response)
            return CurationDecision(
                topic_id=topic_id,
                action=data.get("action", "COVERED").upper(),
                refined_words=data.get("refined_words", default_words),
                reason=data.get("reason", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return CurationDecision(
            topic_id=topic_id, action="COVERED",
            reason="LLM parse failure",
        )

    def _stage2_fallback(
        self,
        local_topics: list[Topic],
        retained_global_topics: list[Topic],
    ) -> list[CurationDecision]:
        """Similarity-based fallback for Stage 2."""
        threshold = self.fallback_config.get("novel_threshold", 0.5)

        if not retained_global_topics:
            return [
                CurationDecision(
                    topic_id=l.id, action="NOVEL",
                    refined_words=l.words,
                    reason="No global topics to compare",
                    confidence=1.0,
                )
                for l in local_topics
            ]

        sim_matrix = cosine_similarity_matrix(local_topics, retained_global_topics)
        decisions = []

        for l_idx, l_topic in enumerate(local_topics):
            max_sim = float(sim_matrix[l_idx].max())
            if max_sim < threshold:
                decisions.append(CurationDecision(
                    topic_id=l_topic.id,
                    action="NOVEL",
                    refined_words=l_topic.words,
                    reason=f"max_sim={max_sim:.3f} < {threshold}",
                    confidence=1.0 - max_sim,
                ))
            else:
                decisions.append(CurationDecision(
                    topic_id=l_topic.id,
                    action="COVERED",
                    reason=f"max_sim={max_sim:.3f} >= {threshold}",
                    confidence=max_sim,
                ))

        return decisions


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _merge_topic_words(topic_a: Topic, topic_b: Topic, max_words: int = 15) -> list[str]:
    """Merge words from two topics, preferring topic_a's order."""
    seen = set()
    merged = []
    for w in topic_a.words:
        if w not in seen:
            merged.append(w)
            seen.add(w)
    for w in topic_b.words:
        if w not in seen and len(merged) < max_words:
            merged.append(w)
            seen.add(w)
    return merged[:max_words]


def _clip_probability(value, eps: float = 1e-4) -> float:
    """Convert an LLM/fallback value into a valid probability."""
    return float(np.clip(float(value), eps, 1.0 - eps))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _extract_json(text: str) -> dict:
    """Extract the first balanced JSON object from the text."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1: text = text[first_nl+1:]
        if text.endswith("```"): text = text[:-3]
    
    start_idx = text.find('{')
    if start_idx == -1:
        raise ValueError("No '{' found in response")
        
    brace_count = 0
    end_idx = -1
    in_string = False
    escape = False
    for i in range(start_idx, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
            
        if not in_string:
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
                    
    if end_idx != -1:
        return json.loads(text[start_idx:end_idx+1])
    raise ValueError("Failed to extract JSON: unmatched braces")
