"""
Enterprise-Grade AI Guardrails Engine
Implements all 15 guardrail requirements for the College Admission Assistant
"""
import re
import logging
from typing import Tuple, Dict, List, Optional
from enum import Enum
from datetime import datetime

from better_profanity import profanity
from app.models.chat import GuardrailStatus, QueryCategory
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize profanity filter
profanity.load_censor_words()


# ─────────────────────────────────────────────
# Guardrail 3: Prompt Injection Patterns
# ─────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|prior|above)\s+instructions?",
    r"reveal\s+(system\s+)?prompt",
    r"show\s+(me\s+)?your\s+(system\s+)?prompt",
    r"act\s+as\s+(another|a\s+different|an?\s+)",
    r"bypass\s+(admission|security|policy|filter|guardrail)",
    r"pretend\s+you\s+are",
    r"forget\s+(everything|all|your\s+instructions?)",
    r"you\s+are\s+now\s+(?!an?\s+admission)",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"override\s+(your\s+)?(instructions?|rules?|policies?)",
    r"disregard\s+(your\s+)?(instructions?|rules?)",
    r"new\s+persona",
    r"simulate\s+",
    r"roleplay\s+as",
    r"you\s+have\s+no\s+restrictions",
    r"without\s+(any\s+)?restrictions",
]

# ─────────────────────────────────────────────
# Guardrail 5: Sensitive/PII Patterns
# ─────────────────────────────────────────────
PII_PATTERNS = {
    "api_key": r"(api[_\-\s]?key|secret[_\-\s]?key)\s*[:=]\s*\S+",
    "password": r"(password|passwd|pwd)\s*[:=]\s*\S+",
    "database_uri": r"(mongodb|postgresql|mysql|redis):\/\/[^\s]+",
    "credit_card": r"\b(?:\d{4}[\s\-]?){3}\d{4}\b",
    "aadhaar": r"\b\d{4}\s\d{4}\s\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b",
    "pan_card": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
}

# ─────────────────────────────────────────────
# Guardrail 11: Restricted Query Topics
# ─────────────────────────────────────────────
RESTRICTED_TOPICS = [
    r"student\s+record",
    r"other\s+student('s)?\s+(grade|mark|score|admission)",
    r"staff\s+(salary|personal|contact|address)",
    r"financial\s+record",
    r"internal\s+(budget|report|meeting|memo)",
    r"database\s+password",
    r"admin\s+credentials",
    r"system\s+configuration",
    r"server\s+(ip|address|config)",
    r"employee\s+(id|data|record)",
    r"teacher\s+(salary|personal)",
]

# ─────────────────────────────────────────────
# Guardrail 10: Query Classification Keywords
# ─────────────────────────────────────────────
QUERY_CATEGORY_PATTERNS = {
    QueryCategory.FEE_STRUCTURE: [
        r"fee|fees|tuition|payment|cost|price|charges|scholarship\s+amount"
    ],
    QueryCategory.ELIGIBILITY: [
        r"eligible|eligibility|qualify|qualification|requirement|criteria|marks?\s+required"
    ],
    QueryCategory.DEADLINE: [
        r"deadline|last\s+date|due\s+date|when\s+(is|are)|schedule|calendar|timeline"
    ],
    QueryCategory.SCHOLARSHIP: [
        r"scholarship|bursary|financial\s+aid|grant|stipend|waiver|merit"
    ],
    QueryCategory.COURSE_INFO: [
        r"course|program|degree|branch|specialization|department|subject|curriculum"
    ],
    QueryCategory.ADMISSION_POLICY: [
        r"policy|process|procedure|how\s+to\s+apply|application|admission\s+process|steps"
    ],
    QueryCategory.GENERAL_FAQ: [
        r"hostel|campus|library|transport|facility|facilities|canteen|sports|contact"
    ],
}

SAFE_FALLBACK = (
    "I could not find this information in the official admission documents. "
    "Please contact the admission office directly for accurate guidance."
)

INJECTION_RESPONSE = (
    "I'm sorry, I cannot process that type of request. "
    "I'm here to help with college admission queries only. "
    "Please ask a genuine admission-related question."
)

TOXICITY_RESPONSE = (
    "Your message contained inappropriate content and could not be processed. "
    "Please maintain respectful communication. "
    "I'm here to assist with admission-related queries."
)

RESTRICTED_RESPONSE = (
    "This information is restricted and cannot be shared through this channel. "
    "Please contact the administration office directly for such requests."
)

LOW_CONFIDENCE_PREFIX = (
    "⚠️ The information retrieved has low confidence. "
    "Please verify with the admission office.\n\n"
)

HUMAN_HANDOFF_MESSAGE = (
    "Your query requires personal assistance from our admission team. "
    "Please contact us:\n"
    "📧 admissions@college.edu\n"
    "📞 +91-XXXXXXXXXX\n"
    "🕐 Mon-Fri, 9 AM - 5 PM"
)


class GuardrailResult:
    def __init__(
        self,
        passed: bool,
        status: GuardrailStatus,
        blocked_response: Optional[str] = None,
        violations: List[str] = None,
        masked_content: Optional[str] = None
    ):
        self.passed = passed
        self.status = status
        self.blocked_response = blocked_response
        self.violations = violations or []
        self.masked_content = masked_content


class GuardrailsEngine:
    """
    Enterprise-grade AI guardrails for College Admission Assistant.
    Implements all 15 safety and compliance requirements.
    """

    def check_prompt_injection(self, text: str) -> GuardrailResult:
        """Guardrail 3: Detect and block prompt injection attacks."""
        text_lower = text.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.warning(f"PROMPT INJECTION detected: pattern={pattern}, text={text[:100]}")
                return GuardrailResult(
                    passed=False,
                    status=GuardrailStatus.BLOCKED_INJECTION,
                    blocked_response=INJECTION_RESPONSE,
                    violations=[f"Injection pattern: {pattern}"]
                )
        return GuardrailResult(passed=True, status=GuardrailStatus.PASSED)

    def check_toxicity(self, text: str) -> GuardrailResult:
        """Guardrail 4: Filter offensive, hateful, or harmful content."""
        if not settings.enable_toxicity_filter:
            return GuardrailResult(passed=True, status=GuardrailStatus.PASSED)

        if profanity.contains_profanity(text):
            logger.warning(f"TOXICITY detected in query: {text[:100]}")
            return GuardrailResult(
                passed=False,
                status=GuardrailStatus.BLOCKED_TOXICITY,
                blocked_response=TOXICITY_RESPONSE,
                violations=["Profanity/offensive content detected"]
            )

        # Additional hate speech patterns
        hate_patterns = [
            r"\b(hate|kill|harm|destroy|attack)\s+(students?|teachers?|college|institution)\b",
            r"\b(racist|sexist|discriminat)\w*\b",
        ]
        for pattern in hate_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return GuardrailResult(
                    passed=False,
                    status=GuardrailStatus.BLOCKED_TOXICITY,
                    blocked_response=TOXICITY_RESPONSE,
                    violations=[f"Hate speech pattern: {pattern}"]
                )
        return GuardrailResult(passed=True, status=GuardrailStatus.PASSED)

    def check_restricted_topics(self, text: str) -> GuardrailResult:
        """Guardrail 11: Block restricted/sensitive topic queries."""
        text_lower = text.lower()
        for pattern in RESTRICTED_TOPICS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.warning(f"RESTRICTED TOPIC detected: {text[:100]}")
                return GuardrailResult(
                    passed=False,
                    status=GuardrailStatus.BLOCKED_RESTRICTED,
                    blocked_response=RESTRICTED_RESPONSE,
                    violations=[f"Restricted topic: {pattern}"]
                )
        return GuardrailResult(passed=True, status=GuardrailStatus.PASSED)

    def check_pii_in_response(self, response_text: str) -> Tuple[str, List[str]]:
        """Guardrail 5: Detect and mask PII/sensitive data in LLM response."""
        violations = []
        masked = response_text

        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, masked, re.IGNORECASE)
            if matches:
                violations.append(f"PII type: {pii_type}")
                masked = re.sub(pattern, f"[{pii_type.upper()} REDACTED]", masked, flags=re.IGNORECASE)
                logger.warning(f"PII found in response: {pii_type}")

        return masked, violations

    def check_confidence(self, score: float) -> GuardrailResult:
        """Guardrail 8: Low confidence threshold check."""
        if score < settings.confidence_threshold:
            return GuardrailResult(
                passed=False,
                status=GuardrailStatus.LOW_CONFIDENCE,
                violations=[f"Confidence {score:.0%} below threshold {settings.confidence_threshold:.0%}"]
            )
        return GuardrailResult(passed=True, status=GuardrailStatus.PASSED)

    def classify_query(self, text: str) -> QueryCategory:
        """Guardrail 10: Classify incoming query into admission categories."""
        text_lower = text.lower()

        # Check restricted first
        for pattern in RESTRICTED_TOPICS:
            if re.search(pattern, text_lower):
                return QueryCategory.RESTRICTED

        for category, patterns in QUERY_CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return category

        return QueryCategory.GENERAL_FAQ

    def should_trigger_human_handoff(
        self,
        confidence_score: float,
        context_found: bool,
        query_category: QueryCategory
    ) -> bool:
        """Guardrail 12: Determine if query should be escalated to human."""
        if not context_found:
            return True
        if confidence_score < 0.4:  # Very low confidence
            return True
        if query_category in [QueryCategory.RESTRICTED, QueryCategory.OUT_OF_SCOPE]:
            return True
        return False

    def validate_query(self, query: str) -> GuardrailResult:
        """
        Full query validation pipeline (input guardrails).
        Runs all input checks in order of priority.
        """
        # 1. Injection check (highest priority)
        result = self.check_prompt_injection(query)
        if not result.passed:
            return result

        # 2. Toxicity check
        result = self.check_toxicity(query)
        if not result.passed:
            return result

        # 3. Restricted topics check
        result = self.check_restricted_topics(query)
        if not result.passed:
            return result

        return GuardrailResult(passed=True, status=GuardrailStatus.PASSED)

    def validate_response(
        self,
        response: str,
        confidence_score: float,
        context_found: bool,
        query_category: QueryCategory
    ) -> Tuple[str, GuardrailStatus, bool, List[str]]:
        """
        Full response validation pipeline (output guardrails).
        Returns: (final_response, status, human_handoff_required, violations)
        """
        violations = []

        # Guardrail 5: Mask PII in response
        response, pii_violations = self.check_pii_in_response(response)
        violations.extend(pii_violations)

        # Guardrail 8: Confidence check
        human_handoff = self.should_trigger_human_handoff(
            confidence_score, context_found, query_category
        )

        if human_handoff:
            return (
                HUMAN_HANDOFF_MESSAGE,
                GuardrailStatus.HUMAN_HANDOFF,
                True,
                violations + ["Human handoff triggered"]
            )

        # Guardrail 8: Low confidence prefix
        confidence_check = self.check_confidence(confidence_score)
        if not confidence_check.passed:
            response = LOW_CONFIDENCE_PREFIX + response
            return (
                response,
                GuardrailStatus.LOW_CONFIDENCE,
                False,
                violations + confidence_check.violations
            )

        # If no context found, use fallback
        if not context_found or not response.strip():
            return (SAFE_FALLBACK, GuardrailStatus.PASSED, False, violations)

        return (response, GuardrailStatus.PASSED, False, violations)


# Singleton instance
guardrails_engine = GuardrailsEngine()
