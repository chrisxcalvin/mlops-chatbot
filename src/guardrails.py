"""Lightweight input/output guardrails."""
import re

INJECTION = re.compile(r"ignore (all |the )?(previous|above|prior) (instructions|rules)|system prompt|"
                       r"reveal your (prompt|instructions)|you are now|jailbreak", re.I)
PII = re.compile(r"\b\d{10}\b|\b[\w.+-]+@[\w-]+\.[\w.]+\b|\b\d{4}\s?\d{4}\s?\d{4}\b")
SMALLTALK = {
    r"^(hi|hello|hey|good (morning|afternoon|evening))\b": "Hello! 👋 I'm the CHRIST Student Helpdesk assistant. Ask me about attendance, exams, fees, hostels, the library, programmes and more — all from the Student Handbook 2026-27.",
    r"^(thanks|thank you|thx)\b": "You're welcome! Let me know if there's anything else I can help with.",
    r"^(bye|goodbye)\b": "Goodbye! Best wishes for your studies at CHRIST.",
}

CAPABILITY = re.compile(r"(what|how)\s+(can|could|do|does)\s+(you|u)\b.*\b(help|do|know|answer)|"
                        r"\bhelp me with\b|\bwho are (you|u)\b|\bwhat are (you|u)\b|\bwhat (is|are) (this|your)\b|"
                        r"\bwhat can (you|u)\b|^help\W*$|\bhow to use\b|\bwhat topics\b", re.I)
CAPABILITY_REPLY = ("I'm the CHRIST Student Helpdesk assistant. I answer questions from the **Student Handbook 2026-27** "
                    "and tell you which page it came from. Try asking me about:\n\n"
                    "- **Exams & assessment** - MSE rules, repeat chances, dress code, credits, promotion\n"
                    "- **Library** - borrowing limits, loan periods, fines, facilities\n"
                    "- **Academic calendar** - semester dates, holidays, exam schedules\n"
                    "- **Programmes & campuses** - which campus offers which course\n"
                    "- **Fees & scholarships** - concessions, waivers\n"
                    "- **Rules & conduct** - hostels, dress code, regulations, surveys, bookings\n"
                    "- **Other** - international students, student council, special-needs policy, NEP guidelines\n\n"
                    "For example: \"What happens if I miss the MSE?\" or \"How many books can a PG student borrow?\"")


def check_input(q: str):
    """Returns (ok, reply_if_blocked, sanitized_query, pii_found)."""
    q = q.strip()[:800]
    if len(q) < 2:
        return False, "Please type your question.", q, False
    if len(q) < 90 and CAPABILITY.search(q):
        return False, CAPABILITY_REPLY, q, False
    for pat, reply in SMALLTALK.items():
        if re.match(pat, q, re.I) and len(q) < 40:
            return False, reply, q, False
    if INJECTION.search(q):
        return False, "I can only answer questions about the CHRIST Student Handbook. Please rephrase your question.", q, False
    pii = bool(PII.search(q))
    return True, None, PII.sub("[redacted]", q), pii
