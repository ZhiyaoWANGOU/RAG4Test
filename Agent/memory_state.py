# Agent/memory_state.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json

@dataclass
class MemoryState:
    """
    Represents all short-term memory for a single feedback processing session.
    Each feedback gets its own MemoryState instance.
    """
    feedback: str
    collected: List[str] = field(default_factory=list)   # relevant but insufficient docs
    decision: Optional[str] = None                      # reasoning output summary
    bug_report: Optional[str] = None                    # final generated report
    metadata: Dict[str, Any] = field(default_factory=dict)  # optional extra info

    def add_evidence(self, doc: str):
        """Add one relevant but insufficient document to the memory."""
        self.collected.append(doc)

    def set_decision(self, decision: str):
        """Record agent reasoning result."""
        self.decision = decision

    def set_bug_report(self, report: str):
        """Store the final generated bug report."""
        self.bug_report = report

    def to_context(self) -> str:
        """
        Generate a full textual context for LLMs.
        This packs all relevant state into a single readable prompt.
        """
        parts = [f"User feedback:\n{self.feedback}\n"]
        if self.collected:
            parts.append("Relevant but partial knowledge:\n" + "\n".join(self.collected))
        if self.decision:
            parts.append(f"Reasoning summary:\n{self.decision}")
        return "\n\n".join(parts)

    def to_json(self) -> str:
        """Convert the current state to JSON (for logging or saving)."""
        return json.dumps({
            "feedback": self.feedback,
            "collected": self.collected,
            "decision": self.decision,
            "bug_report": self.bug_report,
            "metadata": self.metadata
        }, indent=2)
