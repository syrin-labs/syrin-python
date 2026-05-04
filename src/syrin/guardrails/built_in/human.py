"""Human approval guardrail for human-in-the-loop."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from syrin.enums import DecisionAction
from syrin.guardrails.base import Guardrail
from syrin.guardrails.context import GuardrailContext
from syrin.guardrails.decision import GuardrailDecision


class HumanApproval(Guardrail):
    """Guardrail requiring explicit human approval.

    Pauses execution and waits for human review and approval.
    Supports dynamic approver selection, 2FA, and justification requirements.

    Example:
        >>> guardrail = HumanApproval(
        ...     approver="admin@example.com",
        ...     timeout=300
        ... )
        >>> result = await guardrail.evaluate(context)
    """

    def __init__(
        self,
        approver: str | Callable[[GuardrailContext], str] | None = None,
        timeout: int = 3600,
        requires_justification: bool = False,
        require_2fa: bool = False,
        escalation_timeout: int | None = None,
        escalation_approver: str | None = None,
        name: str | None = None,
    ):
        """Initialize human approval.

        Args:
            approver: Approver email/ID or function to determine approver.
            timeout: Timeout in seconds.
            requires_justification: Whether approver must provide justification.
            require_2fa: Whether 2FA is required for approval.
            escalation_timeout: Seconds before escalating to another approver.
            escalation_approver: Approver to escalate to.
            name: Optional custom name.
        """
        super().__init__(name)
        self.approver = approver
        self.timeout = timeout
        self.requires_justification = requires_justification
        self.require_2fa = require_2fa
        self.escalation_timeout = escalation_timeout
        self.escalation_approver = escalation_approver

        # Track approvals: request_id -> approval data
        self._approvals: dict[str, dict[str, object]] = {}

    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        """Check if human approval has been granted.

        Args:
            context: Guardrail context.

        Returns:
            GuardrailDecision with approval status.
        """
        # Get request ID
        request_id = str(context.metadata.get("request_id", "default"))

        # Check if already approved/rejected
        if request_id in self._approvals:
            approval = self._approvals[request_id]

            if approval["status"] == "approved":
                return GuardrailDecision(
                    passed=True,
                    rule="human_approved",
                    metadata={
                        "request_id": request_id,
                        "approved_by": approval["approver"],
                        "approved_at": approval["timestamp"],
                        "justification": approval.get("justification"),
                    },
                )
            elif approval["status"] == "rejected":
                return GuardrailDecision(
                    passed=False,
                    rule="human_rejected",
                    reason=f"Rejected by {approval['approver']}: {approval.get('reason', 'No reason')}",
                    action=DecisionAction.BLOCK,
                    metadata={
                        "request_id": request_id,
                        "rejected_by": approval["approver"],
                        "reason": approval.get("reason"),
                        "rejected_at": approval["timestamp"],
                    },
                )

        # Determine approver
        approver_id = self._get_approver(context)

        # Request approval
        return GuardrailDecision(
            passed=False,
            rule="approval_required",
            reason=f"Human approval required from {approver_id}",
            action=DecisionAction.REQUEST_APPROVAL,
            metadata={
                "request_id": request_id,
                "approver": approver_id,
                "timeout": self.timeout,
                "requires_justification": self.requires_justification,
                "requires_2fa": self.require_2fa,
                "escalation": {
                    "timeout": self.escalation_timeout,
                    "approver": self.escalation_approver,
                }
                if self.escalation_timeout
                else None,
            },
            alternatives=[f"Contact {approver_id} for approval", "Wait for approval notification"],
        )

    def _get_approver(self, context: GuardrailContext) -> str:
        """Determine approver for context.

        Args:
            context: Guardrail context.

        Returns:
            Approver identifier.
        """
        if self.approver is None:
            return "default@example.com"

        if callable(self.approver):
            try:
                return self.approver(context)
            except Exception:
                return "default@example.com"

        return self.approver

    def approve(
        self,
        request_id: str,
        approver: str,
        justification: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Record approval for a request.

        Args:
            request_id: Request identifier.
            approver: Approver identifier.
            justification: Optional justification.
            metadata: Additional metadata.
        """
        self._approvals[request_id] = {
            "status": "approved",
            "approver": approver,
            "justification": justification,
            "timestamp": datetime.now(),
            "metadata": metadata or {},
        }

    def reject(self, request_id: str, approver: str, reason: str | None = None) -> None:
        """Record rejection for a request.

        Args:
            request_id: Request identifier.
            approver: Approver identifier.
            reason: Rejection reason.
        """
        self._approvals[request_id] = {
            "status": "rejected",
            "approver": approver,
            "reason": reason,
            "timestamp": datetime.now(),
        }

    def reset(self, request_id: str) -> None:
        """Reset approval state for a request.

        Args:
            request_id: Request identifier.
        """
        self._approvals.pop(request_id, None)
