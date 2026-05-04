"""Threshold approval guardrail for K-of-N consensus."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from syrin.enums import DecisionAction
from syrin.guardrails.base import Guardrail
from syrin.guardrails.context import GuardrailContext
from syrin.guardrails.decision import GuardrailDecision


class ThresholdApproval(Guardrail):
    """Guardrail requiring K-of-N approvals for critical actions.

    Implements threshold cryptography principles where critical actions
    require approval from K independent approvers out of N total.

    Example:
        >>> guardrail = ThresholdApproval(k=2, n=3)
        >>> result = await guardrail.evaluate(context)
        >>> # Returns REQUEST_APPROVAL until threshold met
    """

    def __init__(
        self,
        k: int,
        n: int,
        approvers: list[str] | None = None,
        timeout: int = 3600,
        condition: Callable[[GuardrailContext], bool] | None = None,
        name: str | None = None,
    ):
        """Initialize threshold approval.

        Args:
            k: Number of approvals required.
            n: Total number of potential approvers.
            approvers: List of approver identifiers.
            timeout: Timeout in seconds.
            condition: Optional condition. If False, check is skipped.
            name: Optional custom name.

        Raises:
            ValueError: If k > n or k < 1.
        """
        super().__init__(name)

        if k > n:
            raise ValueError(f"k ({k}) cannot be greater than n ({n})")
        if k < 1:
            raise ValueError(f"k ({k}) must be at least 1")

        self.k = k
        self.n = n
        self.approvers = approvers or []
        self.timeout = timeout
        self.condition = condition

        # Track approvals: request_id -> set of approvers
        self._approvals: dict[str, set[str]] = {}
        self._rejections: dict[str, list[dict[str, object]]] = {}
        self._request_times: dict[str, datetime] = {}

    async def evaluate(self, context: GuardrailContext) -> GuardrailDecision:
        """Check if threshold approval has been met.

        Args:
            context: Guardrail context.

        Returns:
            GuardrailDecision with approval status.
        """
        # Check condition
        if self.condition is not None and not self.condition(context):
            return GuardrailDecision(
                passed=True,
                rule="condition_not_met",
                reason="Condition not met, threshold check skipped",
                metadata={"k": self.k, "n": self.n},
            )

        # Get request ID
        request_id = str(context.metadata.get("request_id", "default"))

        # Check if request has expired
        if request_id in self._request_times:
            elapsed = (datetime.now() - self._request_times[request_id]).total_seconds()
            if elapsed > self.timeout:
                return GuardrailDecision(
                    passed=False,
                    rule="approval_timeout",
                    reason=f"Approval timed out after {self.timeout}s",
                    action=DecisionAction.BLOCK,
                    metadata={
                        "request_id": request_id,
                        "elapsed": elapsed,
                        "timeout": self.timeout,
                    },
                )
        else:
            # First time seeing this request
            self._request_times[request_id] = datetime.now()

        # Check for rejections
        if request_id in self._rejections and self._rejections[request_id]:
            rejection = self._rejections[request_id][0]
            return GuardrailDecision(
                passed=False,
                rule="approval_rejected",
                reason=f"Rejected by {rejection['approver']}: {rejection.get('reason', 'No reason')}",
                action=DecisionAction.BLOCK,
                metadata={"request_id": request_id, "rejections": self._rejections[request_id]},
            )

        # Check approvals
        approvals = self._approvals.get(request_id, set())
        approval_count = len(approvals)

        if approval_count >= self.k:
            # Threshold met!
            return GuardrailDecision(
                passed=True,
                rule="threshold_met",
                metadata={
                    "request_id": request_id,
                    "approvals_received": approval_count,
                    "approvals_needed": self.k,
                    "approvers": list(approvals),
                },
            )

        # Still waiting for approvals
        return GuardrailDecision(
            passed=False,
            rule="approval_pending",
            reason=f"Awaiting approval: {approval_count}/{self.k} received",
            action=DecisionAction.REQUEST_APPROVAL,
            metadata={
                "request_id": request_id,
                "approvals_received": approval_count,
                "approvals_needed": self.k,
                "approvers_total": self.n,
                "current_approvers": list(approvals),
                "timeout": self.timeout,
            },
            alternatives=[
                f"Request approval from {self.k - approval_count} more approver(s)",
                "Wait for pending approvals",
            ],
        )

    def add_approval(self, request_id: str, approver: str) -> bool:
        """Add an approval for a request.

        Args:
            request_id: Request identifier.
            approver: Approver identifier.

        Returns:
            True if approval was added, False if duplicate.
        """
        if request_id not in self._approvals:
            self._approvals[request_id] = set()

        # Prevent duplicate approvals
        if approver in self._approvals[request_id]:
            return False

        self._approvals[request_id].add(approver)

        # Initialize request time if not set
        if request_id not in self._request_times:
            self._request_times[request_id] = datetime.now()

        return True

    def add_rejection(self, request_id: str, approver: str, reason: str | None = None) -> None:
        """Add a rejection for a request.

        Args:
            request_id: Request identifier.
            approver: Approver identifier.
            reason: Optional rejection reason.
        """
        if request_id not in self._rejections:
            self._rejections[request_id] = []

        self._rejections[request_id].append(
            {"approver": approver, "reason": reason, "timestamp": datetime.now()}
        )

    def get_approvals(self, request_id: str) -> set[str]:
        """Get current approvals for a request.

        Args:
            request_id: Request identifier.

        Returns:
            Set of approvers.
        """
        return self._approvals.get(request_id, set()).copy()

    def reset(self, request_id: str) -> None:
        """Reset approvals for a request.

        Args:
            request_id: Request identifier.
        """
        self._approvals.pop(request_id, None)
        self._rejections.pop(request_id, None)
        self._request_times.pop(request_id, None)
