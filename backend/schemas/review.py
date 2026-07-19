"""Review 相关 Pydantic schemas（API 请求/响应）。"""
from datetime import datetime

from pydantic import BaseModel, Field

from backend.models.review import Review

_VALID_LANGUAGES = {"python", "javascript", "typescript", "go", "java"}


class ReviewRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=50000)
    language: str = Field(..., pattern="^(python|javascript|typescript|go|java)$")

    model_config = {"extra": "forbid"}


class PRReviewRequest(BaseModel):
    pr_url: str = Field(..., min_length=10, max_length=500)

    model_config = {"extra": "forbid"}


class ReviewResponse(BaseModel):
    review_id: str
    status: str
    language: str | None = None
    code_length: int | None = None
    report: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None

    @classmethod
    def from_orm_model(cls, review: Review, review_id: str) -> "ReviewResponse":
        return cls(
            review_id=review_id,
            status=review.status,
            language=review.language,
            code_length=len(review.code_content or ""),
            report=review.report,
            created_at=review.created_at,
            completed_at=review.updated_at if review.status in ("completed", "failed") else None,
        )
