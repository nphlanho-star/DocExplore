"""
api/deps.py — Dependencies dùng chung cho các routes.

Chế độ dev: không cần JWT, tự động dùng dev user duy nhất trong DB.
"""
from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User

DEV_EMAIL = "dev@local"


async def get_current_user(
    db: AsyncSession = Depends(get_db),
) -> User:
    """Trả về dev user. Không yêu cầu token."""
    result = await db.execute(select(User).where(User.email == DEV_EMAIL))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dev user chưa được khởi tạo. Khởi động lại backend.",
        )
    return user


async def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    return current_user
