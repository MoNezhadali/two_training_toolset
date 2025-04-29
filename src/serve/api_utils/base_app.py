"""Module for base FastAPI application setup."""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from src.serve.api_utils.authentication import dummy_authenticator

router = APIRouter()


@router.get("/health", summary="Health Check")
async def health_check(
    verified_token: bool = Depends(dummy_authenticator),
):
    """
    Health check endpoint to verify if the service is running.

    Args:
    ----
        verified_token (bool): Token verification status.

    Returns:
    -------
        dict: Health status message.

    Raises:
    ------
        HTTPException: If the token verification fails.

    """
    if not verified_token:
        logger.warning("Unauthorized access to /health endpoint")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return {"status": "ok", "message": "Service is healthy."}
