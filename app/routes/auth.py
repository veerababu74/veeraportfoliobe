from fastapi import APIRouter, HTTPException, Depends
from app.models import AdminLogin, TokenResponse
from app.services.auth import (
    verify_credentials,
    create_access_token,
    get_current_admin,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(data: AdminLogin):
    if not verify_credentials(data.username, data.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(data={"sub": data.username})
    return TokenResponse(access_token=token)


@router.get("/me")
async def get_me(current_admin=Depends(get_current_admin)):
    return {"username": current_admin["username"]}
