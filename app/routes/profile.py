from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime
from app.models import (
    Profile,
    ProfileUpdate,
    About,
    AboutUpdate,
    SiteSettings,
    SiteSettingsUpdate,
)
from app.services.auth import get_current_admin
from app.database import get_db

router = APIRouter(prefix="/api/profile", tags=["Profile"])


# ─── Profile ───
@router.get("/", response_model=Profile)
async def get_profile():
    db = get_db()
    profile = await db.profile.find_one({})
    if not profile:
        return Profile()
    profile.pop("_id", None)
    return Profile(**profile)


@router.put("/")
async def update_profile(data: ProfileUpdate, _=Depends(get_current_admin)):
    db = get_db()
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    update_data["updated_at"] = datetime.utcnow()

    existing = await db.profile.find_one({})
    if existing:
        await db.profile.update_one({}, {"$set": update_data})
    else:
        await db.profile.insert_one(update_data)

    return {"message": "Profile updated"}


# ─── About ───
@router.get("/about", response_model=About)
async def get_about():
    db = get_db()
    about = await db.about.find_one({})
    if not about:
        return About()
    about.pop("_id", None)
    return About(**about)


@router.put("/about")
async def update_about(data: AboutUpdate, _=Depends(get_current_admin)):
    db = get_db()
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    update_data["updated_at"] = datetime.utcnow()

    existing = await db.about.find_one({})
    if existing:
        await db.about.update_one({}, {"$set": update_data})
    else:
        await db.about.insert_one(update_data)

    return {"message": "About updated"}


# ─── Site Settings ───
@router.get("/settings", response_model=SiteSettings)
async def get_site_settings():
    db = get_db()
    s = await db.site_settings.find_one({})
    if not s:
        return SiteSettings()
    s.pop("_id", None)
    return SiteSettings(**s)


@router.put("/settings")
async def update_site_settings(data: SiteSettingsUpdate, _=Depends(get_current_admin)):
    db = get_db()
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    update_data["updated_at"] = datetime.utcnow()

    existing = await db.site_settings.find_one({})
    if existing:
        await db.site_settings.update_one({}, {"$set": update_data})
    else:
        await db.site_settings.insert_one(update_data)

    return {"message": "Settings updated"}
