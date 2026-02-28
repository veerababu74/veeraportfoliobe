from fastapi import APIRouter, Depends, HTTPException
from typing import List
from bson import ObjectId
from app.models import (
    Skill,
    SkillCreate,
    Experience,
    ExperienceCreate,
    Education,
    EducationCreate,
    Project,
    ProjectCreate,
    Certification,
    CertificationCreate,
    Testimonial,
    TestimonialCreate,
)
from app.services.auth import get_current_admin
from app.database import get_db

router = APIRouter(prefix="/api/content", tags=["Content"])


def serialize_doc(doc):
    doc["id"] = str(doc.pop("_id"))
    return doc


# ─── Generic CRUD helpers ───
async def _list_items(collection_name: str):
    db = get_db()
    items = await db[collection_name].find({}).sort("order", 1).to_list(1000)
    return [serialize_doc(item) for item in items]


async def _create_item(collection_name: str, data: dict):
    db = get_db()
    result = await db[collection_name].insert_one(data)
    return {"id": str(result.inserted_id), "message": "Created"}


async def _update_item(collection_name: str, item_id: str, data: dict):
    db = get_db()
    update_data = {k: v for k, v in data.items() if v is not None}
    result = await db[collection_name].update_one(
        {"_id": ObjectId(item_id)}, {"$set": update_data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"message": "Updated"}


async def _delete_item(collection_name: str, item_id: str):
    db = get_db()
    result = await db[collection_name].delete_one({"_id": ObjectId(item_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"message": "Deleted"}


# ─── Skills ───
@router.get("/skills")
async def get_skills():
    return await _list_items("skills")


@router.post("/skills")
async def create_skill(data: SkillCreate, _=Depends(get_current_admin)):
    return await _create_item("skills", data.model_dump())


@router.put("/skills/{item_id}")
async def update_skill(item_id: str, data: SkillCreate, _=Depends(get_current_admin)):
    return await _update_item("skills", item_id, data.model_dump())


@router.delete("/skills/{item_id}")
async def delete_skill(item_id: str, _=Depends(get_current_admin)):
    return await _delete_item("skills", item_id)


# ─── Experience ───
@router.get("/experience")
async def get_experience():
    return await _list_items("experience")


@router.post("/experience")
async def create_experience(data: ExperienceCreate, _=Depends(get_current_admin)):
    return await _create_item("experience", data.model_dump())


@router.put("/experience/{item_id}")
async def update_experience(
    item_id: str, data: ExperienceCreate, _=Depends(get_current_admin)
):
    return await _update_item("experience", item_id, data.model_dump())


@router.delete("/experience/{item_id}")
async def delete_experience(item_id: str, _=Depends(get_current_admin)):
    return await _delete_item("experience", item_id)


# ─── Education ───
@router.get("/education")
async def get_education():
    return await _list_items("education")


@router.post("/education")
async def create_education(data: EducationCreate, _=Depends(get_current_admin)):
    return await _create_item("education", data.model_dump())


@router.put("/education/{item_id}")
async def update_education(
    item_id: str, data: EducationCreate, _=Depends(get_current_admin)
):
    return await _update_item("education", item_id, data.model_dump())


@router.delete("/education/{item_id}")
async def delete_education(item_id: str, _=Depends(get_current_admin)):
    return await _delete_item("education", item_id)


# ─── Projects ───
@router.get("/projects")
async def get_projects():
    return await _list_items("projects")


@router.post("/projects")
async def create_project(data: ProjectCreate, _=Depends(get_current_admin)):
    return await _create_item("projects", data.model_dump())


@router.put("/projects/{item_id}")
async def update_project(
    item_id: str, data: ProjectCreate, _=Depends(get_current_admin)
):
    return await _update_item("projects", item_id, data.model_dump())


@router.delete("/projects/{item_id}")
async def delete_project(item_id: str, _=Depends(get_current_admin)):
    return await _delete_item("projects", item_id)


# ─── Certifications ───
@router.get("/certifications")
async def get_certifications():
    return await _list_items("certifications")


@router.post("/certifications")
async def create_certification(data: CertificationCreate, _=Depends(get_current_admin)):
    return await _create_item("certifications", data.model_dump())


@router.put("/certifications/{item_id}")
async def update_certification(
    item_id: str, data: CertificationCreate, _=Depends(get_current_admin)
):
    return await _update_item("certifications", item_id, data.model_dump())


@router.delete("/certifications/{item_id}")
async def delete_certification(item_id: str, _=Depends(get_current_admin)):
    return await _delete_item("certifications", item_id)


# ─── Testimonials ───
@router.get("/testimonials")
async def get_testimonials():
    return await _list_items("testimonials")


@router.post("/testimonials")
async def create_testimonial(data: TestimonialCreate, _=Depends(get_current_admin)):
    return await _create_item("testimonials", data.model_dump())


@router.put("/testimonials/{item_id}")
async def update_testimonial(
    item_id: str, data: TestimonialCreate, _=Depends(get_current_admin)
):
    return await _update_item("testimonials", item_id, data.model_dump())


@router.delete("/testimonials/{item_id}")
async def delete_testimonial(item_id: str, _=Depends(get_current_admin)):
    return await _delete_item("testimonials", item_id)
