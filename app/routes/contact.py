from fastapi import APIRouter, Depends, HTTPException
from app.models import ContactMessageCreate
from app.database import get_db
from app.services.auth import get_current_admin
from datetime import datetime
from bson import ObjectId

router = APIRouter(prefix="/api/contact", tags=["Contact"])


def serialize_doc(doc):
    doc["id"] = str(doc.pop("_id"))
    return doc


@router.post("/")
async def submit_contact(data: ContactMessageCreate):
    """Public endpoint: visitors can send contact messages."""
    db = get_db()
    msg = data.model_dump()
    msg["created_at"] = datetime.utcnow()
    msg["is_read"] = False
    result = await db.contact_messages.insert_one(msg)
    return {"message": "Message sent successfully", "id": str(result.inserted_id)}


@router.get("/")
async def get_messages(_=Depends(get_current_admin)):
    """Admin: list all contact messages."""
    db = get_db()
    messages = await db.contact_messages.find({}).sort("created_at", -1).to_list(500)
    return [serialize_doc(m) for m in messages]


@router.put("/{message_id}/read")
async def mark_as_read(message_id: str, _=Depends(get_current_admin)):
    db = get_db()
    await db.contact_messages.update_one(
        {"_id": ObjectId(message_id)}, {"$set": {"is_read": True}}
    )
    return {"message": "Marked as read"}


@router.delete("/{message_id}")
async def delete_message(message_id: str, _=Depends(get_current_admin)):
    db = get_db()
    result = await db.contact_messages.delete_one({"_id": ObjectId(message_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"message": "Deleted"}
