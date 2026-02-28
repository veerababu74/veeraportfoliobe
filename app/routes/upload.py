from fastapi import APIRouter, Depends, UploadFile, File
from app.services.cloudinary_service import upload_image, delete_image
from app.services.auth import get_current_admin

router = APIRouter(prefix="/api/upload", tags=["Upload"])


@router.post("/")
async def upload_file(file: UploadFile = File(...), _=Depends(get_current_admin)):
    """Upload a file to Cloudinary (images, PDFs, etc.)."""
    result = await upload_image(file.file, folder="portfolio")
    return result


@router.delete("/")
async def delete_file(public_id: str, _=Depends(get_current_admin)):
    """Delete a file from Cloudinary by public_id."""
    success = await delete_image(public_id)
    if success:
        return {"message": "File deleted"}
    return {"message": "Failed to delete file"}
