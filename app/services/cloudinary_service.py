import cloudinary
import cloudinary.uploader
from app.config import settings

cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
)


async def upload_image(file, folder: str = "portfolio") -> dict:
    """Upload an image to Cloudinary and return URL + public_id."""
    result = cloudinary.uploader.upload(
        file,
        folder=folder,
        resource_type="auto",
    )
    return {
        "url": result["secure_url"],
        "public_id": result["public_id"],
    }


async def delete_image(public_id: str) -> bool:
    """Delete an image from Cloudinary."""
    try:
        result = cloudinary.uploader.destroy(public_id)
        return result.get("result") == "ok"
    except Exception:
        return False
