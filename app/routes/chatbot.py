from fastapi import APIRouter, Depends, HTTPException, Query
from app.models import ChatMessage, ChatResponse, VectorUpload, LLMSettingsUpdate
from app.services.chatbot import (
    query_chatbot,
    upsert_vectors,
    delete_all_vectors,
    get_llm_settings,
    save_llm_settings,
    list_chunks,
    delete_chunk,
)
from app.services.auth import get_current_admin
import traceback

router = APIRouter(prefix="/api/chatbot", tags=["Chatbot"])


@router.post("/chat", response_model=ChatResponse)
async def chat(data: ChatMessage):
    """Public endpoint: visitors can ask questions about the portfolio owner."""
    try:
        result = await query_chatbot(data.message)
        return ChatResponse(**result)
    except Exception as e:
        print(f"[Chatbot Route] ERROR: {e}")
        traceback.print_exc()
        return ChatResponse(
            response="Sorry, I'm having trouble right now. Please try again later.",
            sources=[],
        )


@router.post("/upload-vectors")
async def upload_vectors(data: VectorUpload, _=Depends(get_current_admin)):
    """Admin: upload text data with LangChain chunking to Pinecone."""
    try:
        count = await upsert_vectors(
            data.texts, data.metadata, data.chunk_size, data.chunk_overlap
        )
        return {"message": f"Uploaded {count} chunks successfully", "count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/vectors")
async def clear_vectors(_=Depends(get_current_admin)):
    """Admin: delete all vectors from Pinecone and MongoDB."""
    try:
        await delete_all_vectors()
        return {"message": "All vectors and chunks deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── LLM Settings ───
@router.get("/llm-settings")
async def get_settings(_=Depends(get_current_admin)):
    """Admin: get LLM provider configuration."""
    return await get_llm_settings()


@router.put("/llm-settings")
async def update_settings(data: LLMSettingsUpdate, _=Depends(get_current_admin)):
    """Admin: update LLM provider configuration."""
    update_data = data.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    result = await save_llm_settings(update_data)
    return {"message": "LLM settings updated", "data": result}


# ─── Chunk Management ───
@router.get("/chunks")
async def get_chunks(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    _=Depends(get_current_admin),
):
    """Admin: list all stored knowledge chunks."""
    return await list_chunks(skip, limit)


@router.delete("/chunks/{chunk_id}")
async def remove_chunk(chunk_id: str, _=Depends(get_current_admin)):
    """Admin: delete a single chunk from Pinecone and MongoDB."""
    success = await delete_chunk(chunk_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return {"message": "Chunk deleted"}
