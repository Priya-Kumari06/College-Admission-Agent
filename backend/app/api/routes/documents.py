from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional
import os, uuid, shutil
from datetime import datetime

from app.core.security import get_current_user, require_officer_or_admin
from app.db.mongodb import get_database
from app.models.document import DocumentUploadResponse, DocumentStatus, DocumentCategory
from app.services.document.processor import extract_text, chunk_text, validate_file, get_file_hash
from app.services.rag.vector_store import vector_store
from app.core.config import settings

router = APIRouter(prefix="/documents", tags=["Document Management"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    category: DocumentCategory = Form(DocumentCategory.GENERAL),
    description: Optional[str] = Form(None),
    current_user=Depends(require_officer_or_admin),
    db=Depends(get_database)
):
    """Upload and process an admission document (Admin/Officer only)."""
    content = await file.read()
    file_size = len(content)

    valid, msg = validate_file(file.filename, file_size)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    # Save to disk
    os.makedirs(settings.upload_dir, exist_ok=True)
    doc_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    stored_name = f"{doc_id}{ext}"
    file_path = os.path.join(settings.upload_dir, stored_name)

    with open(file_path, "wb") as f:
        f.write(content)

    # Create DB record
    doc_record = {
        "_id": doc_id,
        "filename": stored_name,
        "original_name": file.filename,
        "category": category.value,
        "description": description or "",
        "status": DocumentStatus.PENDING.value,
        "uploaded_by": current_user["_id"],
        "uploaded_at": datetime.utcnow(),
        "chunk_count": 0,
        "file_size_kb": round(file_size / 1024, 2),
        "file_hash": get_file_hash(file_path)
    }
    await db.documents.insert_one(doc_record)

    # Process in background (extract, chunk, embed)
    try:
        await db.documents.update_one(
            {"_id": doc_id},
            {"$set": {"status": DocumentStatus.PROCESSING.value}}
        )
        pages = extract_text(file_path)
        chunks = chunk_text(pages, doc_name=file.filename)
        chunk_count = vector_store.add_documents(chunks, doc_id)

        await db.documents.update_one(
            {"_id": doc_id},
            {"$set": {
                "status": DocumentStatus.INDEXED.value,
                "chunk_count": chunk_count,
                "indexed_at": datetime.utcnow()
            }}
        )
    except Exception as e:
        await db.documents.update_one(
            {"_id": doc_id},
            {"$set": {"status": DocumentStatus.FAILED.value, "error": str(e)}}
        )
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

    return DocumentUploadResponse(
        id=doc_id,
        filename=stored_name,
        original_name=file.filename,
        category=category,
        status=DocumentStatus.INDEXED,
        uploaded_by=current_user["_id"],
        uploaded_at=doc_record["uploaded_at"],
        chunk_count=chunk_count,
        file_size_kb=doc_record["file_size_kb"]
    )


@router.get("/", response_model=dict)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: Optional[DocumentCategory] = None,
    current_user=Depends(get_current_user),
    db=Depends(get_database)
):
    """List all indexed documents."""
    query_filter = {}
    if category:
        query_filter["category"] = category.value

    total = await db.documents.count_documents(query_filter)
    cursor = db.documents.find(query_filter).skip((page - 1) * page_size).limit(page_size)
    docs = await cursor.to_list(length=page_size)

    return {
        "documents": [
            {
                "id": d["_id"],
                "original_name": d["original_name"],
                "category": d["category"],
                "status": d["status"],
                "chunk_count": d.get("chunk_count", 0),
                "file_size_kb": d.get("file_size_kb", 0),
                "uploaded_at": d["uploaded_at"].isoformat()
            }
            for d in docs
        ],
        "total": total,
        "page": page,
        "page_size": page_size
    }


@router.delete("/{doc_id}", status_code=200)
async def delete_document(
    doc_id: str,
    current_user=Depends(require_officer_or_admin),
    db=Depends(get_database)
):
    """Delete a document and its vectors (Admin/Officer only)."""
    doc = await db.documents.find_one({"_id": doc_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from vector store
    deleted_chunks = vector_store.delete_document(doc_id)

    # Remove file
    file_path = os.path.join(settings.upload_dir, doc["filename"])
    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove from DB
    await db.documents.delete_one({"_id": doc_id})

    return {"message": f"Document deleted. Removed {deleted_chunks} chunks from vector store."}
