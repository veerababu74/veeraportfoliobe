"""
Chatbot service with Groq (primary) + Google AI (fallback) LLM support,
custom text splitting, Pinecone RAG, and fallback resume context.
"""

from pinecone import Pinecone
from groq import Groq
import google.generativeai as genai
from app.config import settings
from app.database import get_db
from typing import List, Optional
from datetime import datetime
import traceback
import hashlib

# ─── Module-level caches ───
_pinecone_index = None
_pinecone_key_hash = None
_pinecone_index_name = None

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful portfolio assistant for Veerababu Pilli. "
    "You answer questions about him based ONLY on the provided context. "
    "If the context doesn't contain relevant information, politely say you don't have that information. "
    "Be concise, friendly, and professional. Keep answers focused and relevant. "
    "Always refer to the portfolio owner as 'Veerababu' or 'Veera'."
)

# ─── Fallback Resume Data (used when Pinecone has no data or fails) ───
FALLBACK_RESUME_CHUNKS = [
    "VEERABABU PILLI is a Python & GenAI Developer based in Hyderabad, India. "
    "Contact: pveerababu199966@gmail.com, +91 9666947399.",
    "PROFESSIONAL SUMMARY: Python & GenAI Developer with 4+ years of experience at Tata Consultancy Services, "
    "specializing in building production-grade Generative AI and Machine Learning solutions for enterprise CPG "
    "marketing applications. Expertise in Azure AI (Azure OpenAI, AI Foundry, AI Search, Vision, Storage, App Services), "
    "Azure Agent Framework, MCP Servers, and Google Vertex AI Studio. Proficient in designing scalable RESTful APIs "
    "using FastAPI and Flask, and developing advanced agentic AI systems using LangChain, LangGraph, RAG, and Graph RAG pipelines.",
    "KEY ACHIEVEMENTS: Delivered measurable impact including 30% reduction in agency costs, "
    "70% reduction in manual effort, and 40% improvement in domain-specific query accuracy.",
    "WORK EXPERIENCE: Tata Consultancy Services, Python & GenAI Developer (05/2022 – Present, Hyderabad, India). "
    "Designed and developed scalable RESTful APIs using FastAPI and Flask to expose GenAI and ML capabilities, "
    "including image generation, RAG-based chatbots, video generation, and social caption automation for enterprise applications.",
    "TCS RESPONSIBILITIES: Provisioned and managed Azure AI Foundry environments, deploying and integrating "
    "Azure OpenAI models (GPT-4, GPT-3.5) into production pipelines with version control, quota optimization, "
    "and secure endpoint configurations. Built end-to-end Azure AI solutions leveraging Azure OpenAI, Azure AI Search, "
    "Azure Vision, Azure Storage, and Azure App Service with secrets management using Azure Key Vault.",
    "TCS RAG & AGENTIC AI WORK: Architected and deployed production-grade RAG and Graph RAG pipelines using "
    "Azure AI Search (hybrid BM25 + vector search), LangChain, and LlamaIndex, improving domain-specific query accuracy "
    "by up to 40%. Orchestrated agentic AI workflows using LangGraph, enabling autonomous decision-making and "
    "coordination across multi-step, multi-tool pipelines, significantly reducing manual intervention.",
    "TCS MULTIMODAL & MCP WORK: Integrated Google Vertex AI (Gemini, Imagen, Veo) for multi-modal content generation "
    "(text, image, video), and combined with Azure OpenAI to design hybrid-cloud GenAI architectures. Applied advanced "
    "prompt engineering techniques (chain-of-thought, few-shot, structured outputs). Leveraged MCP (Model Context Protocol) "
    "servers by designing, building, and deploying MCP services using the FastMCP framework on Azure App Service.",
    "SKILLS - Programming Languages: Python. Machine Learning: Linear Regression, Logistic Regression, Decision Trees, "
    "Random Forest, XGBoost. Frameworks & Libraries: FastAPI, Flask, LangChain, LangGraph, Azure Agent Framework, "
    "NumPy, Pandas, Scikit-learn, FastMCP.",
    "SKILLS - Cloud & Databases: Microsoft Azure (Azure OpenAI, AI Foundry, AI Search, Vision, App Services), "
    "Google Cloud (Vertex AI). Databases: MySQL, MongoDB, Vector Databases (FAISS, ChromaDB, Azure AI Search). "
    "Tools & DevOps: Git, Azure DevOps.",
    "SKILLS - GenAI & Agentic AI: GPT (OpenAI), Gemini (Google), RAG, Graph RAG, Prompt Engineering. "
    "LLM Agents, Multi-Agent Systems, Tool Calling, Function Calling, Agent Orchestration (LangGraph), "
    "MCP (Model Context Protocol) Servers.",
    "EDUCATION: Bachelor of Technology - Electronics & Communication Engineering (B.Tech ECE), "
    "VSM College of Engineering, Ramachandrapuram.",
    "PROJECT - CPG GenAI Marketing & Content Generation Suite: Technologies: Python, FastAPI, Azure OpenAI GPT-4, "
    "Google Imagen3, Nanobanana, Google Veo, FFmpeg. Built Python (FastAPI) backend APIs with a custom prompt "
    "generator and GPT-4 integration to enforce brand guidelines for image creation, reducing agency costs by 30%. "
    "Automated product image placement on AI-generated backgrounds using Nanobanana with embedded metadata, "
    "decreasing manual effort by 60% and improving tagging accuracy by 40%.",
    "PROJECT - Social Caption Automation: Developed backend pipelines in Python to analyze real-time "
    "voice-of-consumer (VoC) data and generate platform-specific captions using GenAI models aligned with "
    "social sentiment and consumer preferences. Automated multi-platform caption creation (Instagram, LinkedIn, "
    "TikTok, Twitter) by embedding consumer insights and brand personality, increasing social media engagement "
    "by 45% and reducing manual effort by 70%.",
    "PROJECT - Video Generation & AI Timeline Automation: Developed a backend video generation module with "
    "a prompt-engineering pipeline to convert user input into structured prompts, automate chapter extraction, "
    "and assemble long-form videos using FFmpeg and AI-suggested timestamps. Designed capabilities to merge "
    "multiple video clips into continuous long-form content, reducing video production turnaround by 50%.",
    "PROJECT - Marketing Assistant Bot (RAG-Powered): Technologies: Python, Azure AI Search, Azure OpenAI GPT-4, "
    "LangChain, Hybrid Search, Azure Semantic Ranking. Developed a production-grade RAG-powered marketing assistant "
    "for enterprise teams to generate marketing briefs and extract geo-specific insights. Implemented hybrid search "
    "with Azure Semantic Ranking and GPT-4-based re-ranking using Reciprocal Rank Fusion (RRF).",
    "PROJECT - Agentic AI Workflow Builder: Technologies: Python, FastAPI, LangGraph, Custom Agent Orchestration. "
    "Developed backend APIs using FastAPI to orchestrate multi-step AI agent workflows. Architected a modular "
    "backend system supporting plug-and-play agent extensions, reducing feature delivery time by 25%.",
    "PROJECT - Code Detection & Data Security Layer: Technologies: Python, Scikit-learn, ML Classification, FastAPI. "
    "Designed a multi-class ML classification system to detect programming language content within user inputs. "
    "Built a large-scale custom dataset (~1M samples). Deployed as FastAPI-based security middleware protecting "
    "proprietary code assets for 20+ internal team members, preventing accidental IP leakage to third-party LLM providers.",
    "CERTIFICATIONS: Microsoft Certified: Azure Fundamentals (AZ-900), "
    "Microsoft Certified: Azure AI Fundamentals (AI-900), Microsoft Certified: Azure AI.",
]


# ─── Default LLM Settings ───
def _default_llm_settings() -> dict:
    return {
        "active_provider": "groq",
        # Groq (primary)
        "groq_api_key": settings.GROQ_API_KEY or "",
        "groq_model": "llama-3.1-8b-instant",
        "groq_models": [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        # Google (fallback LLM + embeddings)
        "google_api_key": settings.GOOGLE_API_KEY or "",
        "google_model": "gemini-1.5-flash",
        "google_models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        # Common
        "temperature": 0.7,
        "max_tokens": 1000,
        # Pinecone
        "pinecone_api_key": settings.PINECONE_API_KEY or "",
        "pinecone_index_name": settings.PINECONE_INDEX_NAME or "portfolio-chatbot",
        # Embedding (Google API key for generating vectors)
        "embedding_google_api_key": settings.GOOGLE_API_KEY or "",
    }


# ══════════════════════════════════════════════════════
#  Settings Management
# ══════════════════════════════════════════════════════


async def get_llm_settings() -> dict:
    """Load LLM settings from MongoDB, merging with env-based defaults."""
    db = get_db()
    defaults = _default_llm_settings()
    doc = await db.llm_settings.find_one({})
    if doc:
        doc.pop("_id", None)
        for k, v in doc.items():
            if v is not None and v != "":
                defaults[k] = v
    return defaults


async def save_llm_settings(data: dict) -> dict:
    """Save LLM settings to MongoDB (upsert)."""
    db = get_db()
    data["updated_at"] = datetime.utcnow().isoformat()
    await db.llm_settings.update_one({}, {"$set": data}, upsert=True)
    return data


# ══════════════════════════════════════════════════════
#  Pinecone Management
# ══════════════════════════════════════════════════════


def _get_pinecone_index(api_key: str, index_name: str):
    """Get or create a cached Pinecone index connection."""
    global _pinecone_index, _pinecone_key_hash, _pinecone_index_name
    key_hash = hashlib.md5(api_key.encode()).hexdigest()
    if (
        _pinecone_index
        and _pinecone_key_hash == key_hash
        and _pinecone_index_name == index_name
    ):
        return _pinecone_index
    try:
        pc = Pinecone(api_key=api_key)
        _pinecone_index = pc.Index(index_name)
        _pinecone_key_hash = key_hash
        _pinecone_index_name = index_name
        print(f"[Chatbot] Pinecone connected: {index_name}")
        return _pinecone_index
    except Exception as e:
        print(f"[Chatbot] Pinecone connection error: {e}")
        return None


# ══════════════════════════════════════════════════════
#  Embeddings (Google Generative AI)
# ══════════════════════════════════════════════════════


def get_embedding(text: str, api_key: str = None) -> List[float]:
    """Generate embedding for a query (RETRIEVAL_QUERY mode)."""
    if api_key:
        genai.configure(api_key=api_key)
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=768,
    )
    return result["embedding"]


def get_document_embedding(text: str, api_key: str = None) -> List[float]:
    """Generate embedding for document storage (RETRIEVAL_DOCUMENT mode)."""
    if api_key:
        genai.configure(api_key=api_key)
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=768,
    )
    return result["embedding"]


# ══════════════════════════════════════════════════════
#  Text Chunking (LangChain)
# ══════════════════════════════════════════════════════


def split_text_into_chunks(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> List[str]:
    """Split text into semantically meaningful chunks using recursive character splitting.

    Mimics LangChain's RecursiveCharacterTextSplitter strategy:
    - Tries splitting by paragraph breaks first, then newlines, sentences, phrases, words
    - Respects chunk_size and chunk_overlap for consistent chunk dimensions
    """
    separators = ["\n\n", "\n", ". ", ", ", " ", ""]
    return _recursive_split(text, separators, chunk_size, chunk_overlap)


def _recursive_split(
    text: str, separators: List[str], chunk_size: int, chunk_overlap: int
) -> List[str]:
    """Recursively split text by trying separators in order."""
    final_chunks: List[str] = []
    if not text:
        return final_chunks

    # Find the best separator that exists in the text
    separator = separators[-1]
    remaining_seps = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining_seps = separators[i + 1 :]
            break

    # Split the text by the chosen separator
    splits = text.split(separator) if separator else list(text)

    # Merge small pieces and recursively split large ones
    current_chunk = ""
    for piece in splits:
        candidate = (current_chunk + separator + piece) if current_chunk else piece
        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # Flush current chunk
            if current_chunk:
                final_chunks.append(current_chunk.strip())
            # If this piece alone exceeds chunk_size, try splitting further
            if len(piece) > chunk_size and remaining_seps:
                sub_chunks = _recursive_split(
                    piece, remaining_seps, chunk_size, chunk_overlap
                )
                final_chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = piece

    # Don't forget the last chunk
    if current_chunk and current_chunk.strip():
        final_chunks.append(current_chunk.strip())

    # Apply overlap: prepend tail of previous chunk to next chunk
    if chunk_overlap > 0 and len(final_chunks) > 1:
        overlapped = [final_chunks[0]]
        for i in range(1, len(final_chunks)):
            prev = final_chunks[i - 1]
            overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
            merged = (overlap_text + " " + final_chunks[i]).strip()
            overlapped.append(merged)
        final_chunks = overlapped

    return [c for c in final_chunks if c.strip()]


# ══════════════════════════════════════════════════════
#  LLM Response Generation (Multi-Provider)
# ══════════════════════════════════════════════════════


def _call_groq(llm_settings: dict, messages: list, temp: float, max_tok: int) -> str:
    """Call Groq API."""
    client = Groq(api_key=llm_settings["groq_api_key"])
    resp = client.chat.completions.create(
        model=llm_settings.get("groq_model", "llama-3.1-8b-instant"),
        messages=messages,
        temperature=temp,
        max_tokens=max_tok,
    )
    return resp.choices[0].message.content


def _call_google(
    llm_settings: dict, system_prompt: str, user_prompt: str, temp: float, max_tok: int
) -> str:
    """Call Google Generative AI API."""
    genai.configure(api_key=llm_settings["google_api_key"])
    model = genai.GenerativeModel(
        llm_settings.get("google_model", "gemini-1.5-flash"),
        system_instruction=system_prompt,
    )
    resp = model.generate_content(
        user_prompt,
        generation_config={
            "temperature": temp,
            "max_output_tokens": max_tok,
        },
    )
    return resp.text


async def _generate_llm_response(
    llm_settings: dict, system_prompt: str, user_prompt: str
) -> str:
    """Generate a response using Groq (primary) with Google AI fallback."""
    provider = llm_settings.get("active_provider", "groq")
    temp = float(llm_settings.get("temperature", 0.7))
    max_tok = int(llm_settings.get("max_tokens", 1000))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Try the active provider first
    if provider == "groq" and llm_settings.get("groq_api_key"):
        try:
            result = _call_groq(llm_settings, messages, temp, max_tok)
            print(f"[Chatbot] Groq responded successfully")
            return result
        except Exception as e:
            print(f"[Chatbot] Groq failed: {e}")
            # Fallback to Google
            if llm_settings.get("google_api_key"):
                print("[Chatbot] Falling back to Google AI...")
                return _call_google(
                    llm_settings, system_prompt, user_prompt, temp, max_tok
                )
            raise

    elif provider == "google" and llm_settings.get("google_api_key"):
        try:
            result = _call_google(
                llm_settings, system_prompt, user_prompt, temp, max_tok
            )
            print(f"[Chatbot] Google AI responded successfully")
            return result
        except Exception as e:
            print(f"[Chatbot] Google failed: {e}")
            # Fallback to Groq
            if llm_settings.get("groq_api_key"):
                print("[Chatbot] Falling back to Groq...")
                return _call_groq(llm_settings, messages, temp, max_tok)
            raise

    # If active provider has no key, try the other one
    if llm_settings.get("groq_api_key"):
        return _call_groq(llm_settings, messages, temp, max_tok)
    if llm_settings.get("google_api_key"):
        return _call_google(llm_settings, system_prompt, user_prompt, temp, max_tok)

    raise ValueError(
        "No LLM provider configured. Set Groq or Google API key in dashboard."
    )


# ══════════════════════════════════════════════════════
#  Vector Operations (Upload / Upsert)
# ══════════════════════════════════════════════════════


async def upsert_vectors(
    texts: List[str],
    metadata: List[dict] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
):
    """Upload text to Pinecone with LangChain chunking + store chunk info in MongoDB."""
    llm_settings = await get_llm_settings()
    pine_key = llm_settings.get("pinecone_api_key", "")
    pine_idx_name = llm_settings.get("pinecone_index_name", "portfolio-chatbot")
    emb_key = llm_settings.get("embedding_google_api_key", "")

    if not pine_key:
        raise Exception(
            "Pinecone API key not configured. Go to Chatbot → LLM Settings."
        )
    if not emb_key:
        raise Exception(
            "Google Embedding API key not configured. Go to Chatbot → LLM Settings."
        )

    idx = _get_pinecone_index(pine_key, pine_idx_name)
    if not idx:
        raise Exception("Could not connect to Pinecone index")

    # Use LangChain to split all input texts into smart chunks
    all_chunks = []
    for text in texts:
        chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise Exception("No valid text chunks produced from input")

    db = get_db()
    vectors = []
    chunk_docs = []

    for i, chunk_text in enumerate(all_chunks):
        chunk_id = f"chunk_{hashlib.md5(chunk_text.encode()).hexdigest()[:12]}_{i}"
        embedding = get_document_embedding(chunk_text, emb_key)

        meta = {
            "text": chunk_text,
            "source": "dashboard",
            "category": "portfolio_info",
        }
        if metadata and i < len(metadata):
            meta.update(metadata[i])
        meta["text"] = chunk_text  # always ensure text is in metadata

        vectors.append(
            {
                "id": chunk_id,
                "values": embedding,
                "metadata": meta,
            }
        )

        chunk_docs.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": meta.get("source", "dashboard"),
                "category": meta.get("category", "portfolio_info"),
                "created_at": datetime.utcnow().isoformat(),
            }
        )

    # Upsert to Pinecone in batches of 100
    for i in range(0, len(vectors), 100):
        batch = vectors[i : i + 100]
        idx.upsert(vectors=batch)

    # Store chunk metadata in MongoDB for dashboard management
    if chunk_docs:
        await db.chatbot_chunks.insert_many(chunk_docs)

    print(f"[Chatbot] Uploaded {len(all_chunks)} chunks to Pinecone + MongoDB")
    return len(all_chunks)


# ══════════════════════════════════════════════════════
#  Chat Query (RAG Pipeline)
# ══════════════════════════════════════════════════════


async def query_chatbot(message: str) -> dict:
    """Main chat function: embed query → search Pinecone → generate with selected LLM."""
    llm_settings = await get_llm_settings()

    # Check if at least one LLM provider is configured
    has_groq = bool(llm_settings.get("groq_api_key"))
    has_google = bool(llm_settings.get("google_api_key"))
    if not has_groq and not has_google:
        return {
            "response": (
                "No LLM provider is configured. "
                "Please set up Groq or Google API keys in the Admin Dashboard → Chatbot → LLM Settings."
            ),
            "sources": [],
        }

    try:
        # 1. Try to get context from Pinecone
        contexts = []
        sources = []
        pinecone_success = False

        pine_key = llm_settings.get("pinecone_api_key", "")
        emb_key = llm_settings.get("embedding_google_api_key", "")

        if pine_key and emb_key:
            try:
                idx = _get_pinecone_index(
                    pine_key,
                    llm_settings.get("pinecone_index_name", "portfolio-chatbot"),
                )
                if idx:
                    query_embedding = get_embedding(message, emb_key)
                    results = idx.query(
                        vector=query_embedding, top_k=5, include_metadata=True
                    )

                    for match in results.get("matches", []):
                        score = match.get("score", 0)
                        text = match.get("metadata", {}).get("text", "")
                        if text and score > 0.3:
                            contexts.append(text)
                            src = match.get("metadata", {}).get("source", "")
                            if src:
                                sources.append(src)

                    if contexts:
                        pinecone_success = True
                        print(
                            f"[Chatbot] Found {len(contexts)} relevant chunks from Pinecone"
                        )
            except Exception as e:
                print(f"[Chatbot] Pinecone query failed, using fallback: {e}")

        # 2. Fallback to resume data if Pinecone failed or had no results
        if not pinecone_success:
            print("[Chatbot] Using fallback resume data")
            contexts = FALLBACK_RESUME_CHUNKS
            sources = ["resume_fallback"]

        context_text = "\n\n".join(contexts)

        # 3. Load system prompt from site settings (if configured)
        system_prompt = DEFAULT_SYSTEM_PROMPT
        try:
            db = get_db()
            site_settings = await db.site_settings.find_one({})
            if site_settings and site_settings.get("chatbot_system_prompt"):
                system_prompt = site_settings["chatbot_system_prompt"]
        except Exception as e:
            print(f"[Chatbot] System prompt load error: {e}")

        # 4. Generate response using the selected LLM
        user_prompt = (
            f"Context about the portfolio owner:\n{context_text}\n\n"
            f"Question: {message}\n\n"
            f"Answer based only on the context above. Be helpful and concise:"
        )

        response_text = await _generate_llm_response(
            llm_settings, system_prompt, user_prompt
        )
        active_provider = llm_settings.get("active_provider", "groq")
        print(f"[Chatbot] Response ({active_provider}): {response_text[:80]}...")

        return {
            "response": response_text,
            "sources": list(set(sources)),
        }

    except Exception as e:
        print(f"[Chatbot] ERROR: {e}")
        traceback.print_exc()
        return {
            "response": "Sorry, I encountered an error processing your question. Please try again.",
            "sources": [],
        }


# ══════════════════════════════════════════════════════
#  Chunk Management
# ══════════════════════════════════════════════════════


async def list_chunks(skip: int = 0, limit: int = 50) -> dict:
    """List all chunks stored in MongoDB (mirrors Pinecone)."""
    db = get_db()
    total = await db.chatbot_chunks.count_documents({})
    cursor = db.chatbot_chunks.find({}).sort("created_at", -1).skip(skip).limit(limit)
    chunks = []
    async for doc in cursor:
        chunks.append(
            {
                "id": str(doc["_id"]),
                "chunk_id": doc.get("chunk_id", ""),
                "text": doc.get("text", ""),
                "source": doc.get("source", ""),
                "category": doc.get("category", ""),
                "created_at": doc.get("created_at", ""),
            }
        )
    return {"chunks": chunks, "total": total}


async def delete_chunk(chunk_id: str) -> bool:
    """Delete a single chunk from both Pinecone and MongoDB."""
    db = get_db()
    llm_settings = await get_llm_settings()

    # Delete from Pinecone
    pine_key = llm_settings.get("pinecone_api_key", "")
    if pine_key:
        try:
            idx = _get_pinecone_index(
                pine_key,
                llm_settings.get("pinecone_index_name", "portfolio-chatbot"),
            )
            if idx:
                idx.delete(ids=[chunk_id])
                print(f"[Chatbot] Deleted {chunk_id} from Pinecone")
        except Exception as e:
            print(f"[Chatbot] Pinecone delete error for {chunk_id}: {e}")

    # Delete from MongoDB
    result = await db.chatbot_chunks.delete_one({"chunk_id": chunk_id})
    return result.deleted_count > 0


async def delete_all_vectors():
    """Delete ALL vectors from Pinecone and all chunk records from MongoDB."""
    llm_settings = await get_llm_settings()
    pine_key = llm_settings.get("pinecone_api_key", "")

    if pine_key:
        idx = _get_pinecone_index(
            pine_key,
            llm_settings.get("pinecone_index_name", "portfolio-chatbot"),
        )
        if idx:
            idx.delete(delete_all=True)
            print("[Chatbot] Deleted all vectors from Pinecone")

    db = get_db()
    await db.chatbot_chunks.delete_many({})
    print("[Chatbot] Deleted all chunk records from MongoDB")
    return True


# ══════════════════════════════════════════════════════
#  Initialization (called at app startup)
# ══════════════════════════════════════════════════════


def init_chatbot():
    """Called at startup. Actual initialization is lazy (per-request from DB settings)."""
    print("[Chatbot] Multi-LLM chatbot service ready (lazy init from DB settings)")
