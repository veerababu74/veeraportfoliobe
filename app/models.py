from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime


# ─── Auth Models ───
class AdminLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ─── Profile / Hero Section ───
class Profile(BaseModel):
    full_name: str = ""
    tagline: str = ""  # e.g. "Full Stack Developer | ML Engineer"
    bio: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    profile_image_url: str = ""
    resume_download_url: str = ""  # Cloudinary link
    resume_view_url: str = ""  # Another link for viewing
    github_url: str = ""
    linkedin_url: str = ""
    twitter_url: str = ""
    website_url: str = ""
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    tagline: Optional[str] = None
    bio: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    profile_image_url: Optional[str] = None
    resume_download_url: Optional[str] = None
    resume_view_url: Optional[str] = None
    github_url: Optional[str] = None
    linkedin_url: Optional[str] = None
    twitter_url: Optional[str] = None
    website_url: Optional[str] = None


# ─── About Section ───
class About(BaseModel):
    title: str = "About Me"
    description: str = ""
    image_url: str = ""
    highlights: List[str] = []
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AboutUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    image_url: Optional[str] = None
    highlights: Optional[List[str]] = None


# ─── Skills ───
class Skill(BaseModel):
    name: str
    category: str  # e.g. "Frontend", "Backend", "DevOps", "ML"
    icon_url: str = ""
    proficiency: int = 80  # 0-100
    order: int = 0


class SkillCreate(BaseModel):
    name: str
    category: str
    icon_url: str = ""
    proficiency: int = 80
    order: int = 0


# ─── Experience ───
class Experience(BaseModel):
    company: str
    role: str
    description: str = ""
    start_date: str = ""
    end_date: str = ""  # "Present" if current
    is_current: bool = False
    technologies: List[str] = []
    company_logo_url: str = ""
    order: int = 0


class ExperienceCreate(BaseModel):
    company: str
    role: str
    description: str = ""
    start_date: str = ""
    end_date: str = ""
    is_current: bool = False
    technologies: List[str] = []
    company_logo_url: str = ""
    order: int = 0


# ─── Education ───
class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str = ""
    start_date: str = ""
    end_date: str = ""
    grade: str = ""
    description: str = ""
    logo_url: str = ""
    order: int = 0


class EducationCreate(BaseModel):
    institution: str
    degree: str
    field_of_study: str = ""
    start_date: str = ""
    end_date: str = ""
    grade: str = ""
    description: str = ""
    logo_url: str = ""
    order: int = 0


# ─── Projects ───
class Project(BaseModel):
    title: str
    description: str = ""
    long_description: str = ""
    image_url: str = ""
    technologies: List[str] = []
    github_url: str = ""
    live_url: str = ""
    category: str = ""  # e.g. "Web", "ML", "Mobile"
    featured: bool = False
    order: int = 0


class ProjectCreate(BaseModel):
    title: str
    description: str = ""
    long_description: str = ""
    image_url: str = ""
    technologies: List[str] = []
    github_url: str = ""
    live_url: str = ""
    category: str = ""
    featured: bool = False
    order: int = 0


# ─── Certifications ───
class Certification(BaseModel):
    title: str
    issuer: str
    date: str = ""
    credential_url: str = ""
    image_url: str = ""
    order: int = 0


class CertificationCreate(BaseModel):
    title: str
    issuer: str
    date: str = ""
    credential_url: str = ""
    image_url: str = ""
    order: int = 0


# ─── Testimonials ───
class Testimonial(BaseModel):
    name: str
    role: str = ""
    company: str = ""
    content: str
    image_url: str = ""
    order: int = 0


class TestimonialCreate(BaseModel):
    name: str
    role: str = ""
    company: str = ""
    content: str
    image_url: str = ""
    order: int = 0


# ─── Contact Messages (from visitors) ───
class ContactMessage(BaseModel):
    name: str
    email: str
    subject: str = ""
    message: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_read: bool = False


class ContactMessageCreate(BaseModel):
    name: str
    email: str
    subject: str = ""
    message: str


# ─── Site Settings ───
class SiteSettings(BaseModel):
    site_title: str = "Portfolio"
    meta_description: str = ""
    favicon_url: str = ""
    primary_color: str = "#6C63FF"
    secondary_color: str = "#FF6584"
    dark_mode_default: bool = True
    footer_text: str = ""
    chatbot_name: str = "Portfolio Assistant"
    chatbot_avatar_url: str = ""
    chatbot_intro_message: str = (
        "Hi! I'm the portfolio assistant. Ask me anything about the portfolio owner!"
    )
    chatbot_system_prompt: str = (
        "You are a helpful portfolio assistant. You answer questions about the portfolio owner based ONLY on the provided context. "
        "If the context doesn't contain relevant information, politely say you don't have that information. "
        "Be concise, friendly, and professional. Keep answers focused and relevant."
    )
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SiteSettingsUpdate(BaseModel):
    site_title: Optional[str] = None
    meta_description: Optional[str] = None
    favicon_url: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    dark_mode_default: Optional[bool] = None
    footer_text: Optional[str] = None
    chatbot_name: Optional[str] = None
    chatbot_avatar_url: Optional[str] = None
    chatbot_intro_message: Optional[str] = None
    chatbot_system_prompt: Optional[str] = None


# ─── LLM Settings ───
class LLMSettings(BaseModel):
    active_provider: str = "groq"  # groq or google
    # Groq (primary)
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    groq_models: List[str] = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    # Google (fallback LLM + embeddings)
    google_api_key: str = ""
    google_model: str = "gemini-1.5-flash"
    google_models: List[str] = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
    ]
    # System Prompt
    system_prompt: str = ""
    # Common
    temperature: float = 0.7
    max_tokens: int = 1000
    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "portfolio-chatbot"
    # Embedding (Google API key for generating vectors)
    embedding_google_api_key: str = ""
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class LLMSettingsUpdate(BaseModel):
    active_provider: Optional[str] = None
    groq_api_key: Optional[str] = None
    groq_model: Optional[str] = None
    groq_models: Optional[List[str]] = None
    google_api_key: Optional[str] = None
    google_model: Optional[str] = None
    google_models: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    pinecone_api_key: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    embedding_google_api_key: Optional[str] = None


# ─── Chatbot ───
class ChatMessage(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []


class VectorUpload(BaseModel):
    texts: List[str]
    metadata: List[dict] = []
    chunk_size: int = 500
    chunk_overlap: int = 50
