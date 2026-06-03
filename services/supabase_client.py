import os
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_URL: str = (os.getenv("SUPABASE_URL") or "").strip().strip('"').strip("'")
SUPABASE_KEY: str = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip().strip('"').strip("'")



_client: Client = None

def get_supabase_client() -> Client:
    """Initialize and return the singleton Supabase client."""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.warning("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY missing from environment variables.")
        
        # Log details to debug DNS resolution / Errno -2 issues
        logger.info(f"Initializing Supabase client with SUPABASE_URL: {SUPABASE_URL!r} (len={len(SUPABASE_URL)})")
        if SUPABASE_KEY:
            logger.info(f"SUPABASE_KEY: len={len(SUPABASE_KEY)}, starts with {SUPABASE_KEY[:10]!r}, ends with {SUPABASE_KEY[-10:]!r}")
        else:
            logger.info("SUPABASE_KEY is empty")

        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client

def download_file(bucket_name: str, storage_path: str, local_path: str):
    """Download a file from Supabase Storage to local disk (e.g. /tmp)."""
    supabase = get_supabase_client()
    try:
        logger.info(f"Downloading {storage_path} from bucket '{bucket_name}' to {local_path}...")
        response = supabase.storage.from_(bucket_name).download(storage_path)
        with open(local_path, "wb") as f:
            f.write(response)
        logger.info("Download completed successfully.")
    except Exception as e:
        logger.error(f"Failed to download file from Supabase Storage: {e}")
        raise

def upload_file(bucket_name: str, storage_path: str, local_path: str, content_type: str) -> str:
    """Upload a local file to Supabase Storage and return its public URL."""
    supabase = get_supabase_client()
    try:
        logger.info(f"Uploading {local_path} to bucket '{bucket_name}' path '{storage_path}'...")
        with open(local_path, "rb") as f:
            supabase.storage.from_(bucket_name).upload(
                path=storage_path,
                file=f,
                file_options={"content-type": content_type, "x-upsert": "true"}
            )
        public_url = supabase.storage.from_(bucket_name).get_public_url(storage_path)
        logger.info(f"Upload completed. Public URL: {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"Failed to upload file to Supabase Storage: {e}")
        raise

def update_job(job_id: str, updates: dict):
    """Update a row in the Supabase 'jobs' table."""
    supabase = get_supabase_client()
    try:
        logger.info(f"Updating job {job_id} with: {updates}")
        supabase.table("jobs").update(updates).eq("id", job_id).execute()
    except Exception as e:
        logger.error(f"Failed to update job in Supabase: {e}")
        raise
