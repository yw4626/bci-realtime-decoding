from __future__ import annotations

import logging
import os
import uuid

logger = logging.getLogger(__name__)


def localize_gcs_uri(uri: str, cache_dir: str) -> str:
    """Return local path; download once if ``uri`` is ``gs://``."""
    if not uri.startswith("gs://"):
        return uri
    os.makedirs(cache_dir, exist_ok=True)
    rest = uri[5:]
    bucket_name, _, blob_path = rest.partition("/")
    safe = blob_path.replace("/", "__") or "object"
    local = os.path.join(cache_dir, safe)
    if os.path.isfile(local):
        return local
    try:
        from google.cloud import storage
    except ImportError as e:
        raise ImportError("读取 gs:// 模型需要安装 google-cloud-storage") from e
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    tmp = os.path.join(cache_dir, f".{safe}.part.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        blob.download_to_filename(tmp)
        os.replace(tmp, local)
    except Exception:
        if os.path.isfile(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    logger.info("Downloaded %s -> %s", uri, local)
    return local
