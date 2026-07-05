import re
import time
import urllib.error
import urllib.request
from collections import OrderedDict

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

router = APIRouter(prefix="/wiki", tags=["wiki"])

WIKI_SECTIONS: list[dict] = [
    {
        "title": "Getting Started",
        "pages": [
            "Home",
            "Onboarding-Guide-for-Newcomers",
            "The-Program",
            "Model-Support-Overview",
            "Diffusion-Models-Overview",
        ],
    },
    {
        "title": "Configuration",
        "pages": [
            "General",
            "Model",
            "Data",
            "Concepts",
            "Aspect-Ratio-Bucketing",
            "How-to-setup-and-evaluate-validation-datasets",
            "Prior-Prediction",
            "How-Validation-works",
        ],
    },
    {
        "title": "Training",
        "pages": [
            "Training",
            "Optimizers",
            "Advanced-Optimizers",
            "Orthogonal-Optimizers",
            "Custom-Scheduler",
            "Quantization",
        ],
    },
    {
        "title": "Output & Tools",
        "pages": [
            "Sampling",
            "Backup-and-Save",
            "Tools",
        ],
    },
    {
        "title": "Methods",
        "pages": [
            "LoRA",
            "Embedding",
            "Additional-Embeddings",
        ],
    },
    {
        "title": "Model Guides",
        "pages": [
            "Flux",
            "Chroma",
            "Qwen-Image",
        ],
    },
    {
        "title": "Cloud & Remote",
        "pages": [
            "Cloud-Training",
            "Manually-setup-OneTrainer-in-Runpod",
            "Training-on-a-remote-Linux-Server",
        ],
    },
    {
        "title": "Guides & FAQ",
        "pages": [
            "F.A.Q.",
            "Lessons-Learnt-and-Tutorials",
            "Common-Mistakes-Coming-From-Kohya",
            "OneTrainer-March-2024-Guide",
        ],
    },
]

_ALL_SLUGS: set[str] = set()
for _section in WIKI_SECTIONS:
    for _page in _section["pages"]:
        _ALL_SLUGS.add(_page)

_cache: dict[str, tuple[str, float]] = {}
_CACHE_TTL = 3600

_RAW_WIKI_BASE = "https://raw.githubusercontent.com/wiki/Nerogar/OneTrainer"


def _fetch_wiki_page(slug: str) -> str | None:
    now = time.time()

    if slug in _cache:
        content, cached_at = _cache[slug]
        if now - cached_at < _CACHE_TTL:
            return content

    url = f"{_RAW_WIKI_BASE}/{slug}.md"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OneTrainerWeb/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode("utf-8")
            _cache[slug] = (content, now)
            return content
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        if slug in _cache:
            return _cache[slug][0]
        return None


def _rewrite_image_urls(content: str) -> str:
    def _rewrite_md_img(m: re.Match) -> str:
        alt, url = m.group(1), m.group(2)
        if url.startswith(("http://", "https://")):
            proxy_url = f"/api/wiki/image?url={urllib.request.quote(url, safe='')}"
            return f"![{alt}]({proxy_url})"
        return m.group(0)

    content = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _rewrite_md_img, content)

    def _rewrite_html_img(m: re.Match) -> str:
        url = m.group(1)
        if url.startswith(("http://", "https://")):
            proxy_url = f"/api/wiki/image?url={urllib.request.quote(url, safe='')}"
            return f'src="{proxy_url}"'
        return m.group(0)

    content = re.sub(r'src="([^"]+)"', _rewrite_html_img, content)

    return content


_MAX_IMAGE_CACHE = 100
_MAX_IMAGE_SIZE = 10 * 1024 * 1024
_image_cache: OrderedDict[str, tuple[str, bytes, float]] = OrderedDict()
_IMAGE_CACHE_TTL = 3600

_ALLOWED_IMAGE_PREFIXES = (
    "https://github.com/",
    "https://raw.githubusercontent.com/",
    "https://user-images.githubusercontent.com/",
)

_ALLOWED_REDIRECT_PREFIXES = _ALLOWED_IMAGE_PREFIXES + (
    "https://github-production-user-asset-",
    "https://objects.githubusercontent.com/",
)


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith(_ALLOWED_REDIRECT_PREFIXES):
            raise urllib.error.URLError(f"Redirect to disallowed domain: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_SafeRedirectHandler)


@router.get("/image")
def proxy_wiki_image(url: str) -> Response:
    now = time.time()

    if not url.startswith(_ALLOWED_IMAGE_PREFIXES):
        return Response(status_code=403, content="Forbidden: only GitHub image URLs are allowed")

    if url in _image_cache:
        content_type, data, cached_at = _image_cache[url]
        if now - cached_at < _IMAGE_CACHE_TTL:
            _image_cache.move_to_end(url)
            return Response(content=data, media_type=content_type)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OneTrainerWeb/1.0"})
        with _opener.open(req, timeout=15) as response:
            content_length = response.headers.get("Content-Length")
            try:
                content_length_int = int(content_length) if content_length is not None else None
            except ValueError:
                content_length_int = None
            if content_length_int is not None and content_length_int > _MAX_IMAGE_SIZE:
                return Response(status_code=413, content="Image too large (exceeds 10 MB limit)")

            data = response.read(_MAX_IMAGE_SIZE + 1)
            if len(data) > _MAX_IMAGE_SIZE:
                return Response(status_code=413, content="Image too large (exceeds 10 MB limit)")

            content_type = response.headers.get("Content-Type", "image/png")

            if not content_type.split(";")[0].strip().startswith("image/"):
                return Response(status_code=415, content="Upstream Content-Type is not an image")

            _image_cache[url] = (content_type, data, now)
            _image_cache.move_to_end(url)
            while len(_image_cache) > _MAX_IMAGE_CACHE:
                _image_cache.popitem(last=False)
            return Response(content=data, media_type=content_type)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        if url in _image_cache:
            content_type, data, _ = _image_cache[url]
            if content_type.split(";")[0].strip().startswith("image/"):
                return Response(content=data, media_type=content_type)
        return Response(status_code=502, content="Failed to fetch image")


@router.get("/pages")
def list_wiki_pages() -> list[dict]:
    return WIKI_SECTIONS


@router.get("/pages/{slug:path}")
def get_wiki_page(slug: str) -> dict[str, str]:
    if slug not in _ALL_SLUGS:
        raise HTTPException(status_code=404, detail=f"Unknown wiki page: {slug}")

    content = _fetch_wiki_page(slug)
    if content is None:
        content = (
            f"# {slug.replace('-', ' ')}\n\n"
            "This page could not be loaded from the OneTrainer wiki at this time. "
            "Please check your internet connection and try again, or visit the wiki directly at "
            f"[GitHub Wiki](https://github.com/Nerogar/OneTrainer/wiki/{slug})."
        )

    content = _rewrite_image_urls(content)

    return {"slug": slug, "content": content}
