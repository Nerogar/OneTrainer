import os

from fastapi import HTTPException

_BLOCKED_NAMES = {
    ".git",
    ".env",
    "__pycache__",
    "secrets.json",
    "node_modules",
    ".ssh",
    ".aws",
    ".azure",
    ".gcloud",
    ".gnupg",
    ".gnome-keyring",
    ".kube",
    ".docker",
    ".npmrc",
    ".netrc",
    ".pgpass",
    ".password-store",
    ".git-credentials",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "known_hosts",
    "authorized_keys",
    "shadow",
    "passwd",
    "sam",
    "system",
    "ntds.dit",
    "wallet.dat",
    ".bash_history",
    ".zsh_history",
    ".python_history",
}

_BLOCKED_PREFIXES = (
    ".env",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
)

_BLOCKED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".key",
    ".pem",
    ".ppk",
    ".pfx",
    ".p12",
    ".p8",
    ".jks",
    ".keystore",
    ".asc",
    ".gpg",
    ".kdbx",
    ".ovpn",
}


def base_match(canonical: str, base: str) -> bool:
    canonical = os.path.normpath(canonical)
    base = os.path.normpath(base)

    if os.name == "nt":
        canonical = canonical.lower()
        base = base.lower()

    base_clean = base.rstrip(os.sep)
    canon_clean = canonical.rstrip(os.sep)

    if canon_clean == base_clean:
        return True

    return canonical.startswith(base_clean + os.sep)


def validate_path(
    user_path: str,
    *,
    must_exist: bool = True,
    allow_file: bool = True,
    allow_dir: bool = True,
) -> str:
    if not user_path or not user_path.strip():
        raise HTTPException(status_code=400, detail="Empty path")

    try:
        canonical = os.path.realpath(user_path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid path: {exc}") from exc

    if must_exist:
        if not os.path.exists(canonical):
            raise HTTPException(status_code=404, detail="Path not found")
        if not allow_file and os.path.isfile(canonical):
            raise HTTPException(status_code=400, detail="Expected directory, got file")
        if not allow_dir and os.path.isdir(canonical):
            raise HTTPException(status_code=400, detail="Expected file, got directory")

    parts = os.path.normpath(canonical).split(os.sep)
    for part in parts:
        part_lower = part.lower()
        if part_lower in _BLOCKED_NAMES or part_lower.startswith(_BLOCKED_PREFIXES):
            raise HTTPException(
                status_code=403,
                detail=f"Access denied: path contains restricted component '{part}'",
            )

    _, ext = os.path.splitext(canonical)
    if ext.lower() in _BLOCKED_SUFFIXES:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: file type '{ext}' is restricted",
        )

    return canonical
