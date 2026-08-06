class ApiError(Exception):
    """
    Base for every error the REST API reports.

    Subclasses set error_type/status_code. RestApi installs a single handler
    that renders any ApiError through envelope(), so a client only ever has to
    parse one error shape.
    """

    error_type = "internal"
    status_code = 500

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def envelope(self) -> dict:
        return {"error": {"type": self.error_type, "message": self.message, "details": self.details}}


class InvalidConfigError(ApiError):
    error_type = "invalid_config"
    status_code = 422


class ConflictError(ApiError):
    error_type = "conflict"
    status_code = 409

    def __init__(self, message: str, run_id: str):
        super().__init__(message, {"run_id": run_id})


class NoActiveRunError(ApiError):
    error_type = "no_active_run"
    status_code = 409
