from typing import Any

from modules.api.rest.ApiError import ApiError, InvalidConfigError
from modules.api.rest.ConfigSource import ConfigSource
from modules.api.rest.TrainingService import TrainingService
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig

from fastapi import APIRouter, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict


class SampleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample: dict[str, Any] | None = None


class RestApi:
    """
    The complete HTTP contract: eight endpoints over a TrainingService.

    This is an adapter, not a new abstraction. Every POST is a single call onto
    the TrainCommands object the trainer polls at step boundaries, and every
    field of the status response arrives from a TrainCallbacks hook. Commands
    answer 202 rather than 200 because none of them happen synchronously.

    It binds no socket and knows nothing about hosts or ports. install() adds
    the routes and the error contract to a FastAPI application somebody else
    owns, so the same contract can be served by scripts/train_rest_server.py or
    mounted inside a larger application.
    """

    def __init__(self, training_service: TrainingService, version: str = "unknown"):
        self.__training_service = training_service
        self.__version = version

        self.__router = APIRouter()
        self.__router.add_api_route("/health", self.health, methods=["GET"])
        self.__router.add_api_route("/config/defaults", self.config_defaults, methods=["GET"])
        self.__router.add_api_route("/training/status", self.training_status, methods=["GET"])
        self.__router.add_api_route("/training/start", self.training_start, methods=["POST"], status_code=202)
        self.__router.add_api_route("/training/stop", self.training_stop, methods=["POST"], status_code=202)
        self.__router.add_api_route("/training/sample", self.training_sample, methods=["POST"], status_code=202)
        self.__router.add_api_route("/training/backup", self.training_backup, methods=["POST"], status_code=202)
        self.__router.add_api_route("/training/save", self.training_save, methods=["POST"], status_code=202)

    # Adds the routes and the error contract to an application the caller owns.
    # prefix namespaces the routes for a host that already serves its own API.
    def install(self, app: FastAPI, prefix: str = "") -> None:
        app.include_router(self.__router, prefix=prefix)
        app.add_exception_handler(ApiError, self.__handle_api_error)
        app.add_exception_handler(RequestValidationError, self.__handle_validation_error)

    # --- endpoints ---
    # Handlers are plain `def`, not `async def`: FastAPI runs them in a
    # threadpool, so the config file reads never block the event loop.

    def health(self) -> dict:
        return {
            "status": "ok",
            "version": self.__version,
            "state": self.__training_service.status()["state"],
        }

    def config_defaults(self) -> dict:
        # to_settings_dict(secrets=False) is the existing secrets-stripping
        # serializer, so the no-secrets guarantee is enforced by core code.
        return TrainConfig.default_values().to_settings_dict(secrets=False)

    def training_status(self) -> dict:
        return self.__training_service.status()

    def training_start(self, body: ConfigSource) -> dict:
        run_id = self.__training_service.start(body.resolve())
        return {"run_id": run_id, "state": "starting"}

    def training_stop(self) -> dict:
        self.__training_service.stop()
        return {"accepted": True}

    def training_sample(self, body: SampleRequest | None = None) -> dict:
        sample = None
        if body is not None and body.sample is not None:
            try:
                sample = SampleConfig.default_values().from_dict(body.sample)
            except Exception as e:
                raise InvalidConfigError(f"Invalid sample config: {e}") from e
        self.__training_service.sample(sample)
        return {"accepted": True}

    def training_backup(self) -> dict:
        self.__training_service.backup()
        return {"accepted": True}

    def training_save(self) -> dict:
        self.__training_service.save()
        return {"accepted": True}

    # --- error contract ---

    @staticmethod
    def __handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.envelope())

    @staticmethod
    def __handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Override FastAPI's default 422 body so a client only ever parses one
        # error shape.
        error = InvalidConfigError("Request body is invalid", {"errors": jsonable_encoder(exc.errors())})
        return JSONResponse(status_code=error.status_code, content=error.envelope())
