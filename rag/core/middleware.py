import time
import uuid
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from rag.core.logging import get_logger

logger = get_logger(__name__)
REQUEST_ID_HEADER = "x-request-id"

class RequestIDMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable):
        req_id = request.headers.get(REQUEST_ID_HEADER, str(uuid.uuid4()))
        request.state.request_id = req_id
        response: Response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = req_id
        return response


class AccessLogMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable):
        start = time.perf_counter()
        path = request.url.path
        method = request.method
        client = request.client.host if request.client else "unknown"

        # Avoid logging sensitive headers; do NOT log API key
        try:
            response: Response = await call_next(request)
            status = response.status_code
        except Exception:
            # duration still useful on error
            duration_ms = int((time.perf_counter() - start) * 1000)
            logger.exception(f"{method} {path} -> 500 in {duration_ms}ms | client={client}")
            raise

        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(f"{method} {path} -> {status} in {duration_ms}ms | client={client}")
        return response
