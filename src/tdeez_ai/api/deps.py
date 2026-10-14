"""Auth dependency: identity comes from the verified token only."""

from typing import Annotated, cast

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from tdeez_ai.container import Container
from tdeez_ai.core.auth import Principal
from tdeez_ai.core.errors import Unauthorized

_bearer = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    return cast(Container, request.app.state.container)


def current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    if credentials is None or not credentials.credentials:
        raise Unauthorized("Please sign in to continue.")
    return get_container(request).verifier.verify(credentials.credentials)


UserDep = Annotated[Principal, Depends(current_user)]
ContainerDep = Annotated[Container, Depends(get_container)]
