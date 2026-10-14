"""SES v2 mailer."""

from collections.abc import Mapping
from typing import Protocol

from botocore.exceptions import BotoCoreError, ClientError

from tdeez_ai.core.errors import UpstreamError


class SesApi(Protocol):
    def send_email(self, **kwargs: object) -> Mapping[str, object]: ...


class SesMailer:
    def __init__(self, client: SesApi, sender: str) -> None:
        self._ses = client
        self._sender = sender

    def send(self, to: str, subject: str, body: str) -> None:
        try:
            self._ses.send_email(
                FromEmailAddress=self._sender,
                Destination={"ToAddresses": [to]},
                Content={
                    "Simple": {
                        "Subject": {"Data": subject, "Charset": "UTF-8"},
                        "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
                    }
                },
            )
        except (ClientError, BotoCoreError) as exc:
            raise UpstreamError("Email could not be sent right now.") from exc
