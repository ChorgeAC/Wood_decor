"""S3 image store. The bucket is private; access is via short-lived presigned URLs only."""

from collections.abc import Mapping
from typing import Protocol

from botocore.exceptions import ClientError

from tdeez_ai.core.errors import NotFound, PayloadTooLarge, UpstreamError
from tdeez_ai.repositories.base import PresignedPut


class S3Api(Protocol):
    """The subset of the boto3 S3 client we use."""

    def generate_presigned_url(self, ClientMethod: str, **kwargs: object) -> str: ...

    def put_object(self, **kwargs: object) -> Mapping[str, object]: ...

    def get_object(self, **kwargs: object) -> Mapping[str, object]: ...

    def head_object(self, **kwargs: object) -> Mapping[str, object]: ...

    def delete_object(self, **kwargs: object) -> Mapping[str, object]: ...


class S3ImageStore:
    def __init__(self, client: S3Api, bucket: str) -> None:
        self._s3 = client
        self._bucket = bucket

    def presign_put(self, key: str, content_type: str, expires_s: int) -> PresignedPut:
        url = self._s3.generate_presigned_url(
            "put_object",
            Params={"Bucket": self._bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=expires_s,
        )
        return PresignedPut(url=url, headers={"Content-Type": content_type})

    def put_bytes(self, key: str, data: bytes, content_type: str) -> None:
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            ServerSideEncryption="AES256",
        )

    def get_bytes(self, key: str, max_bytes: int) -> bytes:
        try:
            head = self._s3.head_object(Bucket=self._bucket, Key=key)
            length = head.get("ContentLength")
            if isinstance(length, int) and length > max_bytes:
                raise PayloadTooLarge("That image is too large. Please use a smaller image.")
            body = self._s3.get_object(Bucket=self._bucket, Key=key).get("Body")
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise NotFound("Image not found. Please upload it again.") from exc
            raise UpstreamError("Storage is temporarily unavailable. Please try again.") from exc
        read = getattr(body, "read", None)
        data = read() if callable(read) else None
        if not isinstance(data, bytes):
            raise UpstreamError("Storage is temporarily unavailable. Please try again.")
        return data

    def delete(self, key: str) -> None:
        self._s3.delete_object(Bucket=self._bucket, Key=key)

    def presign_get(self, key: str, expires_s: int) -> str:
        return self._s3.generate_presigned_url(
            "get_object", Params={"Bucket": self._bucket, "Key": key}, ExpiresIn=expires_s
        )
