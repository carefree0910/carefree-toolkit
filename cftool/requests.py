from typing import Any
from typing import Dict
from typing import Callable
from typing import Optional
from dataclasses import dataclass

from .misc import retry

try:
    from aiohttp import ClientSession
    from aiohttp import ClientResponse
    from aiohttp.typedefs import LooseHeaders
except:
    ClientSession = None


@dataclass
class RequestsConfig:
    headers: Optional[LooseHeaders] = None
    params: Optional[Dict[str, str]] = None
    data: Any = None
    json: Any = None


async def to_json(res: ClientResponse) -> Dict[str, Any]:
    json_res = await res.json()
    await res.release()
    return json_res


class Requests:
    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        before_request: Optional[Callable[[RequestsConfig], RequestsConfig]] = None,
        request_error: Optional[Callable[[Exception], None]] = None,
        before_response: Optional[Callable[[ClientResponse], ClientResponse]] = None,
        response_error: Optional[Callable[[ClientResponse], ClientResponse]] = None,
    ) -> None:
        self.sess = ClientSession(base_url)
        self.retry = retry
        self.before_request = before_request
        self.request_error = request_error
        self.before_response = before_response
        self.response_error = response_error

    # apis

    async def get(
        self,
        endpoint: str,
        config: Optional[RequestsConfig] = None,
    ) -> ClientResponse:
        return await self._request("get", endpoint, config)

    async def post(
        self,
        endpoint: str,
        config: Optional[RequestsConfig] = None,
    ) -> ClientResponse:
        return await self._request("post", endpoint, config)

    async def get_json(
        self,
        endpoint: str,
        config: Optional[RequestsConfig] = None,
        *,
        num_retry: Optional[int] = None,
        health_check: Optional[Callable[[Dict[str, Any]], bool]] = None,
        error_verbose_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        return await self._request_json(
            "get",
            endpoint,
            config,
            num_retry=num_retry,
            health_check=health_check,
            error_verbose_fn=error_verbose_fn,
        )

    async def post_json(
        self,
        endpoint: str,
        config: Optional[RequestsConfig] = None,
        *,
        num_retry: Optional[int] = None,
        health_check: Optional[Callable[[Dict[str, Any]], bool]] = None,
        error_verbose_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        return await self._request_json(
            "post",
            endpoint,
            config,
            num_retry=num_retry,
            health_check=health_check,
            error_verbose_fn=error_verbose_fn,
        )

    async def close(self) -> None:
        await self.sess.close()

    # internal

    async def _request(
        self,
        method: str,
        endpoint: str,
        config: Optional[RequestsConfig] = None,
    ) -> ClientResponse:
        config = self._intercept_config(config)
        try:
            res = await self.sess.request(method, endpoint, **config.__dict__)
            if res.ok:
                return self._intercept_response(res)
            return self._intercept_response_error(res)
        except Exception as e:
            self._intercept_request_error(e)

    async def _request_json(
        self,
        method: str,
        endpoint: str,
        config: Optional[RequestsConfig] = None,
        *,
        num_retry: Optional[int],
        health_check: Optional[Callable[[Dict[str, Any]], bool]],
        error_verbose_fn: Optional[Callable[[Dict[str, Any]], None]],
    ) -> Dict[str, Any]:
        async def _req() -> Dict[str, Any]:
            return await to_json(await self._request(method, endpoint, config))

        return await retry(
            _req,
            num_retry=num_retry,
            health_check=health_check,
            error_verbose_fn=error_verbose_fn,
        )

    def _intercept_config(self, c: Optional[RequestsConfig] = None) -> RequestsConfig:
        if c is None:
            c = RequestsConfig()
        if self.before_request is not None:
            c = self.before_request(c)
        return c

    def _intercept_request_error(self, e: Exception) -> None:
        if self.request_error is not None:
            self.request_error(e)
        raise e

    def _intercept_response(self, r: ClientResponse) -> ClientResponse:
        if self.before_response is None:
            return r
        return self.before_response(r)

    def _intercept_response_error(self, r: ClientResponse) -> ClientResponse:
        if self.response_error is None:
            return r
        return self.response_error(r)


__all__ = [
    "Requests",
    "RequestsConfig",
]
