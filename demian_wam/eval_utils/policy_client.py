"""WebSocket client for communicating with a WAM policy server.

Adapted from DreamZero's eval_utils/policy_client.py.
"""

import logging
import time
from typing import Dict, Tuple

import websockets.sync.client
from typing_extensions import override

from openpi_client.base_policy import BasePolicy
from openpi_client import msgpack_numpy

# Long timeout for first inference (torch.compile can take >10min)
PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 3600


class WebsocketClientPolicy(BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        try:
            conn = websockets.sync.client.connect(
                self._uri,
                compression=None,
                max_size=None,
                ping_interval=PING_INTERVAL_SECS,
                ping_timeout=PING_TIMEOUT_SECS,
            )
            metadata = msgpack_numpy.unpackb(conn.recv())
            return conn, metadata
        except Exception:
            logging.info("Connection to server with ws:// failed. Trying wss:// ...")

        self._uri = "wss://" + self._uri.split("//")[1]
        conn = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata

    @override
    def infer(self, obs: Dict) -> Dict:
        obs["endpoint"] = "infer"
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self, reset_info: Dict) -> None:
        reset_info["endpoint"] = "reset"
        data = self._packer.pack(reset_info)
        self._ws.send(data)
        response = self._ws.recv()
        return response
