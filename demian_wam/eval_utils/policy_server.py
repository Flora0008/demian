"""WebSocket policy server for serving WAM policies.

Adapted from DreamZero's eval_utils/policy_server.py.
Compatible with the openpi_client interface used by RoboCasa evaluation.
"""

import asyncio
import dataclasses
import logging
import traceback

from openpi_client.base_policy import BasePolicy
from openpi_client import msgpack_numpy
import websockets.asyncio.server
import websockets.frames


@dataclasses.dataclass
class PolicyServerConfig:
    # Resolution that images get resized to client-side, None means no resizing.
    image_resolution: tuple[int, int] | None = (224, 224)
    # Whether or not wrist camera image(s) should be sent.
    needs_wrist_camera: bool = True
    # Number of external cameras to send.
    n_external_cameras: int = 2  # RoboCasa has 2 agentview cameras
    # Whether or not stereo camera image(s) should be sent.
    needs_stereo_camera: bool = False
    # Whether or not the unique eval session id should be sent.
    needs_session_id: bool = True
    # Which action space to use.
    action_space: str = "cartesian_position"


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol."""

    def __init__(
        self,
        policy: BasePolicy,
        server_config: PolicyServerConfig,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self._policy = policy
        self._server_config = server_config
        self._host = host
        self._port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_interval=None,
            ping_timeout=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # Send server config to client
        await websocket.send(packer.pack(dataclasses.asdict(self._server_config)))

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())

                endpoint = obs.pop("endpoint", "infer")
                if endpoint == "reset":
                    self._policy.reset(obs)
                    to_return = "reset successful"
                else:
                    action = self._policy.infer(obs)
                    to_return = packer.pack(action)
                await websocket.send(to_return)
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
