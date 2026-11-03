"""Append-only episode trace. Images are content-addressed, never truncated in logs."""
import base64
import hashlib
import json
from pathlib import Path
import threading
import time


class EventLog(list):
    def __init__(self, directory=None, metadata=None, retain=True):
        super().__init__()
        self.retain = retain
        self.sequence = 0
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.directory = Path(directory) if directory is not None else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=False)
            (self.directory / "images").mkdir()
        self.append({"event": "metadata", "metadata": metadata or {}})

    def _externalize(self, value):
        if isinstance(value, dict):
            output = {}
            for key, item in value.items():
                if key in {"images", "start_images"} and isinstance(item, list):
                    output[key] = []
                    for encoded in item:
                        raw = base64.b64decode(encoded, validate=True)
                        name = hashlib.sha256(raw).hexdigest() + ".jpg"
                        path = self.directory / "images" / name
                        if not path.exists():
                            path.write_bytes(raw)
                        output[key].append("images/" + name)
                else:
                    output[key] = self._externalize(item)
            return output
        if isinstance(value, list):
            return [self._externalize(x) for x in value]
        return value

    def append(self, event):
        with self.lock:
            event = {**event, "sequence": self.sequence, "wall_time": time.time(),
                     "elapsed_seconds": time.monotonic() - self.started}
            if self.directory:
                event = self._externalize(event)
                with (self.directory / "events.jsonl").open("a") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                with (self.directory / "timeline.md").open("a") as stream:
                    stream.write(f"\n### {event['sequence']} · {event['event']} · step {event.get('step', '-')}\n\n")
                    stream.write("```json\n" + json.dumps(event, ensure_ascii=False, indent=2) + "\n```\n")
            self.sequence += 1
            if self.retain:
                super().append(event)
