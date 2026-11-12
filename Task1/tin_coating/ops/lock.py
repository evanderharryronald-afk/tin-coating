"""跨平台文件锁。

用途：定时任务（quality/health/retrain）与人工操作可能同时触发，
必须保证同一时刻只有一个进程在写 registry / 产出模型，避免互相覆盖。

Windows 用 msvcrt.locking，POSIX 用 fcntl.flock；都是非阻塞尝试 + 超时重试。
"""
from __future__ import annotations

import os
import time
from pathlib import Path

_IS_WINDOWS = os.name == "nt"


class LockTimeout(RuntimeError):
    pass


class FileLock:
    def __init__(self, path: str | Path, timeout: float = 10.0, poll: float = 0.05):
        self.path = Path(path)
        self.timeout = float(timeout)
        self.poll = float(poll)
        self._fh = None

    def _try_acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            if _IS_WINDOWS:
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fh = fh
            return True
        except OSError:
            fh.close()
            return False

    def acquire(self):
        deadline = time.time() + self.timeout
        while True:
            if self._try_acquire():
                try:
                    self._fh.seek(0)
                    self._fh.truncate()
                    self._fh.write(f"{os.getpid()}\n".encode())
                    self._fh.flush()
                except Exception:
                    pass
                return self
            if time.time() >= deadline:
                raise LockTimeout(f"等待文件锁超时: {self.path}")
            time.sleep(self.poll)

    def release(self):
        if self._fh is None:
            return
        try:
            if _IS_WINDOWS:
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False
