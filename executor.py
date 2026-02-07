import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import time
import re


@dataclass
class RemoteConfig:
    host: str
    user: str
    remote_tmp_screen_path: str = "/tmp/agent_screen.png"
    ssh_options: List[str] = None
    rate_limit_seconds: float = 5.0


class RemoteMacExecutor:
    def __init__(self, cfg: RemoteConfig):
        self.cfg = cfg
        self._last_exec_ts: float = 0.0

    def _ssh_base(self) -> List[str]:
        opts = self._expand_ssh_options(self.cfg.ssh_options or [])
        return ["ssh", *opts, f"{self.cfg.user}@{self.cfg.host}"]

    @staticmethod
    def _expand_ssh_options(opts: List[str]) -> List[str]:
        expanded: List[str] = []
        it = iter(opts)
        for opt in it:
            expanded.append(opt)
            if opt == "-i":
                try:
                    value = next(it)
                except StopIteration:
                    break
                expanded.append(str(Path(value).expanduser()))
        return expanded

    def _rate_limit(self, rate_limit_seconds: Optional[float] = None):
        seconds = self.cfg.rate_limit_seconds if rate_limit_seconds is None else rate_limit_seconds
        if seconds <= 0:
            self._last_exec_ts = time.time()
            return
        now = time.time()
        elapsed = now - self._last_exec_ts
        if elapsed < seconds:
            time.sleep(seconds - elapsed)
        self._last_exec_ts = time.time()

    def run_osascript(
        self,
        script: str,
        timeout: int = 20,
        rate_limit_seconds: Optional[float] = None,
    ) -> Tuple[int, str, str]:
        """
        Run AppleScript remotely via: ssh user@host osascript
        Script is passed through stdin (more robust than complex shell quoting).
        """
        self._rate_limit(rate_limit_seconds)

        proc = subprocess.run(
            [*self._ssh_base(), "osascript"],
            input=script,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

    def capture_screen_to_local(
        self,
        local_path: Path,
        timeout: int = 20,
        rate_limit_seconds: Optional[float] = None,
    ) -> Path:
        """
        1) remote: screencapture -x /tmp/...
        2) local: scp user@host:/tmp/... local_path
        """
        self._rate_limit(rate_limit_seconds)

        remote = self.cfg.remote_tmp_screen_path
        # 1) Remote screenshot
        subprocess.run(
            [*self._ssh_base(), f"screencapture -x {remote}"],
            check=True,
            timeout=timeout,
        )

        local_path.parent.mkdir(parents=True, exist_ok=True)

        # 2) SCP back
        scp_cmd = ["scp"]
        # reuse ssh options that are relevant to scp (-i, StrictHostKeyChecking, etc.)
        # scp supports -o too; easiest is to pass the same options.
        if self.cfg.ssh_options:
            scp_cmd.extend(self._expand_ssh_options(self.cfg.ssh_options))

        scp_cmd.extend([f"{self.cfg.user}@{self.cfg.host}:{remote}", str(local_path)])

        subprocess.run(
            scp_cmd,
            check=True,
            timeout=timeout,
        )
        # Best-effort cleanup on remote after successful transfer.
        subprocess.run(
            [*self._ssh_base(), f"rm -f {remote}"],
            timeout=timeout,
        )
        return local_path

    def get_screen_size_debug(
        self,
        timeout: int = 20,
        rate_limit_seconds: Optional[float] = None,
    ) -> Tuple[Optional[Tuple[int, int]], str, str, int]:
        """
        Return (size, stdout, stderr, returncode) for remote screen size.
        """
        self._rate_limit(rate_limit_seconds)
        script = 'osascript -e \'tell application "Finder" to get bounds of window of desktop\''
        proc = subprocess.run(
            [*self._ssh_base(), script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        size: Optional[Tuple[int, int]] = None
        if proc.returncode == 0:
            nums = [int(n) for n in re.findall(r"-?\d+", proc.stdout)]
            if len(nums) >= 4:
                left, top, right, bottom = nums[:4]
                width = max(0, right - left)
                height = max(0, bottom - top)
                if width > 0 and height > 0:
                    size = (width, height)
        return size, proc.stdout.strip(), proc.stderr.strip(), proc.returncode

    def get_screen_size(
        self,
        timeout: int = 20,
        rate_limit_seconds: Optional[float] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Return remote screen size in pixels as (width, height).
        Uses Finder desktop bounds via osascript.
        """
        size, _out, _err, _rc = self.get_screen_size_debug(
            timeout=timeout,
            rate_limit_seconds=rate_limit_seconds,
        )
        return size


class LocalMacExecutor:
    def __init__(self, rate_limit_seconds: float = 5.0):
        self.rate_limit_seconds = rate_limit_seconds
        self._last_exec_ts: float = 0.0

    def _rate_limit(self, rate_limit_seconds: Optional[float] = None):
        seconds = self.rate_limit_seconds if rate_limit_seconds is None else rate_limit_seconds
        if seconds <= 0:
            self._last_exec_ts = time.time()
            return
        now = time.time()
        elapsed = now - self._last_exec_ts
        if elapsed < seconds:
            time.sleep(seconds - elapsed)
        self._last_exec_ts = time.time()

    def run_osascript(
        self,
        script: str,
        timeout: int = 20,
        rate_limit_seconds: Optional[float] = None,
    ) -> Tuple[int, str, str]:
        """
        Run AppleScript locally via: osascript
        Script is passed through stdin to avoid shell quoting issues.
        """
        self._rate_limit(rate_limit_seconds)
        proc = subprocess.run(
            ["osascript"],
            input=script,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()

    def capture_screen_to_local(
        self,
        local_path: Path,
        timeout: int = 20,
        rate_limit_seconds: Optional[float] = None,
    ) -> Path:
        """
        Capture local screen to the given path via: screencapture -x
        """
        self._rate_limit(rate_limit_seconds)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["screencapture", "-x", str(local_path)],
            check=True,
            timeout=timeout,
        )
        return local_path

    def get_screen_size_debug(
        self,
        timeout: int = 20,
        rate_limit_seconds: Optional[float] = None,
    ) -> Tuple[Optional[Tuple[int, int]], str, str, int]:
        """
        Return (size, stdout, stderr, returncode) for local screen size.
        """
        self._rate_limit(rate_limit_seconds)
        script = 'tell application "Finder" to get bounds of window of desktop'
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        size: Optional[Tuple[int, int]] = None
        if proc.returncode == 0:
            nums = [int(n) for n in re.findall(r"-?\d+", proc.stdout)]
            if len(nums) >= 4:
                left, top, right, bottom = nums[:4]
                width = max(0, right - left)
                height = max(0, bottom - top)
                if width > 0 and height > 0:
                    size = (width, height)
        return size, proc.stdout.strip(), proc.stderr.strip(), proc.returncode

    def get_screen_size(
        self,
        timeout: int = 20,
        rate_limit_seconds: Optional[float] = None,
    ) -> Optional[Tuple[int, int]]:
        size, _out, _err, _rc = self.get_screen_size_debug(
            timeout=timeout,
            rate_limit_seconds=rate_limit_seconds,
        )
        return size
