"""当前单 GPU 部署的串行调用、硬时限、显存准入与进程回收。"""

from contextlib import contextmanager
import os
import signal
import subprocess
import time


# 带 SAPIEN/Matplotlib 窗口实测进程峰值 1142 MiB，加 256 MiB 余量。
GRASPNET_MIN_FREE_VRAM = 1398


class ResourceManager:
    def __init__(self, deadline=None, notify=None):
        self.deadline = deadline
        self.notify = notify
        self.process = None
        self.busy = None
        self.previous_alarm = None
        self.report = {"grasp_required_free_mib": GRASPNET_MIN_FREE_VRAM, "events": []}

    def remaining(self, limit):
        seconds = limit if self.deadline is None else self.deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("task deadline exceeded")
        return min(seconds, limit)

    def __enter__(self):
        if self.deadline is not None:
            seconds = self.remaining(180)
            self.previous_alarm = (signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL))
            signal.signal(signal.SIGALRM, self._timed_out)
            signal.setitimer(signal.ITIMER_REAL, seconds)
        return self

    @staticmethod
    def _timed_out(signum, frame):
        raise TimeoutError("task deadline exceeded")

    @contextmanager
    def phase(self, name):
        if self.busy is not None:
            raise RuntimeError(f"resource phase already active: {self.busy}")
        event = {"phase": name, "status": "failed"}
        self.report["events"].append(event)
        self.busy = name
        started = time.monotonic()
        try:
            if self.notify is not None:
                self.notify(name, {})
            yield event
            event["status"] = "complete"
        finally:
            event["elapsed_s"] = time.monotonic() - started
            self.busy = None

    def call_vlm(self, function, *args, trial, stage, retry=False, **kwargs):
        with self.phase(stage):
            for attempt in range(1, 3 if retry else 2):
                trial[stage + "_requests"] = attempt
                try:
                    return function(*args, timeout=self.remaining(60), **kwargs)
                except ConnectionError:
                    if not retry or attempt == 2:
                        raise

    def _query(self, query):
        try:
            return subprocess.check_output(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                                           text=True, stderr=subprocess.DEVNULL, timeout=self.remaining(3))
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("GPU memory monitoring unavailable; inference refused") from exc

    def gpu_memory(self):
        # shortcut: 当前仅支持单 GPU，增加第二张显卡时必须按设备 UUID 选择。
        rows = self._query("--query-gpu=memory.total,memory.free").strip().splitlines()
        if len(rows) != 1:
            raise RuntimeError("GPU memory admission requires exactly one GPU")
        try:
            total, free = (int(value.strip()) for value in rows[0].split(","))
        except ValueError as exc:
            raise RuntimeError("GPU memory monitoring returned invalid data") from exc
        return {"total_mib": total, "free_mib": free}

    def process_memory(self, pid):
        for row in self._query("--query-compute-apps=pid,used_gpu_memory").splitlines():
            fields = row.split(",")
            if len(fields) == 2 and fields[0].strip() == str(pid):
                try:
                    return int(fields[1].strip())
                except ValueError as exc:
                    raise RuntimeError("GPU process memory unavailable") from exc
        return 0

    def run_grasp(self, command, log_path, cwd):
        with self.phase("grasp_inference") as event:
            event["gpu_before"] = self.gpu_memory()
            if event["gpu_before"]["free_mib"] < GRASPNET_MIN_FREE_VRAM:
                raise RuntimeError(f"insufficient free GPU memory: require {GRASPNET_MIN_FREE_VRAM} MiB")
            end = time.monotonic() + self.remaining(120)
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    self.process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                                    start_new_session=True)
                    event.update(pid=self.process.pid, process_peak_mib=0)
                    while self.process.poll() is None:
                        self.remaining(120)
                        if time.monotonic() >= end:
                            raise TimeoutError("GraspNet inference deadline exceeded")
                        event["process_peak_mib"] = max(event["process_peak_mib"], self.process_memory(self.process.pid))
                        try:
                            self.process.wait(timeout=min(0.2, max(0.001, end - time.monotonic())))
                        except subprocess.TimeoutExpired:
                            pass
                    event["returncode"] = self.process.returncode
                    if self.process.returncode:
                        message = log_path.read_text(encoding="utf-8", errors="replace").lower()
                        if "cuda" in message and "out of memory" in message:
                            raise RuntimeError("GraspNet CUDA OOM; task stopped")
                        raise RuntimeError(f"GraspNet exited {self.process.returncode}; inspect inference.log")
            except BaseException as exc:
                event["error_type"] = type(exc).__name__
                raise
            finally:
                self._stop_process()

    def _stop_process(self):
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
        finally:
            self.process = None

    def close(self):
        if self.previous_alarm is not None:
            handler, timer = self.previous_alarm
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, *timer)
            self.previous_alarm = None
        self._stop_process()

    def __exit__(self, *exc):
        self.close()
