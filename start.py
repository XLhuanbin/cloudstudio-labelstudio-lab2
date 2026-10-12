"""Run Label Studio and its ML backend in one Cloud Studio workspace."""
from pathlib import Path
import argparse
import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def prepare(state):
    data = ROOT / "实验2数据"
    if not data.is_dir():
        print("未找到实验2数据：可先打开标注页面；上传数据后重启以生成任务。", flush=True)
        return False
    subprocess.run([sys.executable, "-B", str(ROOT / "源码/build_tasks.py"),
                    "--data-root", str(data), "--output", str(state / "tasks.json")],
                   cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-B", str(ROOT / "源码/evaluate_annotations.py"),
                    "--tasks", str(state / "tasks.json"),
                    "--report", str(state / "task_check.json")], cwd=ROOT, check=True)
    return True


def require_port(host, port):
    with socket.socket() as probe:
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is unavailable; stop the existing service or change the port.") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("APP_PORT", "8080")))
    parser.add_argument("--backend-port", type=int, default=int(os.getenv("ML_BACKEND_PORT", "9090")))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--state-dir", type=Path,
                        default=Path(os.getenv("APP_STATE_DIR", str(ROOT / ".runtime"))))
    args = parser.parse_args()
    state = args.state_dir.resolve()
    state.mkdir(parents=True, exist_ok=True)
    prepared = prepare(state)
    if args.prepare_only:
        return 0 if prepared else 1
    if args.port == args.backend_port:
        parser.error("Label Studio and ML backend must use different ports")
    for module in ("label_studio", "label_studio_ml"):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError("Missing dependencies: run bash setup.sh first.")
    require_port("0.0.0.0", args.port)
    require_port("127.0.0.1", args.backend_port)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["USE_REDIS"] = "false"
    env["LABEL_STUDIO_BASE_DATA_DIR"] = str(state / "labelstudio")
    env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(ROOT)
    env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LABEL_STUDIO_URL"] = f"http://127.0.0.1:{args.port}"
    # Label Studio 1.13 reads ML_TIMEOUT_PREDICT, not LABEL_STUDIO_ML_TIMEOUT.
    env.setdefault("ML_TIMEOUT_PREDICT", "3600")
    env["ML_PREDICTION_CACHE_DIR"] = str(state / "predictions")
    # Empty overrides in .env must not override Label Studio's default host or login.
    for key in ("LABEL_STUDIO_HOST", "LABEL_STUDIO_USERNAME", "LABEL_STUDIO_PASSWORD"):
        if not env.get(key):
            env.pop(key, None)
    if env.get("LABEL_STUDIO_USERNAME") and env.get("LABEL_STUDIO_PASSWORD"):
        env["LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK"] = "true"
    commands = [
        ("ml_backend", [sys.executable, "-u", str(ROOT / "源码/start_backend.py"),
                        "--host", "127.0.0.1", "--port", str(args.backend_port),
                        "--model-dir", str(state / "backend")]),
        ("labelstudio", [sys.executable, "-u", "-m", "label_studio.server", "start",
                         "--internal-host", "0.0.0.0", "--port", str(args.port), "--no-browser"]),
    ]
    children, logs = [], []

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, stop)
    try:
        for name, command in commands:
            log = (state / f"{name}.log").open("a", encoding="utf-8")
            logs.append(log)
            children.append(subprocess.Popen(command, cwd=ROOT, env=env,
                                             stdout=log, stderr=subprocess.STDOUT))
        print(f"服务启动中：在 Cloud Studio 预览端口 {args.port}。", flush=True)
        print(f"ML backend 内部地址 http://127.0.0.1:{args.backend_port}；日志目录 {state}", flush=True)
        if not env.get("QWEN_VL_API_URL"):
            print("视觉模型接口待配置；可先导入数据和人工标注，模型推理尚未启用。", flush=True)
        while True:
            for (name, _), child in zip(commands, children):
                code = child.poll()
                if code is not None:
                    print(f"{name} exited ({code}); see {state / (name + '.log')}", file=sys.stderr)
                    return code or 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    sys.exit(main())
