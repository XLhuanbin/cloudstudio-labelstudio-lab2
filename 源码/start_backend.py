"""Start the ML SDK HTTP server directly, on Windows or Cloud Studio."""
from pathlib import Path
import argparse
import os

from label_studio_ml.api import init_app
from ml_backend import NewModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--model-dir", type=Path, default=Path("_工作区/backend_state"))
    args = parser.parse_args()
    os.environ.setdefault("USE_REDIS", "false")
    app = init_app(NewModel, model_dir=str(args.model_dir.resolve()))
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
