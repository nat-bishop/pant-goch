import uvicorn

from .config import Settings


def main() -> None:
    cfg = Settings()
    uvicorn.run(
        "handler.main:app",
        host="127.0.0.1",
        port=cfg.handler_port,
        log_level=cfg.handler_log_level,
    )


if __name__ == "__main__":
    main()
