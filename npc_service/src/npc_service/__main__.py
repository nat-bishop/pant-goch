import uvicorn

from .config import Settings


def main() -> None:
    cfg = Settings()
    uvicorn.run(
        "npc_service.main:app",
        host="127.0.0.1",
        port=cfg.npc_service_port,
        log_level=cfg.npc_log_level,
    )


if __name__ == "__main__":
    main()
