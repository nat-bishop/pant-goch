import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "npc_service.main:app",
        host=os.getenv("NPC_SERVICE_HOST", "127.0.0.1"),
        port=int(os.getenv("NPC_SERVICE_PORT", "8000")),
        log_level=os.getenv("NPC_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
