#!/usr/bin/env python3
"""
Entry point for Doc Flash application
"""

import uvicorn
from docflash.app import app

if __name__ == "__main__":
    uvicorn.run(
        "docflash.app:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        access_log=True
    )