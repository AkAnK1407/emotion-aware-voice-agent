"""
clear_room.py — Deletes the demo LiveKit room server-side, if it exists.

Run before/after restarting the agent worker. Without this, a room that
still has a "ghost" participant from a process that was killed abruptly
(Stop-Process -Force, a crash, a network drop) won't get a fresh job
dispatch on the next join — LiveKit doesn't notice the old session is
dead until something forces it to, and until then new joins to that same
room name silently get no agent. Deleting the room clears that state.
"""
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from livekit import api

ROOM_NAME = sys.argv[1] if len(sys.argv) > 1 else "adaptivecx-live-room"


async def main():
    lkapi = api.LiveKitAPI(
        url=os.getenv("LIVEKIT_URL").replace("wss://", "https://"),
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
    )
    try:
        await lkapi.room.delete_room(api.DeleteRoomRequest(room=ROOM_NAME))
        print(f"cleared room: {ROOM_NAME}")
    except Exception as e:
        print(f"no stale room to clear ({ROOM_NAME}): {e}")
    await lkapi.aclose()


asyncio.run(main())
