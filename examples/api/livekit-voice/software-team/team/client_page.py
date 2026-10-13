"""The client's page, served by the example's own server on port 8000.

The page (panel/, a small React app) joins the call and draws the project board beside it. This
serves its built files and gives it a LiveKit token, as the ``examples/api/agui`` example serves its
UI; and it lets the page view and download the project's files, and share files with the team.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from team.approval import ClientApproval
from team.project_files import ProjectFiles
from team.room_access import RoomAccess
from team.state import TeamCalls


class ClientPage:
    """Routes for the client's page: the page itself, its token, and the project's files.

    ``GET /`` and ``GET /assets/...`` serve the built page; ``GET /token`` gives it a room token;
    ``GET /project/files``, ``GET /project/files/{path}`` and ``GET /project/download`` list, read
    and zip the project folder; ``POST /project/share`` saves a file the client shares;
    ``POST /project/approval`` takes the client's approval of the requirements and the design, or
    their wish to change something first.

    None of them asks who is calling, which is fine on your own machine but not on a public server:
    put real authentication in front of them before deploying.
    """

    DIST = Path(__file__).parent.parent / "panel" / "dist"
    BUILD_HINT = (
        "<h1>The client page is not built</h1>"
        "<p>From <code>panel/</code> run <code>npm install &amp;&amp; npm run dev</code> and open "
        '<a href="http://localhost:5173">http://localhost:5173</a>, or run <code>npm run build</code> to serve it here.</p>'
    )

    def __init__(self, access: RoomAccess, calls: TeamCalls, dist: Path = DIST):
        self._access = access
        self._calls = calls
        self._dist = dist

    def router(self) -> APIRouter:
        router = APIRouter()
        router.add_api_route("/", self.index, methods=["GET"], response_class=HTMLResponse)
        router.add_api_route("/assets/{filename}", self.asset, methods=["GET"], include_in_schema=False)
        router.add_api_route("/token", self.token, methods=["GET"])
        router.add_api_route("/project/files", self.files, methods=["GET"])
        router.add_api_route("/project/files/{path:path}", self.file, methods=["GET"])
        router.add_api_route("/project/download", self.download, methods=["GET"])
        router.add_api_route("/project/share", self.share, methods=["POST"])
        router.add_api_route("/project/approval", self.approval, methods=["POST"])
        return router

    def index(self) -> HTMLResponse:
        """The built page, or how to build it."""
        entry = self._dist / "index.html"
        if not entry.is_file():
            return HTMLResponse(self.BUILD_HINT, status_code=503)
        return HTMLResponse(entry.read_text(encoding="utf-8"))

    def asset(self, filename: str) -> FileResponse:
        """A file of the built page, by exact name only."""
        root = self._dist / "assets"
        match = next((p for p in root.iterdir() if p.name == filename and p.is_file()), None) if root.is_dir() else None
        if match is None:
            raise HTTPException(status_code=404, detail="No such asset")
        return FileResponse(match)

    def token(self, name: str = "") -> dict:
        """Where the room is and a token to join it as the client, under their name (``?name=Pulindu``)."""
        if not self._access.configured:
            raise HTTPException(
                status_code=503, detail="Set AK_LIVEKIT__URL, AK_LIVEKIT__API_KEY and AK_LIVEKIT__API_SECRET"
            )
        display = RoomAccess.display_name(name)
        # Kept for a renewed model connection, which has lost the greeting that named the client.
        self._calls.get(self._access.room).client_name = display
        return {
            "url": self._access.url,
            "room": self._access.room,
            "name": display,
            "token": self._access.client_token(display),
        }

    def files(self) -> dict:
        """The files in the project folder."""
        return {"files": self._project().list()}

    def file(self, path: str) -> FileResponse:
        """One file of the project, to view or save."""
        try:
            return FileResponse(self._project().path(path))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="No such file in the project")

    def download(self) -> Response:
        """The whole project as a zip file."""
        project = self._project()
        return Response(
            project.zip(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{project.folder.name}.zip"'},
        )

    async def share(self, request: Request, name: str, by: str = "The client") -> dict:
        """Save a file the client shares (the request body is the file) and tell the team about it."""
        try:
            saved = self._project().share(name, await request.body(), RoomAccess.display_name(by))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"saved": saved}

    def approval(self, decision: str, by: str = "The client") -> dict:
        """The client's answer to the requirements and the design: ``?decision=approve`` or ``?decision=change``."""
        actions = {"approve": ClientApproval.approve, "change": ClientApproval.ask_for_change}
        if decision not in actions:
            raise HTTPException(status_code=400, detail="decision is approve or change")
        try:
            actions[decision](ClientApproval(self._calls.get(self._access.room)), RoomAccess.display_name(by))
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"decision": decision}

    def _project(self) -> ProjectFiles:
        return ProjectFiles(self._calls.get(self._access.room))
