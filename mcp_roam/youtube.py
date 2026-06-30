"""MCP tools: transcribe YouTube videos into roam notes via yt-service.

Fire-and-forget pattern for long (hour+) videos: roam_youtube_note starts a
job and returns immediately; roam_youtube_note_status checks on it later.
Uses only the stdlib so no new dependencies.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request

from mcp.server.fastmcp import FastMCP

YT_BASE = os.environ.get('YT_SERVICE_URL', 'http://localhost:9000/yt')


def _post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def register_youtube(mcp: FastMCP) -> None:
    """Register YouTube transcription tools on the FastMCP instance."""

    @mcp.tool(name='roam_youtube_note')
    async def roam_youtube_note(
        url: str,
        language: str | None = None,
        topic: str | None = None,
    ) -> str:
        """Start transcribing a YouTube video into a roam note (non-blocking).

        Returns a job_id immediately; the job runs in the background on
        yt-service. Check the result later with roam_youtube_note_status.

        Args:
            url: YouTube video URL.
            language: Optional ISO language code (e.g. 'es', 'en'); auto-detected if omitted.
            topic: Optional roam node title to link the note under *Related*.

        Returns:
            A job_id plus a hint to poll with roam_youtube_note_status.
        """
        try:
            started = await asyncio.to_thread(
                _post_json,
                f'{YT_BASE}/transcribe',
                {'url': url, 'language': language, 'topic': topic},
            )
        except urllib.error.URLError as e:
            return (
                f'Could not reach yt-service at {YT_BASE} ({e}). '
                'Is the utils-server supervisor running?'
            )
        job_id = started['job_id']
        return (
            f'Started transcription job {job_id} for {url}.\n'
            f'Check status with roam_youtube_note_status(job_id="{job_id}").'
        )

    @mcp.tool(name='roam_youtube_note_status')
    async def roam_youtube_note_status(job_id: str) -> str:
        """Check the status of a YouTube transcription job.

        Args:
            job_id: The job_id returned by roam_youtube_note.

        Returns:
            Current status; when done, the created note path + transcript path.
        """
        try:
            data = await asyncio.to_thread(_get_json, f'{YT_BASE}/jobs/{job_id}')
        except urllib.error.URLError as e:
            return f'Could not reach yt-service at {YT_BASE} ({e}).'
        status = data.get('status')
        if status == 'done':
            result = data.get('result') or {}
            return (
                f"DONE - created roam note: {result.get('note_path')}\n"
                f"Title: {result.get('title')}\n"
                f"Transcript: {result.get('transcript_path')}"
            )
        if status == 'error':
            return f"ERROR - {data.get('error')}"
        if status == 'interrupted':
            return (
                'INTERRUPTED - service restarted while the job was running. '
                'Resubmit to retry.'
            )
        return f'STATUS: {status} (still working).'
