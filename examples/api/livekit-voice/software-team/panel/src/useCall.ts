// Joins the call's LiveKit room as the client: publishes the microphone, plays the agents, and
// collects what the page shows (who is here, who has the call, what they said, the project board).

import { useCallback, useEffect, useRef, useState } from "react";
import { Room, RoomEvent, Track, type RemoteParticipant, type RemoteTrack } from "livekit-client";

import { AGENT_PREFIX, memberOf } from "./team.ts";
import type { BoardSnapshot } from "./types.ts";

// The project board's topic (team/board.py), and the gateway's chat topic, on which each agent
// publishes its words. `ak.active` marks the agent participant that has the call.
const BOARD_TOPIC = "ak.project";
const CHAT_TOPIC = "lk-chat";
// The client's own words, as the model heard them, which the project board passes on (team/board.py).
const CLIENT_WORDS_TOPIC = "ak.client";
const ACTIVE_ATTRIBUTE = "ak.active";
const MAX_LINES = 80;
// The gateway also reports a failed model connection on the chat topic, in the agent's name.
const NOTICE = /socket error|exceeded your current quota|^error\b/i;
// Gemini sometimes transcribes a tool call it is making, alone or mid-sentence ("call:transfer_to_agent{agent_name:worker}"):
// not words to show. One cut short ends where the next sentence runs on ("...like cut-Understood, ...").
const TOOL_CALL_TEXT = /call:\w+\{[^}]*?(?:\}|[a-z,-](?=[A-Z])|$)/g;
// Or says it in words, with no call made ('transfer_to_agent agent_name: "architect"').
const SPOKEN_HANDOVER = /\btransfer_to_agent\b(?:\s*agent_name:?\s*"?\w+"?)?/g;
// Words an agent said after a tool call are joined to those before it with no space ("written?Pulindu").
const RUN_ON_SENTENCE = /([.!?])(?=[A-Z])/g;

export type Line = { id: string; who: string; text: string; notice: boolean; you: boolean };
export type CallStatus = "idle" | "connecting" | "connected";

type TokenReply = { url: string; token: string; name: string };

function agentKey(identity: string): string {
  return identity.slice(AGENT_PREFIX.length);
}

function lineFrom(payload: Uint8Array, participant: RemoteParticipant): Line | null {
  try {
    const { id, message } = JSON.parse(new TextDecoder().decode(payload)) as { id?: string; message?: string };
    const text = (message ?? "")
      .replace(TOOL_CALL_TEXT, " ")
      .replace(SPOKEN_HANDOVER, " ")
      .replace(RUN_ON_SENTENCE, "$1 ")
      .replace(/\s+/g, " ")
      .trim();
    if (!text) return null;
    const who = agentKey(participant.identity);
    const name = memberOf(who)?.name ?? who;
    return { id: id || crypto.randomUUID(), who: name, text, notice: NOTICE.test(text), you: false };
  } catch {
    return null;
  }
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export function useCall() {
  const roomRef = useRef<Room | null>(null);
  const [room, setRoom] = useState<Room | null>(null);
  const [status, setStatus] = useState<CallStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [present, setPresent] = useState<string[]>([]);
  const [onCall, setOnCall] = useState<string | null>(null);
  const [lines, setLines] = useState<Line[]>([]);
  const [board, setBoard] = useState<BoardSnapshot | null>(null);
  const [connectedAt, setConnectedAt] = useState<number | null>(null);
  const [micOn, setMicOn] = useState(false);
  const [micError, setMicError] = useState<string | null>(null);
  const [microphones, setMicrophones] = useState<MediaDeviceInfo[]>([]);
  const [microphone, setMicrophone] = useState("");
  const [name, setName] = useState("");

  // The microphone can fail on its own: another tab or app holds it, permission was refused, or the
  // default device cannot record (a Bluetooth headset in music mode has no microphone). The call and
  // the board work without it, so a failure never ends the call; each microphone is tried in turn.
  const enableMicrophone = useCallback(async (deviceId?: string) => {
    const current = roomRef.current;
    if (!current) return;
    const devices = await Room.getLocalDevices("audioinput", false).catch(() => [] as MediaDeviceInfo[]);
    setMicrophones(devices.filter((d) => d.deviceId && d.deviceId !== "communications"));
    const candidates = deviceId
      ? [deviceId]
      : [undefined, ...devices.map((d) => d.deviceId).filter((id) => id && id !== "default" && id !== "communications")];
    let reason = "No microphone found";
    for (const candidate of candidates) {
      try {
        await current.localParticipant.setMicrophoneEnabled(true, candidate ? { deviceId: { exact: candidate } } : undefined);
        setMicrophone(current.getActiveDevice("audioinput") ?? candidate ?? "default");
        setMicOn(true);
        setMicError(null);
        return;
      } catch (e) {
        reason = errorText(e);
        await current.localParticipant.setMicrophoneEnabled(false).catch(() => undefined);
      }
    }
    setMicOn(false);
    setMicError(`${reason}. Close other tabs or apps using the microphone, or choose another one.`);
  }, []);

  const toggleMicrophone = useCallback(async () => {
    const current = roomRef.current;
    if (!current) return;
    if (micOn) {
      await current.localParticipant.setMicrophoneEnabled(false);
      setMicOn(false);
    } else {
      await enableMicrophone();
    }
  }, [micOn, enableMicrophone]);

  const chooseMicrophone = useCallback(
    async (deviceId: string) => {
      const current = roomRef.current;
      if (!current) return;
      if (micOn && (await current.switchActiveDevice("audioinput", deviceId).catch(() => false))) {
        setMicrophone(deviceId);
        return;
      }
      await enableMicrophone(deviceId);
    },
    [micOn, enableMicrophone],
  );

  const join = useCallback(async (yourName: string) => {
    setError(null);
    setMicError(null);
    setStatus("connecting");
    const next = new Room();
    roomRef.current = next;
    const refreshAgents = () => {
      const agents = Array.from(next.remoteParticipants.values()).filter((p) => p.identity.startsWith(AGENT_PREFIX));
      setPresent(agents.map((p) => agentKey(p.identity)));
      const holder = agents.find((p) => p.attributes[ACTIVE_ATTRIBUTE] === "true");
      setOnCall(holder ? agentKey(holder.identity) : null);
    };
    next
      .on(RoomEvent.ParticipantConnected, refreshAgents)
      .on(RoomEvent.ParticipantDisconnected, refreshAgents)
      .on(RoomEvent.ParticipantAttributesChanged, refreshAgents)
      .on(RoomEvent.TrackSubscribed, (track: RemoteTrack) => {
        if (track.kind === Track.Kind.Audio) document.body.appendChild(track.attach());
      })
      .on(RoomEvent.TrackUnsubscribed, (track: RemoteTrack) => {
        track.detach().forEach((element) => element.remove());
      })
      .on(RoomEvent.DataReceived, (payload, participant, _kind, topic) => {
        if (topic !== CHAT_TOPIC || !participant?.identity.startsWith(AGENT_PREFIX)) return;
        const line = lineFrom(payload, participant);
        if (line) setLines((all) => [...all, line].slice(-MAX_LINES));
      })
      .on(RoomEvent.Disconnected, () => {
        setStatus("idle");
        setRoom(null);
        setPresent([]);
        setOnCall(null);
        setMicOn(false);
        setConnectedAt(null);
      });
    // Registered before connecting, so the snapshot the board sends a new participant is not missed.
    next.registerTextStreamHandler(CLIENT_WORDS_TOPIC, async (reader) => {
      try {
        const { text } = JSON.parse(await reader.readAll()) as { text?: string };
        if (text?.trim()) {
          const line = { id: reader.info.id, who: "", text: text.trim(), notice: false, you: true };
          setLines((all) => [...all, line].slice(-MAX_LINES));
        }
      } catch (e) {
        console.error("Could not read the client's words", e);
      }
    });
    next.registerTextStreamHandler(BOARD_TOPIC, async (reader) => {
      try {
        setBoard(JSON.parse(await reader.readAll()) as BoardSnapshot);
      } catch (e) {
        console.error("Could not read the project board", e);
      }
    });

    try {
      const response = await fetch(`/token?name=${encodeURIComponent(yourName)}`);
      if (!response.ok) throw new Error(`No token (${response.status}): ${await response.text()}`);
      const { url, token, name: joinedAs } = (await response.json()) as TokenReply;
      setName(joinedAs);
      await next.connect(url, token);
      await next.startAudio();
    } catch (e) {
      setError(errorText(e));
      setStatus("idle");
      await next.disconnect();
      roomRef.current = null;
      return;
    }
    refreshAgents();
    setRoom(next);
    setConnectedAt(Date.now());
    setStatus("connected");
    await enableMicrophone();
  }, [enableMicrophone]);

  const addMine = useCallback((text: string) => {
    setLines((all) => [...all, { id: crypto.randomUUID(), who: "", text, notice: false, you: true }].slice(-MAX_LINES));
  }, []);

  // Typed messages go to the room chat, which the gateway passes to the agent with the call as the client's words.
  const sendChat = useCallback(
    async (text: string) => {
      const current = roomRef.current;
      const message = text.trim();
      if (!current || !message) return;
      const payload = JSON.stringify({ id: crypto.randomUUID(), message, timestamp: Date.now() });
      await current.localParticipant.publishData(new TextEncoder().encode(payload), { reliable: true, topic: CHAT_TOPIC });
      addMine(message);
    },
    [addMine],
  );

  // A shared file is saved in the project, announced to the team, and given to the Coder (team/project_files.py).
  const shareFile = useCallback(
    async (file: File) => {
      const by = encodeURIComponent(name || "The client");
      const response = await fetch(`/project/share?name=${encodeURIComponent(file.name)}&by=${by}`, { method: "POST", body: file });
      if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail ?? `Could not share ${file.name}`);
      const { saved } = (await response.json()) as { saved: string };
      addMine(`📎 Shared ${saved}`);
    },
    [name, addMine],
  );

  // The client's answer to the requirements and the design, from the approval card (team/approval.py).
  const answerApproval = useCallback(
    async (decision: "approve" | "change") => {
      const by = encodeURIComponent(name || "The client");
      const response = await fetch(`/project/approval?decision=${decision}&by=${by}`, { method: "POST" });
      if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail ?? "Could not send your answer");
      addMine(decision === "approve" ? "✓ Approved the requirements and the design" : "✎ Asked for a change");
    },
    [name, addMine],
  );

  const leave = useCallback(async () => {
    await roomRef.current?.disconnect();
    roomRef.current = null;
  }, []);

  useEffect(() => () => void roomRef.current?.disconnect(), []);

  return {
    room,
    name,
    status,
    error,
    present,
    onCall,
    lines,
    board,
    connectedAt,
    micOn,
    micError,
    microphones,
    microphone,
    join,
    leave,
    toggleMicrophone,
    chooseMicrophone,
    sendChat,
    shareFile,
    answerApproval,
  };
}

// How loud each speaker is right now (0 to 1), by member key ("you", "pm", ...), ten times a second.
export function useAudioLevels(room: Room | null): Record<string, number> {
  const [levels, setLevels] = useState<Record<string, number>>({});
  useEffect(() => {
    if (!room) {
      setLevels({});
      return;
    }
    const level = (speaking: boolean, audioLevel: number) => (speaking ? Math.max(audioLevel, 0.25) : 0);
    const timer = window.setInterval(() => {
      const next: Record<string, number> = {
        you: level(room.localParticipant.isSpeaking, room.localParticipant.audioLevel),
      };
      room.remoteParticipants.forEach((p) => {
        if (p.identity.startsWith(AGENT_PREFIX)) next[agentKey(p.identity)] = level(p.isSpeaking, p.audioLevel);
      });
      setLevels(next);
    }, 100);
    return () => window.clearInterval(timer);
  }, [room]);
  return levels;
}
