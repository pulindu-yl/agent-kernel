// The client's screen: the call on the left, the project board on the right, the controls below.

import { useState } from "react";

import Avatar from "./Avatar.tsx";
import Board from "./Board.tsx";
import Controls from "./Controls.tsx";
import Stage from "./Stage.tsx";
import { TEAM } from "./team.ts";
import { useAudioLevels, useCall } from "./useCall.ts";

// The name is remembered in this browser only, as a convenience; storage may be unavailable.
const NAME_KEY = "software-team.name";

function rememberedName(): string {
  try {
    return window.localStorage.getItem(NAME_KEY) ?? "";
  } catch {
    return "";
  }
}

function remember(name: string): void {
  try {
    window.localStorage.setItem(NAME_KEY, name);
  } catch {
    // Not remembered; the call works the same.
  }
}

type WelcomeProps = { joining: boolean; error: string | null; onJoin: (name: string) => void };

function Welcome({ joining, error, onJoin }: WelcomeProps) {
  const [name, setName] = useState(rememberedName);
  const join = () => {
    remember(name.trim());
    onJoin(name.trim());
  };
  return (
    <section className="welcome">
      <div className="welcome-card">
        <div className="welcome-team">
          {TEAM.map((m) => (
            <div key={m.key} className="welcome-member">
              <Avatar member={m} size={84} />
              <div className="tile-name">{m.name}</div>
              <div className="tile-role">{m.role}</div>
            </div>
          ))}
        </div>
        <h2>Meet your software team</h2>
        <p className="dim">
          Tell the PM what you need. The team agrees it with you, designs it, builds it and tests it, and you watch every
          step on the project board.
        </p>
        <form
          className="join-form"
          onSubmit={(event) => {
            event.preventDefault();
            join();
          }}
        >
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Your name"
            aria-label="Your name"
            maxLength={40}
            autoFocus
          />
          <button className="join" type="submit" disabled={joining || !name.trim()}>
            {joining ? "Joining…" : "Join the call"}
          </button>
        </form>
        {error && <p className="error">{error}</p>}
      </div>
    </section>
  );
}

export default function App() {
  const call = useCall();
  const levels = useAudioLevels(call.room);
  const connected = call.status === "connected";

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="logo">◆</span> Software team <span className="dim">· Agent Kernel</span>
        </div>
        {call.board && <div className="topbar-project">{call.board.project}</div>}
      </header>

      {connected ? (
        <Stage
          name={call.name}
          onSend={call.sendChat}
          onShare={call.shareFile}
          levels={levels}
          present={call.present}
          onCall={call.onCall}
          micOn={call.micOn}
          lines={call.lines}
        />
      ) : (
        <Welcome joining={call.status === "connecting"} error={call.error} onJoin={(name) => void call.join(name)} />
      )}

      {connected && (
        <Controls
          micOn={call.micOn}
          micError={call.micError}
          microphones={call.microphones}
          microphone={call.microphone}
          connectedAt={call.connectedAt}
          onToggleMic={() => void call.toggleMicrophone()}
          onChooseMic={(deviceId) => void call.chooseMicrophone(deviceId)}
          onLeave={() => void call.leave()}
        />
      )}

      <Board board={call.board} onApproval={call.answerApproval} />
    </>
  );
}
