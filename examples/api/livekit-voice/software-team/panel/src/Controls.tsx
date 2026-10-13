// The control bar: the microphone (with its device menu), the call timer, and leaving.

import { useEffect, useState } from "react";

type Props = {
  micOn: boolean;
  micError: string | null;
  microphones: MediaDeviceInfo[];
  microphone: string;
  connectedAt: number | null;
  onToggleMic: () => void;
  onChooseMic: (deviceId: string) => void;
  onLeave: () => void;
};

function MicIcon({ off }: { off: boolean }) {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" className="icon">
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
      {off && <path d="M4 4l16 16" />}
    </svg>
  );
}

function Timer({ since }: { since: number | null }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  if (since === null) return null;
  const seconds = Math.max(0, Math.floor((now - since) / 1000));
  return (
    <span className="timer">
      <span className="live-dot" />
      {String(Math.floor(seconds / 60)).padStart(2, "0")}:{String(seconds % 60).padStart(2, "0")}
    </span>
  );
}

export default function Controls(props: Props) {
  const { micOn, micError, microphones, microphone, connectedAt, onToggleMic, onChooseMic, onLeave } = props;
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <footer className="controls">
      <Timer since={connectedAt} />
      <div className="control-group">
        <div className="mic-control">
          <button className={`round ${micOn ? "" : "off"}`} onClick={onToggleMic} title={micOn ? "Mute" : "Unmute"}>
            <MicIcon off={!micOn} />
          </button>
          <button className="chevron" onClick={() => setMenuOpen((open) => !open)} title="Choose a microphone">
            ▾
          </button>
          {menuOpen && (
            <div className="menu" onMouseLeave={() => setMenuOpen(false)}>
              <div className="menu-title">Microphone</div>
              {microphones.length === 0 && <div className="menu-item dim">No microphones found</div>}
              {microphones.map((m) => (
                <button
                  key={m.deviceId}
                  className={`menu-item ${m.deviceId === microphone ? "chosen" : ""}`}
                  onClick={() => {
                    onChooseMic(m.deviceId);
                    setMenuOpen(false);
                  }}
                >
                  {m.label || "Microphone"}
                </button>
              ))}
            </div>
          )}
        </div>
        <button className="leave" onClick={onLeave}>
          Leave
        </button>
      </div>
      <span className="mic-error">{micError}</span>
    </footer>
  );
}
