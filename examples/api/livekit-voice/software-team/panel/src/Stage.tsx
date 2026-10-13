// The call: a tile for you and for each voice agent, and what the team said.

import { useEffect, useRef, useState, type CSSProperties } from "react";

import Avatar from "./Avatar.tsx";
import { TEAM, YOU, type Member } from "./team.ts";
import type { Line } from "./useCall.ts";

const BAR_SHAPE = [0.55, 0.85, 1, 0.75, 0.5];

function VoiceBars({ level }: { level: number }) {
  return (
    <div className="bars" aria-hidden="true">
      {BAR_SHAPE.map((shape, i) => (
        <span key={i} style={{ height: `${4 + Math.min(1, level * 2.2) * 18 * shape}px` }} />
      ))}
    </div>
  );
}

type TileProps = { member: Member; level: number; here: boolean; hasCall: boolean; muted?: boolean };

function Tile({ member, level, here, hasCall, muted }: TileProps) {
  const speaking = level > 0.01;
  const classes = ["tile", hasCall && "has-call", speaking && "speaking", !here && "away"].filter(Boolean).join(" ");
  return (
    <div className={classes} style={{ "--color": member.color } as CSSProperties}>
      {hasCall && <div className="badge">On the call</div>}
      <Avatar member={member} level={level} size={104} />
      <div className="tile-footer">
        <div>
          <div className="tile-name">{member.name}</div>
          <div className="tile-role">{here ? member.role : "Joining…"}</div>
        </div>
        {muted ? <span className="muted">Muted</span> : <VoiceBars level={level} />}
      </div>
    </div>
  );
}

function Composer({ onSend, onShare }: { onSend: (text: string) => Promise<void>; onShare: (file: File) => Promise<void> }) {
  const [text, setText] = useState("");
  const [problem, setProblem] = useState<string | null>(null);
  const picker = useRef<HTMLInputElement>(null);
  const report = (e: unknown) => setProblem(e instanceof Error ? e.message : String(e));

  return (
    <form
      className="composer"
      onSubmit={(event) => {
        event.preventDefault();
        setProblem(null);
        onSend(text)
          .then(() => setText(""))
          .catch(report);
      }}
    >
      <button type="button" className="attach" title="Share a file with the team" onClick={() => picker.current?.click()}>
        📎
      </button>
      <input
        ref={picker}
        type="file"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (file) {
            setProblem(null);
            onShare(file).catch(report);
          }
        }}
      />
      <input
        className="message"
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="Type a link, a name, or anything easier written than said"
        aria-label="Message to the team"
      />
      <button type="submit" className="send" disabled={!text.trim()}>
        Send
      </button>
      {problem && <span className="error composer-error">{problem}</span>}
    </form>
  );
}

type StageProps = {
  name: string;
  onSend: (text: string) => Promise<void>;
  onShare: (file: File) => Promise<void>;
  levels: Record<string, number>;
  present: string[];
  onCall: string | null;
  micOn: boolean;
  lines: Line[];
};

export default function Stage({ name, onSend, onShare, levels, present, onCall, micOn, lines }: StageProps) {
  const you = { ...YOU, name: name || YOU.name };
  const end = useRef<HTMLDivElement>(null);
  // A block body, not an expression: an effect's return value is its cleanup, and Chrome's smooth
  // scrollIntoView returns a promise, which React then tried to call and crashed the page.
  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [lines.length]);

  return (
    <section className="stage">
      <div className="tiles">
        <Tile member={you} level={levels.you ?? 0} here hasCall={false} muted={!micOn} />
        {TEAM.map((m) => (
          <Tile key={m.key} member={m} level={levels[m.key] ?? 0} here={present.includes(m.key)} hasCall={onCall === m.key} />
        ))}
      </div>

      <div className="transcript">
        <div className="section-title">Conversation</div>
        {lines.length === 0 && <p className="dim">What you and the team say appears here.</p>}
        {lines.map((line) =>
          line.notice ? (
            <div key={line.id} className="notice">
              {line.text}
            </div>
          ) : (
            <div key={line.id} className={`line${line.you ? " you" : ""}`}>
              <span className="who" style={{ color: line.you ? YOU.glow : TEAM.find((m) => m.name === line.who)?.glow }}>
                {line.you ? you.name : line.who}
              </span>
              <span>{line.text}</span>
            </div>
          ),
        )}
        <div ref={end} />
      </div>
      <Composer onSend={onSend} onShare={onShare} />
    </section>
  );
}
