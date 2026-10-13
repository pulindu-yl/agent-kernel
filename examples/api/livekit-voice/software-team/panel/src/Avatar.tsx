// Each member's avatar: a gradient orb with the member's symbol, and an aura that grows with their voice.

import type { CSSProperties, ReactElement } from "react";

import type { Member } from "./team.ts";

// Line drawings on a 96 x 96 canvas, centered on the orb.
const SYMBOLS: Record<string, ReactElement> = {
  // A clipboard with a check: the requirements the PM keeps.
  pm: (
    <>
      <rect x="31" y="27" width="34" height="44" rx="6" />
      <rect x="40" y="22" width="16" height="10" rx="3" />
      <path d="M39 51 l7 7 l13 -14" />
    </>
  ),
  // A drafting compass: the Architect's design.
  architect: (
    <>
      <circle cx="48" cy="31" r="5" />
      <path d="M48 20 v6 M45 35 L34 70 M51 35 L62 70 M37 58 Q48 64 59 58" />
    </>
  ),
  // Code brackets: the Worker builds.
  worker: <path d="M38 34 L25 48 L38 62 M58 34 L71 48 L58 62 M53 30 L43 66" />,
  // A person: the client.
  you: (
    <>
      <circle cx="48" cy="38" r="11" />
      <path d="M27 72 Q48 48 69 72" />
    </>
  ),
};

type Props = { member: Member; level?: number; size?: number };

export default function Avatar({ member, level = 0, size = 96 }: Props) {
  const style = {
    "--color": member.color,
    "--glow": member.glow,
    "--level": level.toFixed(2),
    width: size,
    height: size,
  } as CSSProperties;
  return (
    <div className="avatar" style={style}>
      <div className="aura" />
      <svg viewBox="0 0 96 96" aria-hidden="true">
        <defs>
          <radialGradient id={`orb-${member.key}`} cx="35%" cy="30%" r="75%">
            <stop offset="0%" stopColor={member.glow} />
            <stop offset="100%" stopColor={member.color} />
          </radialGradient>
        </defs>
        <circle cx="48" cy="48" r="46" fill={`url(#orb-${member.key})`} />
        <g className="symbol">{SYMBOLS[member.key]}</g>
      </svg>
    </div>
  );
}
