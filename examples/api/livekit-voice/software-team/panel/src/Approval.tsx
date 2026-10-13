// The approval stage: once the Architect has written the design, the client looks over what will be
// built and how, and answers with a click (team/approval.py records it and tells the team).

import { useState } from "react";

import Markdown from "./Markdown.tsx";
import { useFile } from "./ProjectViews.tsx";

export type ApprovalDecision = "approve" | "change";

const VIEWS = [
  { file: "requirements.md", name: "What we'll build" },
  { file: "design.md", name: "How we'll build it" },
];

type Props = { version: number; onAnswer: (decision: ApprovalDecision) => Promise<void> };

export default function ApprovalCard({ version, onAnswer }: Props) {
  const [view, setView] = useState(VIEWS[0].file);
  const [sending, setSending] = useState<ApprovalDecision | null>(null);
  const [askedForChange, setAskedForChange] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const text = useFile(view, version);

  async function answer(decision: ApprovalDecision) {
    setSending(decision);
    setError(null);
    try {
      await onAnswer(decision);
      if (decision === "change") setAskedForChange(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(null);
    }
  }

  return (
    <section className="card approval">
      <h3>Ready for your approval</h3>
      <p className="dim">This is what the team will build, and how. Look it over, then approve it or ask for a change.</p>
      <div className="doc-bar">
        {VIEWS.map((v) => (
          <button key={v.file} className={v.file === view ? "chip active" : "chip"} onClick={() => setView(v.file)}>
            {v.name}
          </button>
        ))}
      </div>
      <div className="doc-scroll">
        {text === undefined && <p className="dim">Loading…</p>}
        {text === null && <p className="dim">Not written yet.</p>}
        {text && <Markdown text={text} />}
      </div>
      {askedForChange && <p className="change-note">Tell the team what you would like to change: say it, or type it in the chat.</p>}
      <div className="actions">
        <button className="approve" disabled={sending !== null} onClick={() => void answer("approve")}>
          {sending === "approve" ? "Sending…" : "Looks good, build it"}
        </button>
        <button className="chip" disabled={sending !== null || askedForChange} onClick={() => void answer("change")}>
          I'd like a change
        </button>
      </div>
      {error && <p className="approval-error">{error}</p>}
    </section>
  );
}
