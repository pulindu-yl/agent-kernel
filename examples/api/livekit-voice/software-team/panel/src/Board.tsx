// The project board: what the team recorded, as the board participant last sent it.

import { useEffect, useRef, useState } from "react";

import ApprovalCard, { type ApprovalDecision } from "./Approval.tsx";
import { DocumentsTab, FilesTab, FileViewer } from "./ProjectViews.tsx";
import { WHO_COLORS } from "./team.ts";
import type { BoardSnapshot, Criterion, StepCard } from "./types.ts";

const RESULT_MARKS: Record<string, string> = { pass: "✓", fail: "✗", "not covered": "–" };

function Steps({ steps }: { steps: StepCard[] }) {
  return (
    <ol className="steps" style={{ gridTemplateColumns: `repeat(${steps.length}, 1fr)` }}>
      {steps.map((step) => (
        <li key={step.name} className={step.state}>
          <span className="step-dot">{step.state === "done" ? "✓" : ""}</span>
          <span className="step-name">{step.name}</span>
        </li>
      ))}
    </ol>
  );
}

function CriterionLine({ criterion }: { criterion: Criterion }) {
  return (
    <li className={`criterion ${criterion.result.replace(" ", "-")}`}>
      <span className="mark">{RESULT_MARKS[criterion.result] ?? "○"}</span>
      <span className="id">{criterion.id}</span>
      <span>{criterion.text}</span>
    </li>
  );
}

function duration(seconds: number): string {
  return seconds < 60 ? `${seconds} s` : `${Math.floor(seconds / 60)} min ${seconds % 60} s`;
}

function clock(at: number): string {
  return new Date(at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

type ProjectTabProps = { board: BoardSnapshot; onBrowse: () => void; onApproval: (decision: ApprovalDecision) => Promise<void> };

function ProjectTab({ board, onBrowse, onApproval }: ProjectTabProps) {
  const criteria = board.requirements.flatMap((r) => r.criteria);
  const passing = criteria.filter((c) => c.result === "pass").length;

  return (
    <>
      {board.approval_pending && <ApprovalCard version={board.updated_at} onAnswer={onApproval} />}

      {board.waiting_for_you.length > 0 && (
        <section className="card waiting">
          <h3>Waiting for you</h3>
          <ul>
            {board.waiting_for_you.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>
      )}

      {board.job && (
        <section className="card building">
          <h3>
            <span className="spinner" /> {board.job.id} · {board.job.kind}
            <span className="dim right">{duration(board.job.seconds)}</span>
          </h3>
          <ul className="timeline">
            {board.job.activity.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        </section>
      )}

      {board.delivered && (
        <section className="card delivered">
          <h3>Delivered</h3>
          <p className="dim">The code, its tests and the documents are ready.</p>
          <div className="actions">
            <a className="download" href="/project/download">
              Download the project (.zip)
            </a>
            <button className="chip" onClick={onBrowse}>
              Browse the files
            </button>
          </div>
        </section>
      )}

      <section className="card">
        <h3>
          Requirements
          <span className={`pill ${board.requirements_confirmed ? "good" : ""}`}>
            {board.requirements_confirmed ? "Confirmed" : "Draft"}
          </span>
          {criteria.length > 0 && (
            <span className="dim right">
              {passing} of {criteria.length} criteria pass
            </span>
          )}
        </h3>
        {board.requirements.length === 0 && <p className="dim">None yet.</p>}
        {board.requirements.map((r) => (
          <div key={r.id} className="requirement">
            <div className="item-title">
              <span className="id">{r.id}</span> {r.title}
            </div>
            <ul className="criteria">
              {r.criteria.map((c) => (
                <CriterionLine key={c.id} criterion={c} />
              ))}
            </ul>
          </div>
        ))}
      </section>

      {board.decisions.length > 0 && (
        <section className="card">
          <h3>
            Design
            <span className={`pill ${board.design_written ? "good" : ""}`}>{board.design_written ? "Written" : "In discussion"}</span>
          </h3>
          {board.decisions.map((d) => (
            <div key={d.id} className="decision">
              <div className="item-title">
                <span className="id">{d.id}</span> {d.title}: {d.choice}
              </div>
              {d.challenge && (
                <div className="challenge">
                  <span>Challenged</span> {d.challenge}
                </div>
              )}
            </div>
          ))}
        </section>
      )}

      {board.tasks.length > 0 && (
        <section className="card">
          <h3>
            Tasks {board.tests && <span className="dim right">Tests: {board.tests}</span>}
          </h3>
          <ul className="tasks">
            {board.tasks.map((t) => (
              <li key={t.id}>
                <span className={`status ${t.status.replace(" ", "-")}`}>{t.status}</span>
                <span className="id">{t.id}</span> {t.title}
              </li>
            ))}
          </ul>
        </section>
      )}

      {(board.findings.length > 0 || board.code_approved) && (
        <section className="card">
          <h3>
            Code review
            <span className={`pill ${board.code_approved ? "good" : ""}`}>
              {board.code_approved ? "Approved by the Architect" : "In review"}
            </span>
          </h3>
          {board.findings.length === 0 && <p className="dim">No findings.</p>}
          <ul className="tasks">
            {board.findings.map((f) => (
              <li key={f.id}>
                <span className={`status ${f.status.replace(" ", "-")}`}>{f.status === "open" ? "to discuss" : f.status}</span>
                <span className="id">{f.id}</span>
                <span>
                  {f.text}
                  {f.reason && <span className="finding-reason">The Architect: {f.reason}</span>}
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
  );
}

function ActivityTab({ board }: { board: BoardSnapshot }) {
  return (
    <ul className="activity">
      {[...board.recent].reverse().map((entry, i) => (
        <li key={`${entry.at}-${i}`}>
          <span className="time">{clock(entry.at)}</span>
          <span className="who" style={{ color: WHO_COLORS[entry.who] ?? "var(--dim)" }}>
            {entry.who}
          </span>
          <span>{entry.what}</span>
        </li>
      ))}
    </ul>
  );
}

type Tab = "project" | "documents" | "files" | "activity";
const TABS: { key: Tab; name: string }[] = [
  { key: "project", name: "Project" },
  { key: "documents", name: "Documents" },
  { key: "files", name: "Files" },
  { key: "activity", name: "Activity" },
];

type BoardProps = { board: BoardSnapshot | null; onApproval: (decision: ApprovalDecision) => Promise<void> };

export default function Board({ board, onApproval }: BoardProps) {
  const [tab, setTab] = useState<Tab>("project");
  const [viewing, setViewing] = useState<string | null>(null);
  const body = useRef<HTMLDivElement>(null);

  // When the plan becomes ready for approval, bring it into view once; the client may then look elsewhere.
  const approvalPending = board?.approval_pending ?? false;
  const wasPending = useRef(false);
  useEffect(() => {
    if (approvalPending && !wasPending.current) {
      setTab("project");
      body.current?.scrollTo({ top: 0, behavior: "smooth" });
    }
    wasPending.current = approvalPending;
  }, [approvalPending]);

  if (!board) {
    return (
      <aside className="board">
        <div className="board-empty">
          <div className="section-title">Project board</div>
          <p className="dim">The project fills in here as the team records it: requirements, design, build progress and results.</p>
        </div>
      </aside>
    );
  }

  return (
    <aside className="board">
      <header className="board-head">
        <span className="phase">{board.phase}</span>
        <h2>{board.project}</h2>
        {board.goal && <p className="goal">{board.goal}</p>}
        <Steps steps={board.steps} />
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.key} className={tab === t.key ? "active" : ""} onClick={() => setTab(t.key)}>
              {t.name}
              {t.key === "files" && board.files.length > 0 && <span className="count">{board.files.length}</span>}
            </button>
          ))}
        </nav>
      </header>
      <div className="board-body" ref={body}>
        {tab === "project" && <ProjectTab board={board} onBrowse={() => setTab("files")} onApproval={onApproval} />}
        {tab === "documents" && <DocumentsTab version={board.updated_at} />}
        {tab === "files" && <FilesTab files={board.files} onOpen={setViewing} />}
        {tab === "activity" && <ActivityTab board={board} />}
      </div>
      {viewing && <FileViewer path={viewing} onClose={() => setViewing(null)} />}
    </aside>
  );
}
