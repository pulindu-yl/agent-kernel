// What the project board sends on the `ak.project` topic: `BoardSnapshot` in team/board.py.

export type Criterion = { id: string; text: string; result: string };
export type RequirementCard = { id: string; title: string; criteria: Criterion[] };
export type DecisionCard = { id: string; title: string; choice: string; challenge: string };
export type TaskCard = { id: string; title: string; status: string };
export type FindingCard = { id: string; text: string; status: string; reason: string };
export type JobCard = { id: string; kind: string; seconds: number; activity: string[] };
export type StepCard = { name: string; state: "done" | "current" | "todo" };
export type LogLine = { at: number; who: string; what: string };

export type BoardSnapshot = {
  project: string;
  goal: string;
  phase: string;
  requirements_confirmed: boolean;
  design_written: boolean;
  approval_pending: boolean;
  code_approved: boolean;
  delivered: boolean;
  folder: string;
  steps: StepCard[];
  waiting_for_you: string[];
  requirements: RequirementCard[];
  decisions: DecisionCard[];
  tasks: TaskCard[];
  job: JobCard | null;
  findings: FindingCard[];
  tests: string;
  files: string[];
  recent: LogLine[];
  updated_at: number;
};
