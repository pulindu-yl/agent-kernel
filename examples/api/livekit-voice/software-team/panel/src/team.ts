// Who is on the call, as the page shows them. The gateway joins each voice agent as `agent-<key>`.

export const AGENT_PREFIX = "agent-";

export type Member = { key: string; name: string; role: string; color: string; glow: string };

export const YOU: Member = { key: "you", name: "You", role: "Client", color: "#3b82f6", glow: "#93c5fd" };

export const TEAM: Member[] = [
  { key: "pm", name: "PM", role: "Project manager", color: "#8b5cf6", glow: "#c4b5fd" },
  { key: "architect", name: "Architect", role: "Technical design", color: "#06b6d4", glow: "#67e8f9" },
  { key: "worker", name: "Worker", role: "Software engineer", color: "#f59e0b", glow: "#fcd34d" },
];

// Colors for the activity log, which also names the background agents and the team's systems.
export const WHO_COLORS: Record<string, string> = {
  pm: "#8b5cf6",
  architect: "#06b6d4",
  worker: "#f59e0b",
  coder: "#22c55e",
  reviewer: "#ec4899",
};

export function memberOf(key: string): Member | undefined {
  return TEAM.find((m) => m.key === key);
}
