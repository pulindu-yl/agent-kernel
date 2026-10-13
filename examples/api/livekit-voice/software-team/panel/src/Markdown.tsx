// Renders the project's documents (team/documents.py writes headings, lists, tables, **bold**, _italic_ and
// `code`). It builds React elements and never injects HTML, so text the client typed cannot run in the page.

import type { ReactNode } from "react";

// **bold**, `code`, and _italic_ only at word edges (so test_r1_1 stays as it is).
const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|(?<!\w)_[^_]+_(?!\w))/g;

function inline(text: string): ReactNode[] {
  return text
    .split(INLINE)
    .filter(Boolean)
    .map((part, i) => {
      if (part.length > 4 && part.startsWith("**") && part.endsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>;
      if (part.length > 2 && part.startsWith("`") && part.endsWith("`")) return <code key={i}>{part.slice(1, -1)}</code>;
      if (part.length > 2 && part.startsWith("_") && part.endsWith("_")) return <em key={i}>{part.slice(1, -1)}</em>;
      return part;
    });
}

function cells(row: string): string[] {
  return row
    .trim()
    .replace(/^\||\|$/g, "")
    .split("|")
    .map((cell) => cell.trim());
}

function Table({ rows }: { rows: string[] }) {
  const [head, ...rest] = rows;
  const body = rest.length && /^\|?\s*:?-/.test(rest[0]) ? rest.slice(1) : rest;
  return (
    <table>
      <thead>
        <tr>
          {cells(head).map((cell, i) => (
            <th key={i}>{inline(cell)}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {body.map((row, r) => (
          <tr key={r}>
            {cells(row).map((cell, i) => (
              <td key={i}>{inline(cell)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function Markdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }
    const heading = /^(#{1,3})\s+(.*)$/.exec(line);
    if (heading) {
      const level = heading[1].length;
      const content = inline(heading[2]);
      blocks.push(level === 1 ? <h1 key={i}>{content}</h1> : level === 2 ? <h2 key={i}>{content}</h2> : <h3 key={i}>{content}</h3>);
      i++;
    } else if (line.startsWith("|")) {
      const rows: string[] = [];
      while (i < lines.length && lines[i].startsWith("|")) rows.push(lines[i++]);
      blocks.push(<Table key={i} rows={rows} />);
    } else if (line.startsWith("- ")) {
      const items: string[] = [];
      while (i < lines.length && lines[i].startsWith("- ")) items.push(lines[i++].slice(2));
      blocks.push(
        <ul key={i}>
          {items.map((item, k) => (
            <li key={k}>{inline(item)}</li>
          ))}
        </ul>,
      );
    } else {
      const paragraph: string[] = [];
      while (i < lines.length && lines[i].trim() && !/^(#{1,3}\s|- |\|)/.test(lines[i])) paragraph.push(lines[i++]);
      blocks.push(<p key={i}>{inline(paragraph.join(" "))}</p>);
    }
  }
  return <div className="markdown">{blocks}</div>;
}
