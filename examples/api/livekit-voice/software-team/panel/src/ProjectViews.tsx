// The board's Documents and Files tabs, and the file viewer: the project as the team writes it, while
// the call goes on, and what was delivered.

import { useEffect, useState } from "react";

import Markdown from "./Markdown.tsx";

const DOCUMENTS = [
  { file: "requirements.md", name: "Requirements" },
  { file: "design.md", name: "Design" },
  { file: "status.md", name: "Status" },
];
const IMAGE = /\.(png|jpe?g|gif|svg|webp)$/i;

function fileUrl(path: string): string {
  return `/project/files/${path.split("/").map(encodeURIComponent).join("/")}`;
}

// A document, read again whenever the board says the project changed.
export function useFile(path: string | null, version: number): string | null | undefined {
  const [text, setText] = useState<string | null | undefined>(undefined);
  useEffect(() => {
    if (!path) return;
    let current = true;
    fetch(fileUrl(path))
      .then((response) => (response.ok ? response.text() : null))
      .then((body) => current && setText(body))
      .catch(() => current && setText(null));
    return () => {
      current = false;
    };
  }, [path, version]);
  return text;
}

export function DocumentsTab({ version }: { version: number }) {
  const [doc, setDoc] = useState(DOCUMENTS[0].file);
  const text = useFile(doc, version);

  return (
    <div className="documents">
      <div className="doc-bar">
        {DOCUMENTS.map((d) => (
          <button key={d.file} className={d.file === doc ? "chip active" : "chip"} onClick={() => setDoc(d.file)}>
            {d.name}
          </button>
        ))}
        <button className="chip right" onClick={() => window.print()} title="Save this document as a PDF">
          Save as PDF
        </button>
      </div>
      <div className="doc-print">
        {text === undefined && <p className="dim">Loading…</p>}
        {text === null && <p className="dim">Not written yet: it appears as soon as the team records something.</p>}
        {text && <Markdown text={text} />}
      </div>
    </div>
  );
}

export function FilesTab({ files, onOpen }: { files: string[]; onOpen: (path: string) => void }) {
  const documents = files.filter((f) => !f.includes("/"));
  const folders = new Map<string, string[]>();
  files
    .filter((f) => f.includes("/"))
    .forEach((f) => {
      const folder = f.slice(0, f.indexOf("/"));
      folders.set(folder, [...(folders.get(folder) ?? []), f]);
    });

  return (
    <div className="files">
      <a className="download" href="/project/download">
        Download the project (.zip)
      </a>
      {files.length === 0 && <p className="dim">No files yet.</p>}
      {documents.length > 0 && <FileGroup title="Documents" files={documents} onOpen={onOpen} />}
      {[...folders.entries()].map(([folder, paths]) => (
        <FileGroup key={folder} title={folder === "code" ? "Code" : folder === "shared" ? "Shared by you" : folder} files={paths} onOpen={onOpen} />
      ))}
    </div>
  );
}

function FileGroup({ title, files, onOpen }: { title: string; files: string[]; onOpen: (path: string) => void }) {
  return (
    <section className="card">
      <h3>{title}</h3>
      <ul className="file-list">
        {files.map((path) => (
          <li key={path}>
            <button onClick={() => onOpen(path)}>{path}</button>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function FileViewer({ path, onClose }: { path: string; onClose: () => void }) {
  const image = IMAGE.test(path);
  const text = useFile(image ? null : path, 0);

  return (
    <div className="viewer-backdrop" onClick={onClose}>
      <div className="viewer" onClick={(event) => event.stopPropagation()}>
        <header>
          <span className="mono">{path}</span>
          <a className="chip" href={fileUrl(path)} download>
            Download
          </a>
          <button className="chip" onClick={onClose}>
            Close
          </button>
        </header>
        {image ? (
          <img src={fileUrl(path)} alt={path} />
        ) : path.endsWith(".md") && text ? (
          <div className="viewer-body">
            <Markdown text={text} />
          </div>
        ) : (
          <pre className="viewer-body">{text === undefined ? "Loading…" : (text ?? "Could not open this file.")}</pre>
        )}
      </div>
    </div>
  );
}
