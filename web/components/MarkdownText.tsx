/**
 * Lightweight markdown renderer for ARIA chat bubbles.
 *
 * Handles the subset Claude actually emits: ## headings, **bold**, *italic*,
 * `inline code`, ``` code blocks, | tables |, - lists, > blockquotes.
 * No external deps. During streaming the parent should render plain text
 * and switch to this component only on finalize (avoids re-parsing every
 * streaming chunk).
 */

import React from "react";

function renderInline(line: string, keyPrefix: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  let i = 0;
  let buf = "";
  let key = 0;

  const flush = () => {
    if (buf) { out.push(buf); buf = ""; }
  };
  const push = (el: React.ReactNode) => {
    flush();
    out.push(<React.Fragment key={`${keyPrefix}-${key++}`}>{el}</React.Fragment>);
  };

  while (i < line.length) {
    // `inline code`
    if (line[i] === "`") {
      const end = line.indexOf("`", i + 1);
      if (end > i) {
        push(<code className="rounded bg-slate-800 px-1 py-0.5 text-[0.85em] font-mono text-sky-200">{line.slice(i + 1, end)}</code>);
        i = end + 1; continue;
      }
    }
    // **bold**
    if (line[i] === "*" && line[i + 1] === "*") {
      const end = line.indexOf("**", i + 2);
      if (end > i) {
        push(<strong className="font-semibold text-slate-100">{line.slice(i + 2, end)}</strong>);
        i = end + 2; continue;
      }
    }
    // *italic*
    if (line[i] === "*" && line[i + 1] !== "*" && line[i + 1] && line[i + 1] !== " ") {
      const end = line.indexOf("*", i + 1);
      if (end > i + 1) {
        push(<em className="italic text-slate-300">{line.slice(i + 1, end)}</em>);
        i = end + 1; continue;
      }
    }
    // [link](url) — render as plain label (no navigation)
    if (line[i] === "[") {
      const bracket = line.indexOf("]", i + 1);
      if (bracket > i && line[bracket + 1] === "(") {
        const paren = line.indexOf(")", bracket + 2);
        if (paren > bracket) {
          const label = line.slice(i + 1, bracket);
          push(<span className="text-sky-400 underline underline-offset-2">{label}</span>);
          i = paren + 1; continue;
        }
      }
    }
    buf += line[i]; i++;
  }
  flush();
  return out;
}

function parseTableRow(line: string): string[] {
  return line
    .replace(/^\|/, "").replace(/\|$/, "")
    .split("|")
    .map((c) => c.trim());
}

function isSeparatorRow(row: string[]): boolean {
  return row.every((c) => /^[-: ]+$/.test(c));
}

export function MarkdownText({ text }: { text: string }) {
  if (!text) return null;

  const lines = text.split("\n");
  const blocks: React.ReactNode[] = [];
  let i = 0;
  let listBuf: string[] | null = null;
  let tableBuf: string[][] | null = null;
  let codeBuf: string[] | null = null;
  let codeLang = "";

  const flushList = () => {
    if (!listBuf) return;
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="my-1.5 ml-4 list-disc space-y-0.5 text-sm text-slate-200">
        {listBuf.map((it, k) => (
          <li key={k}>{renderInline(it, `li-${blocks.length}-${k}`)}</li>
        ))}
      </ul>
    );
    listBuf = null;
  };

  const flushTable = () => {
    if (!tableBuf || tableBuf.length < 2) { tableBuf = null; return; }
    const [head, sep, ...rows] = tableBuf;
    if (!isSeparatorRow(sep ?? [])) {
      // No separator — treat as list of plain lines
      tableBuf = null; return;
    }
    blocks.push(
      <div key={`tbl-${blocks.length}`} className="my-2 overflow-x-auto">
        <table className="w-full min-w-max border-collapse text-sm">
          <thead>
            <tr>
              {head.map((c, k) => (
                <th key={k} className="border border-slate-700 bg-slate-800/70 px-2 py-1 text-left font-semibold text-slate-200">
                  {renderInline(c, `th-${k}`)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, ri) => (
              <tr key={ri} className={ri % 2 === 0 ? "bg-slate-900/40" : "bg-slate-900/20"}>
                {r.map((c, k) => (
                  <td key={k} className="border border-slate-700/60 px-2 py-1 text-slate-300">
                    {renderInline(c, `td-${ri}-${k}`)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
    tableBuf = null;
  };

  const flushCode = () => {
    if (!codeBuf) return;
    blocks.push(
      <pre key={`code-${blocks.length}`} className="my-2 overflow-x-auto rounded-md border border-slate-700 bg-slate-900/80 p-2 text-xs text-slate-300">
        <code className={codeLang ? `language-${codeLang}` : ""}>{codeBuf.join("\n")}</code>
      </pre>
    );
    codeBuf = null; codeLang = "";
  };

  while (i < lines.length) {
    const ln = lines[i];

    // ``` code block
    if (ln.startsWith("```")) {
      if (codeBuf !== null) {
        flushCode();
        i++; continue;
      }
      flushList(); flushTable();
      codeBuf = [];
      codeLang = ln.slice(3).trim();
      i++; continue;
    }
    if (codeBuf !== null) { codeBuf.push(ln); i++; continue; }

    // Table row
    if (ln.startsWith("|")) {
      flushList();
      const row = parseTableRow(ln);
      tableBuf = tableBuf ? [...tableBuf, row] : [row];
      i++; continue;
    }
    flushTable();

    // Blank line
    if (ln.trim() === "") {
      flushList();
      i++; continue;
    }

    // ### heading
    const hMatch = ln.match(/^(#{1,3})\s+(.+)/);
    if (hMatch) {
      flushList();
      const level = hMatch[1].length;
      const cls = level === 1
        ? "mt-3 mb-1 text-base font-bold text-slate-100"
        : level === 2
          ? "mt-2.5 mb-1 text-sm font-bold text-slate-200"
          : "mt-2 mb-0.5 text-sm font-semibold text-slate-300";
      const Tag = (level === 1 ? "h3" : level === 2 ? "h4" : "h5") as keyof React.JSX.IntrinsicElements;
      blocks.push(<Tag key={`h-${blocks.length}`} className={cls}>{renderInline(hMatch[2], `h${level}-${blocks.length}`)}</Tag>);
      i++; continue;
    }

    // > blockquote
    if (ln.startsWith("> ")) {
      flushList();
      blocks.push(
        <blockquote key={`bq-${blocks.length}`} className="my-1 border-l-2 border-sky-600/60 pl-3 text-sm italic text-slate-400">
          {renderInline(ln.slice(2), `bq-${blocks.length}`)}
        </blockquote>
      );
      i++; continue;
    }

    // - list item (or * -)
    if (/^[-*]\s+/.test(ln)) {
      if (!listBuf) listBuf = [];
      listBuf.push(ln.replace(/^[-*]\s+/, ""));
      i++; continue;
    }

    // Numbered list  1. ...
    if (/^\d+\.\s+/.test(ln)) {
      flushList();
      // Render as a plain paragraph for now (numbered lists are rare in ARIA output)
      blocks.push(
        <p key={`ol-${blocks.length}`} className="my-0.5 text-sm text-slate-200">
          {renderInline(ln, `ol-${blocks.length}`)}
        </p>
      );
      i++; continue;
    }

    // Horizontal rule
    if (/^---+$/.test(ln.trim())) {
      flushList();
      blocks.push(<hr key={`hr-${blocks.length}`} className="my-2 border-slate-700" />);
      i++; continue;
    }

    // Default paragraph
    flushList();
    blocks.push(
      <p key={`p-${blocks.length}`} className="my-0.5 text-sm leading-relaxed text-slate-200">
        {renderInline(ln, `p-${blocks.length}`)}
      </p>
    );
    i++;
  }

  flushList();
  flushTable();
  flushCode();

  return <div className="md-body space-y-0.5">{blocks}</div>;
}
