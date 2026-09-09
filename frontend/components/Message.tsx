import type { ReactNode } from "react";

import type { ChatReply } from "@/lib/api";

// The model answers in light markdown: **bold**, [n] citations, "- " bullets.
function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|\[\d+\])/g).map((part, i) => {
    if (part.startsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (/^\[\d+\]$/.test(part)) return <span key={i} className="cite">{part}</span>;
    return part;
  });
}

function blocks(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  let bullets: string[] = [];
  const flush = () => {
    if (bullets.length) {
      out.push(<ul key={`ul${out.length}`}>{bullets.map((b, i) => <li key={i}>{inline(b)}</li>)}</ul>);
      bullets = [];
    }
  };
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (/^[-•*]\s+/.test(trimmed)) bullets.push(trimmed.replace(/^[-•*]\s+/, ""));
    else {
      flush();
      out.push(<p key={`p${out.length}`}>{inline(trimmed)}</p>);
    }
  }
  flush();
  return out;
}

export function UserMessage({ text }: { text: string }) {
  return <div className="msg user">{text}</div>;
}

export function BotMessage({ reply }: { reply: ChatReply }) {
  if (reply.access_denied) {
    return (
      <div className="msg bot deny">
        <span className="tag deny">Access restricted</span>
        {blocks(reply.answer)}
      </div>
    );
  }
  const isSql = reply.retrieval_type === "sql_rag";
  return (
    <div className="msg bot">
      <span className={`tag ${isSql ? "sql" : "hybrid"}`}>{isSql ? "SQL RAG" : "Hybrid RAG"}</span>
      {blocks(reply.answer)}
      {reply.sources.length > 0 && (
        <div className="sources">
          <div className="sources-title">Sources</div>
          {reply.sources.map((s, i) => (
            <div key={i} className="source">
              <span className="doc">{s.source_document}</span>
              <span>{s.section_title}</span>
              <span className="col">{s.collection}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function ErrorMessage({ text }: { text: string }) {
  return <div className="msg bot error">{text}</div>;
}
