"use client";

import { useRouter } from "next/navigation";
import { type FormEvent, useEffect, useRef, useState } from "react";

import { BotMessage, ErrorMessage, UserMessage } from "@/components/Message";
import { Sidebar } from "@/components/Sidebar";
import { ApiError, type ChatReply, chat } from "@/lib/api";
import { clearSession, useStoredSession } from "@/lib/session";

type Turn = { id: number; question: string; reply?: ChatReply; error?: string };

export default function ChatPage() {
  const router = useRouter();
  const session = useStoredSession();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!session) router.replace("/");
  }, [session, router]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  function signOut() {
    clearSession();
    router.replace("/");
  }


  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    const q = question.trim();
    if (!q || !session || busy) return;
    const id = Date.now();
    setTurns((t) => [...t, { id, question: q }]);
    setQuestion("");
    setBusy(true);
    try {
      const reply = await chat(session.token, q);
      setTurns((t) => t.map((turn) => (turn.id === id ? { ...turn, reply } : turn)));
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) return signOut();
      const message = e instanceof ApiError ? e.message : "Something went wrong. Please try again.";
      setTurns((t) => t.map((turn) => (turn.id === id ? { ...turn, error: message } : turn)));
    } finally {
      setBusy(false);
    }
  }

  if (!session) return null;

  return (
    <div className="app">
      <Sidebar session={session} onSignOut={signOut} />
      <section className="main">
        <div className="msgs">
          {turns.length === 0 && (
            <div className="empty">
              Ask about treatment protocols, the drug formulary, nursing procedures, billing codes, equipment
              manuals or HR policy. You will only see answers from the collections listed on the left.
            </div>
          )}
          {turns.map((turn) => (
            <div key={turn.id} style={{ display: "contents" }}>
              <UserMessage text={turn.question} />
              {turn.reply && <BotMessage reply={turn.reply} />}
              {turn.error && <ErrorMessage text={turn.error} />}
              {!turn.reply && !turn.error && <div className="thinking">MediBot is thinking…</div>}
            </div>
          ))}
          <div ref={bottom} />
        </div>
        <form className="ask" onSubmit={onSubmit}>
          <input
            className="input"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask MediBot…"
            aria-label="Question"
            maxLength={2000}
          />
          <button className="btn" type="submit" disabled={busy || !question.trim()}>Send</button>
        </form>
      </section>
    </div>
  );
}
