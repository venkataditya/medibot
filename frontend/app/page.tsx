"use client";

import { useRouter } from "next/navigation";
import { type FormEvent, useState } from "react";

import { ApiError, login } from "@/lib/api";
import { saveSession } from "@/lib/session";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      saveSession(await login(username.trim(), password));
      router.push("/chat");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong. Please try again.");
      setBusy(false);
    }
  }

  return (
    <main className="login">
      <form className="card" onSubmit={onSubmit}>
        <div className="brand">
          <div className="logo">M</div>
          <h2>MediBot</h2>
        </div>
        <p className="sub">MediAssist Health Network · internal assistant</p>
        <label htmlFor="username">Username</label>
        <input id="username" className="input" value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" required />
        <label htmlFor="password">Password</label>
        <input id="password" className="input" type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required />
        <button className="btn" type="submit" disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
        {error && <div className="form-error" role="alert">{error}</div>}
      </form>
    </main>
  );
}
