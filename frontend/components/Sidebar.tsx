import type { Session } from "@/lib/api";
import { ALL_COLLECTIONS, roleLabel } from "@/lib/roles";

export function Sidebar({ session, onSignOut }: { session: Session; onSignOut: () => void }) {
  return (
    <aside className="side">
      <div className="brand">
        <div className="logo">M</div>
        <h2>MediBot</h2>
      </div>
      <div>
        <span className={`role ${session.role}`}>{roleLabel(session.role)}</span>
        <div className="who">signed in as {session.username}</div>
      </div>
      <div>
        <h4>Collections you can access</h4>
        <div className="chips">
          {ALL_COLLECTIONS.map((c) => (
            <span key={c} className={session.collections.includes(c) ? "chip" : "chip off"}>
              {c}
            </span>
          ))}
        </div>
      </div>
      <div className="sql-note">
        Analytics (claims &amp; tickets):{" "}
        <b>{session.sql_access ? "available" : "not available for this role"}</b>
      </div>
      <button type="button" className="signout" onClick={onSignOut}>
        Sign out
      </button>
    </aside>
  );
}
