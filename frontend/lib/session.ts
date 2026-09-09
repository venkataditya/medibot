import { useMemo, useSyncExternalStore } from "react";

import type { Session } from "./api";

const KEY = "medibot.session";

// sessionStorage: the token dies with the tab, which is the right default for a
// shared hospital workstation.
export function saveSession(session: Session): void {
  window.sessionStorage.setItem(KEY, JSON.stringify(session));
}

export function clearSession(): void {
  window.sessionStorage.removeItem(KEY);
}

function readRaw(): string | null {
  try {
    return window.sessionStorage.getItem(KEY);
  } catch {
    return null;
  }
}

const noSubscribe = () => () => {};

/** The stored session, or null during server render and when signed out. */
export function useStoredSession(): Session | null {
  const raw = useSyncExternalStore(noSubscribe, readRaw, () => null);
  return useMemo(() => {
    if (!raw) return null;
    try {
      return JSON.parse(raw) as Session;
    } catch {
      return null;
    }
  }, [raw]);
}
