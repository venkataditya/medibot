import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { clearSession, saveSession, useStoredSession } from "./session";

const session = { token: "t", username: "dr.mehta", role: "doctor", collections: ["general"], sql_access: false };

beforeEach(() => window.sessionStorage.clear());

describe("session storage", () => {
  it("round-trips a saved session through the hook", () => {
    saveSession(session);
    const { result } = renderHook(() => useStoredSession());
    expect(result.current).toEqual(session);
  });

  it("is null when nothing is stored or after sign-out", () => {
    expect(renderHook(() => useStoredSession()).result.current).toBeNull();
    saveSession(session);
    clearSession();
    expect(renderHook(() => useStoredSession()).result.current).toBeNull();
  });

  it("treats a corrupt entry as signed out instead of crashing", () => {
    window.sessionStorage.setItem("medibot.session", "{not json");
    expect(renderHook(() => useStoredSession()).result.current).toBeNull();
  });
});
