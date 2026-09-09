import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, chat, login } from "./api";

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => vi.unstubAllGlobals());

describe("login", () => {
  it("posts the credentials and returns the session", async () => {
    const fetchMock = mockFetch(200, {
      token: "t.o.k",
      username: "nurse.priya",
      role: "nurse",
      collections: ["general", "nursing"],
      sql_access: false,
    });
    const session = await login("nurse.priya", "nurse123");
    expect(session.role).toBe("nurse");
    expect(session.token).toBe("t.o.k");
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toMatch(/\/login$/);
    expect(JSON.parse(String(init.body))).toEqual({ username: "nurse.priya", password: "nurse123" });
  });

  it("surfaces the backend's message on a 401", async () => {
    mockFetch(401, { detail: "Invalid username or password." });
    await expect(login("x", "y")).rejects.toMatchObject({ status: 401, message: "Invalid username or password." });
  });

  it("explains when the backend is not reachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(login("x", "y")).rejects.toThrow(/reach the MediBot backend/);
  });
});

describe("chat", () => {
  it("sends the bearer token and returns the typed response", async () => {
    const fetchMock = mockFetch(200, {
      answer: "24G [1]",
      sources: [{ source_document: "icu.pdf", section_title: "Sizing", collection: "nursing" }],
      retrieval_type: "hybrid_rag",
      role: "nurse",
      access_denied: false,
    });
    const reply = await chat("tok", "cannula size?");
    expect(reply.sources[0].source_document).toBe("icu.pdf");
    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers.Authorization).toBe("Bearer tok");
    expect(JSON.parse(String(init.body))).toEqual({ question: "cannula size?" });
  });

  it("turns a 502 into the friendly upstream message", async () => {
    mockFetch(502, { detail: "Groq rate limit reached. Wait a moment and try again." });
    await expect(chat("tok", "q")).rejects.toMatchObject({ status: 502, message: /rate limit/ });
  });

  it("flattens a 422 validation detail into one sentence", async () => {
    mockFetch(422, { detail: [{ loc: ["body", "question"], msg: "Value error, question must not be blank" }] });
    await expect(chat("tok", " ")).rejects.toThrow(/question must not be blank/);
  });

  it("exposes the status so the UI can log out on 401", async () => {
    mockFetch(401, { detail: "Session token is invalid or has expired. Please log in again." });
    const err = await chat("bad", "q").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(401);
  });
});
