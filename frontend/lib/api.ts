export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Session = {
  token: string;
  username: string;
  role: string;
  collections: string[];
  sql_access: boolean;
};

export type Source = { source_document: string; section_title: string; collection: string };

export type ChatReply = {
  answer: string;
  sources: Source[];
  retrieval_type: "hybrid_rag" | "sql_rag";
  role: string;
  access_denied: boolean;
};

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
  }
}

type ValidationItem = { loc?: unknown[]; msg?: string };

function detailToMessage(detail: unknown, status: number): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return (detail as ValidationItem[])
      .map((d) => (d.msg ?? "invalid input").replace(/^Value error, /, ""))
      .join("; ");
  }
  return `Request failed (HTTP ${status}).`;
}

async function request<T>(path: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError(`Cannot reach the MediBot backend at ${API_URL}. Is it running?`, 0);
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new ApiError(detailToMessage((body as { detail?: unknown }).detail, response.status), response.status);
  }
  return body as T;
}

export function login(username: string, password: string): Promise<Session> {
  return request<Session>("/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
}

export function chat(token: string, question: string): Promise<ChatReply> {
  return request<ChatReply>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({ question }),
  });
}
