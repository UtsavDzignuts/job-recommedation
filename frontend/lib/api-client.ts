import type {
  AskAIResponse,
  RecommendationResponse,
  ImproveDescriptionResponse,
  ImprovementMode,
  AgentResponse,
  SyncReport,
  ApiError,
} from "./types";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export class ApiClientError extends Error {
  constructor(
    public status: number,
    public body: ApiError | null,
    message: string
  ) {
    super(message);
    this.name = "ApiClientError";
  }
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new ApiClientError(
      0,
      null,
      "Backend is unreachable. Please check your network connection."
    );
  }

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const message =
      body?.message ??
      body?.detail?.message ??
      body?.detail ??
      `Request failed with status ${response.status}`;
    throw new ApiClientError(response.status, body, message);
  }

  return response.json() as Promise<T>;
}

export function askAI(query: string): Promise<AskAIResponse> {
  return request<AskAIResponse>(
    `${BASE_URL}/ask-ai?query=${encodeURIComponent(query)}`
  );
}

export function recommend(resumeText: string): Promise<RecommendationResponse> {
  return request<RecommendationResponse>(`${BASE_URL}/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume_text: resumeText }),
  });
}

export function improveDescription(
  description: string,
  mode: ImprovementMode
): Promise<ImproveDescriptionResponse> {
  return request<ImproveDescriptionResponse>(
    `${BASE_URL}/improve-description`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ description, mode }),
    }
  );
}

export function agentTask(task: string): Promise<AgentResponse> {
  return request<AgentResponse>(`${BASE_URL}/agent/task`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task }),
  });
}

export function syncFull(): Promise<SyncReport> {
  return request<SyncReport>(`${BASE_URL}/sync/full`, { method: "POST" });
}
