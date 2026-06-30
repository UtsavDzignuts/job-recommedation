// --- Ask AI ---

export interface SourceReference {
  entity_type: string;
  entity_id: string;
  text_snippet: string;
  relevance_score: number;
}

export interface AskAIResponse {
  answer: string;
  sources: SourceReference[];
  query: string;
}

// --- Recommendations ---

export interface RecommendRequest {
  resume_text: string;
}

export interface JobRecommendation {
  job_title: string;
  job_id: string;
  match_reason: string;
  confidence_score: number;
}

export interface RecommendationResponse {
  recommendations: JobRecommendation[];
  message: string | null;
}

// --- Description Improvement ---

export type ImprovementMode =
  | "short_and_crisp"
  | "detailed_and_formal"
  | "marketing_oriented";

export interface ImproveDescriptionRequest {
  description: string;
  mode: ImprovementMode;
}

export interface ImproveDescriptionResponse {
  improved_description: string;
  mode: ImprovementMode;
}

// --- AI Agent ---

export interface AgentTaskRequest {
  task: string;
}

export interface ToolInvocation {
  tool_name: string;
  input: Record<string, unknown>;
  output: string;
  reasoning: string;
}

export interface AgentResponse {
  answer: string;
  steps: ToolInvocation[];
  completed: boolean;
  message: string | null;
}

// --- Sync ---

export interface SyncReport {
  total_entities: number;
  created: number;
  updated: number;
  deleted: number;
  failed: number;
  duration_seconds: number;
}

// --- Error ---

export interface ApiError {
  error: string;
  message: string;
  details?: Record<string, unknown>;
}
