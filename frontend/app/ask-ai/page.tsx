"use client";

import { useState } from "react";
import PageShell from "@/components/PageShell";
import LoadingSpinner from "@/components/LoadingSpinner";
import ErrorAlert from "@/components/ErrorAlert";
import MarkdownContent from "@/components/MarkdownContent";
import { useApiRequest } from "@/hooks/useApiRequest";
import { askAI } from "@/lib/api-client";
import type { AskAIResponse } from "@/lib/types";

const SUGGESTED_QUERIES = [
  "List top remote Python jobs for a fresher",
  "Which companies are hiring for data science?",
  "Find senior backend roles paying over $150k",
  "What DevOps or cloud engineering positions are available?",
  "Show me candidates with machine learning experience",
];

export default function AskAIPage() {
  const [query, setQuery] = useState("");
  const { data, error, loading, execute, dismiss } = useApiRequest<AskAIResponse>();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    execute(() => askAI(query.trim()));
  };

  const handleSuggestedQuery = (suggested: string) => {
    setQuery(suggested);
    execute(() => askAI(suggested));
  };

  return (
    <PageShell title="Ask AI" subtitle="Ask natural language questions about jobs, companies, and candidates in the platform.">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="query" className="block text-sm font-semibold text-gray-700 mb-2">
            Your Question
          </label>
          <div className="relative">
            <input
              id="query"
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              maxLength={1000}
              placeholder="e.g., Which companies are hiring for data science?"
              className="w-full rounded-xl border border-gray-200 bg-white px-4 py-3.5 pr-16 text-sm shadow-sm focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 outline-none transition-all placeholder:text-gray-400"
            />
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-400">
              {query.length}/1000
            </span>
          </div>
        </div>

        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-200/50 hover:shadow-indigo-300/60 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 transition-all duration-200"
        >
          {loading ? "Thinking..." : "Ask Question"}
        </button>
      </form>

      {/* Suggested queries */}
      {!data && !loading && (
        <div className="mt-6">
          <p className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3">Try these</p>
          <div className="flex flex-wrap gap-2">
            {SUGGESTED_QUERIES.map((q) => (
              <button
                key={q}
                onClick={() => handleSuggestedQuery(q)}
                className="rounded-lg border border-gray-200 bg-white px-3 py-2 text-xs text-gray-600 hover:border-indigo-300 hover:text-indigo-700 hover:bg-indigo-50/50 transition-all duration-200 shadow-sm"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {loading && <LoadingSpinner label="Generating answer..." />}

      {error && <ErrorAlert message={error} onDismiss={dismiss} onRetry={handleSubmit as () => void} />}

      {data && (
        <div className="mt-8 space-y-6 animate-slide-up">
          {/* Answer card */}
          <div className="rounded-2xl border border-emerald-100 bg-gradient-to-br from-emerald-50/80 to-teal-50/50 p-6">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-7 h-7 rounded-lg bg-emerald-500 flex items-center justify-center shadow-sm">
                <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h2 className="text-sm font-bold text-emerald-800">Answer</h2>
              <span className="ml-auto text-xs text-gray-400">
                Query: &ldquo;{data.query}&rdquo;
              </span>
            </div>
            <MarkdownContent content={data.answer} />
          </div>

          {/* Sources */}
          {data.sources.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <h3 className="text-sm font-bold text-gray-700">Sources</h3>
                <span className="text-xs font-medium text-gray-400 bg-gray-100 rounded-full px-2.5 py-0.5">
                  {data.sources.length} references
                </span>
              </div>
              <div className="grid gap-2 sm:grid-cols-2">
                {data.sources.map((source, i) => (
                  <div
                    key={i}
                    className="rounded-xl border border-gray-100 bg-white p-4 shadow-sm hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`rounded-md px-2 py-0.5 text-xs font-semibold ${
                        source.entity_type === "job_post"
                          ? "bg-blue-50 text-blue-700"
                          : source.entity_type === "company"
                          ? "bg-purple-50 text-purple-700"
                          : "bg-teal-50 text-teal-700"
                      }`}>
                        {source.entity_type === "job_post" ? "Job" : source.entity_type === "company" ? "Company" : "Candidate"}
                      </span>
                      <span className="text-xs text-gray-400">ID: {source.entity_id}</span>
                      <span className="ml-auto flex items-center gap-1">
                        <span className={`inline-block w-2 h-2 rounded-full ${
                          source.relevance_score >= 0.8 ? "bg-emerald-400" :
                          source.relevance_score >= 0.6 ? "bg-amber-400" : "bg-gray-300"
                        }`} />
                        <span className="text-xs font-bold text-gray-600">
                          {(source.relevance_score * 100).toFixed(0)}%
                        </span>
                      </span>
                    </div>
                    <p className="text-xs text-gray-600 line-clamp-3 leading-relaxed">{source.text_snippet}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </PageShell>
  );
}
