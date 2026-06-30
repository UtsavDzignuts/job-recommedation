"use client";

import { useState } from "react";
import PageShell from "@/components/PageShell";
import LoadingSpinner from "@/components/LoadingSpinner";
import ErrorAlert from "@/components/ErrorAlert";
import MarkdownContent from "@/components/MarkdownContent";
import { useApiRequest } from "@/hooks/useApiRequest";
import { agentTask } from "@/lib/api-client";
import type { AgentResponse } from "@/lib/types";

const SUGGESTED_TASKS = [
  "Find the top 3 jobs related to cloud computing and summarize them in bullet points.",
  "A recruiter wants a report of promising candidates for the Senior Backend role. Fetch data and generate a report.",
  "Which companies are hiring the most? Analyze and give me a summary.",
  "Find Python developer jobs and compare them with available candidates to suggest best matches.",
];

export default function AgentPage() {
  const [task, setTask] = useState("");
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());
  const { data, error, loading, execute, dismiss } = useApiRequest<AgentResponse>();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!task.trim()) return;
    setExpandedSteps(new Set());
    execute(() => agentTask(task.trim()));
  };

  const handleSuggestedTask = (suggested: string) => {
    setTask(suggested);
    setExpandedSteps(new Set());
    execute(() => agentTask(suggested));
  };

  const toggleStep = (index: number) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  return (
    <PageShell title="AI Agent" subtitle="Give the agent a complex task — it will reason step-by-step using multiple tools to find the answer.">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="task" className="block text-sm font-semibold text-gray-700 mb-2">
            Task Description
          </label>
          <textarea
            id="task"
            value={task}
            onChange={(e) => setTask(e.target.value)}
            rows={4}
            placeholder="e.g., Find the top 3 jobs related to cloud computing posted in the last 30 days and summarize them."
            className="w-full rounded-xl border border-gray-200 bg-white px-4 py-3 text-sm shadow-sm focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 outline-none transition-all resize-y placeholder:text-gray-400"
          />
        </div>

        <button
          type="submit"
          disabled={loading || !task.trim()}
          className="rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-emerald-200/50 hover:shadow-emerald-300/60 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 transition-all duration-200"
        >
          {loading ? "Agent working..." : "Run Agent"}
        </button>
      </form>

      {/* Suggested tasks */}
      {!data && !loading && (
        <div className="mt-6">
          <p className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3">Suggested tasks</p>
          <div className="grid gap-2 sm:grid-cols-2">
            {SUGGESTED_TASKS.map((t) => (
              <button
                key={t}
                onClick={() => handleSuggestedTask(t)}
                className="rounded-lg border border-gray-200 bg-white px-3 py-2.5 text-xs text-gray-600 text-left hover:border-emerald-300 hover:text-emerald-700 hover:bg-emerald-50/50 transition-all duration-200 shadow-sm leading-relaxed"
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      )}

      {loading && <LoadingSpinner label="Agent is reasoning and executing tools..." />}

      {error && (
        <ErrorAlert
          message={error}
          onDismiss={dismiss}
          onRetry={handleSubmit as unknown as () => void}
        />
      )}

      {data && (
        <div className="mt-8 space-y-6 animate-slide-up">
          {/* Completion status */}
          {!data.completed && data.message && (
            <div className="rounded-xl border border-amber-100 bg-amber-50/50 p-4">
              <p className="text-sm text-amber-700 font-medium">⚠️ {data.message}</p>
            </div>
          )}

          {/* Final answer */}
          <div className="rounded-2xl border border-emerald-100 bg-gradient-to-br from-emerald-50 to-teal-50/50 p-6">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-6 h-6 rounded-lg bg-emerald-500 flex items-center justify-center">
                <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h2 className="text-sm font-bold text-emerald-800">Final Answer</h2>
            </div>
            <MarkdownContent content={data.answer} />
          </div>

          {/* Reasoning steps */}
          {data.steps.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <h3 className="text-sm font-bold text-gray-700">Reasoning Steps</h3>
                <span className="text-xs font-medium text-gray-400 bg-gray-100 rounded-full px-2.5 py-0.5">
                  {data.steps.length}
                </span>
              </div>
              <div className="space-y-2">
                {data.steps.map((step, i) => (
                  <div
                    key={i}
                    className="rounded-xl border border-gray-100 bg-white overflow-hidden shadow-sm"
                  >
                    <button
                      onClick={() => toggleStep(i)}
                      className="w-full flex items-center justify-between px-4 py-3.5 text-left hover:bg-gray-50/50 transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        <span className="flex-shrink-0 w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-500 text-white text-xs font-bold flex items-center justify-center shadow-sm">
                          {i + 1}
                        </span>
                        <span className="text-sm font-semibold text-gray-800">
                          {step.tool_name}
                        </span>
                      </div>
                      <svg
                        className={`h-4 w-4 text-gray-400 transition-transform duration-200 ${
                          expandedSteps.has(i) ? "rotate-180" : ""
                        }`}
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </button>

                    {expandedSteps.has(i) && (
                      <div className="px-4 pb-4 space-y-3 border-t border-gray-50 pt-3 animate-fade-in">
                        <div>
                          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">Reasoning</p>
                          <p className="text-sm text-gray-600 mt-1 leading-relaxed">{step.reasoning}</p>
                        </div>
                        <div>
                          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">Input</p>
                          <pre className="text-xs text-gray-600 mt-1 bg-gray-50 rounded-lg p-3 overflow-x-auto border border-gray-100">
                            {JSON.stringify(step.input, null, 2)}
                          </pre>
                        </div>
                        <div>
                          <p className="text-xs font-bold text-gray-400 uppercase tracking-wider">Output</p>
                          <pre className="text-xs text-gray-600 mt-1 bg-gray-50 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap border border-gray-100">
                            {step.output}
                          </pre>
                        </div>
                      </div>
                    )}
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
