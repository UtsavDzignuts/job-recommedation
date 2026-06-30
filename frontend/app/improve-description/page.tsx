"use client";

import { useState } from "react";
import PageShell from "@/components/PageShell";
import LoadingSpinner from "@/components/LoadingSpinner";
import ErrorAlert from "@/components/ErrorAlert";
import CopyButton from "@/components/CopyButton";
import MarkdownContent from "@/components/MarkdownContent";
import { useApiRequest } from "@/hooks/useApiRequest";
import { improveDescription } from "@/lib/api-client";
import type { ImproveDescriptionResponse, ImprovementMode } from "@/lib/types";

const MODE_OPTIONS: { value: ImprovementMode; label: string; description: string }[] = [
  { value: "short_and_crisp", label: "Short & Crisp", description: "Concise and scannable" },
  { value: "detailed_and_formal", label: "Detailed & Formal", description: "Professional tone" },
  { value: "marketing_oriented", label: "Marketing Oriented", description: "Compelling and engaging" },
];

export default function ImproveDescriptionPage() {
  const [description, setDescription] = useState("");
  const [mode, setMode] = useState<ImprovementMode>("short_and_crisp");
  const { data, error, loading, execute, dismiss } =
    useApiRequest<ImproveDescriptionResponse>();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!description.trim()) return;
    execute(() => improveDescription(description.trim(), mode));
  };

  return (
    <PageShell title="Improve Job Description" subtitle="Paste a raw job description and choose a style — the AI will rewrite it for you.">
      <form onSubmit={handleSubmit} className="space-y-5">
        <div>
          <label htmlFor="description" className="block text-sm font-semibold text-gray-700 mb-2">
            Job Description
          </label>
          <textarea
            id="description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            maxLength={50000}
            rows={8}
            placeholder="Paste the raw job description here..."
            className="w-full rounded-xl border border-gray-200 bg-white px-4 py-3 text-sm shadow-sm focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 outline-none transition-all resize-y placeholder:text-gray-400"
          />
          <p className="mt-1.5 text-xs text-gray-400">
            {description.length.toLocaleString()}/50,000 characters
          </p>
        </div>

        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Style
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {MODE_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => setMode(opt.value)}
                className={`rounded-xl border p-3.5 text-left transition-all duration-200 ${
                  mode === opt.value
                    ? "border-indigo-300 bg-indigo-50/50 ring-2 ring-indigo-100 shadow-sm"
                    : "border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50/50"
                }`}
              >
                <p className={`text-sm font-semibold ${mode === opt.value ? "text-indigo-700" : "text-gray-700"}`}>
                  {opt.label}
                </p>
                <p className="text-xs text-gray-500 mt-0.5">{opt.description}</p>
              </button>
            ))}
          </div>
        </div>

        <button
          type="submit"
          disabled={loading || !description.trim()}
          className="rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-orange-200/50 hover:shadow-orange-300/60 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 transition-all duration-200"
        >
          {loading ? "Improving..." : "Improve Description"}
        </button>
      </form>

      {loading && <LoadingSpinner label="Rewriting description..." />}

      {error && (
        <ErrorAlert
          message={error}
          onDismiss={dismiss}
          onRetry={handleSubmit as unknown as () => void}
        />
      )}

      {data && (
        <div className="mt-8 space-y-3 animate-slide-up">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-gray-700">
              Improved Version{" "}
              <span className="font-normal text-gray-400">
                • {MODE_OPTIONS.find((o) => o.value === data.mode)?.label}
              </span>
            </h2>
            <CopyButton text={data.improved_description} />
          </div>
          <div className="rounded-2xl border border-emerald-100 bg-gradient-to-br from-emerald-50 to-teal-50/50 p-6">
            <MarkdownContent content={data.improved_description} />
          </div>
        </div>
      )}
    </PageShell>
  );
}
