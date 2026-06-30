"use client";

import { useState } from "react";
import PageShell from "@/components/PageShell";
import LoadingSpinner from "@/components/LoadingSpinner";
import ErrorAlert from "@/components/ErrorAlert";
import MarkdownContent from "@/components/MarkdownContent";
import { useApiRequest } from "@/hooks/useApiRequest";
import { recommend } from "@/lib/api-client";
import type { RecommendationResponse } from "@/lib/types";

const SAMPLE_RESUMES = [
  {
    label: "Python Backend Developer",
    text: "5 years of experience as a Python backend developer. Proficient in FastAPI, Django, PostgreSQL, Docker, and AWS. Built microservices handling 10M+ requests/day. Experience with Redis, Celery, and CI/CD pipelines. Looking for senior backend roles at product companies.",
  },
  {
    label: "Frontend React Developer",
    text: "3 years building responsive web applications with React, TypeScript, and Next.js. Strong knowledge of Tailwind CSS, component design systems, and accessibility standards. Experience with REST APIs and GraphQL. Seeking roles at design-forward companies.",
  },
  {
    label: "Data Scientist (ML/NLP)",
    text: "PhD in Computer Science with 4 years of industry experience in machine learning and NLP. Published research in ACL/EMNLP. Skilled in Python, PyTorch, TensorFlow, scikit-learn, and LangChain. Built recommendation systems and RAG pipelines in production.",
  },
  {
    label: "Fresh Graduate",
    text: "Recent CS graduate from University of Illinois. Completed 3 internships in backend (Python/Flask), frontend (React), and data engineering. Strong foundations in data structures, algorithms, SQL, and Git. Eager to learn and grow in a fast-paced environment.",
  },
];

export default function RecommendPage() {
  const [resumeText, setResumeText] = useState("");
  const { data, error, loading, execute, dismiss } =
    useApiRequest<RecommendationResponse>();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!resumeText.trim()) return;
    execute(() => recommend(resumeText.trim()));
  };

  const handleSampleResume = (text: string) => {
    setResumeText(text);
  };

  return (
    <PageShell title="Job Recommendations" subtitle="Paste your resume or profile description to get AI-powered job recommendations.">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <div className="flex items-center justify-between mb-2">
            <label htmlFor="resume" className="block text-sm font-semibold text-gray-700">
              Resume / Profile Text
            </label>
          </div>
          <textarea
            id="resume"
            value={resumeText}
            onChange={(e) => setResumeText(e.target.value)}
            maxLength={10000}
            rows={8}
            placeholder="Paste your resume or describe your skills, experience, and what you're looking for..."
            className="w-full rounded-xl border border-gray-200 bg-white px-4 py-3 text-sm shadow-sm focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 outline-none transition-all resize-y placeholder:text-gray-400"
          />
          <div className="flex items-center justify-between mt-1.5">
            <p className="text-xs text-gray-400">
              {resumeText.length.toLocaleString()}/10,000 characters
            </p>
          </div>
        </div>

        {/* Sample resume buttons */}
        {!resumeText && (
          <div>
            <p className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">Quick fill with sample</p>
            <div className="flex flex-wrap gap-2">
              {SAMPLE_RESUMES.map((sample) => (
                <button
                  key={sample.label}
                  type="button"
                  onClick={() => handleSampleResume(sample.text)}
                  className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-xs text-gray-600 hover:border-violet-300 hover:text-violet-700 hover:bg-violet-50/50 transition-all duration-200 shadow-sm"
                >
                  {sample.label}
                </button>
              ))}
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={loading || !resumeText.trim()}
          className="rounded-xl bg-gradient-to-r from-violet-600 to-purple-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-purple-200/50 hover:shadow-purple-300/60 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 transition-all duration-200"
        >
          {loading ? "Finding matches..." : "Get Recommendations"}
        </button>
      </form>

      {loading && <LoadingSpinner label="Analyzing your profile and matching jobs..." />}

      {error && (
        <ErrorAlert
          message={error}
          onDismiss={dismiss}
          onRetry={handleSubmit as unknown as () => void}
        />
      )}

      {data && (
        <div className="mt-8 animate-slide-up">
          {data.recommendations.length === 0 ? (
            <div className="rounded-2xl border border-amber-100 bg-amber-50/50 p-6 text-center">
              <p className="text-sm text-amber-700">
                {data.message || "No matching jobs found for your profile. Try a more detailed description."}
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-bold text-gray-700">
                  Top Recommendations
                </h2>
                <span className="text-xs font-medium text-gray-400 bg-gray-100 rounded-full px-3 py-1">
                  {data.recommendations.length} match{data.recommendations.length !== 1 && "es"}
                </span>
              </div>
              {data.recommendations.map((rec, i) => (
                <div
                  key={i}
                  className="rounded-xl border border-gray-100 bg-white p-5 shadow-sm hover:shadow-md transition-all duration-200"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-start gap-3 flex-1">
                      <span className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-purple-500 text-white text-sm font-bold flex items-center justify-center shadow-sm mt-0.5">
                        {i + 1}
                      </span>
                      <div className="flex-1">
                        <h3 className="font-semibold text-gray-900">{rec.job_title}</h3>
                        <p className="text-xs text-gray-400 mt-0.5">ID: {rec.job_id}</p>
                      </div>
                    </div>
                    <div className="flex-shrink-0">
                      <div className="relative">
                        <svg className="w-12 h-12" viewBox="0 0 36 36">
                          <path
                            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                            fill="none"
                            stroke="#e5e7eb"
                            strokeWidth="3"
                          />
                          <path
                            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                            fill="none"
                            stroke={rec.confidence_score >= 0.8 ? "#10b981" : rec.confidence_score >= 0.6 ? "#f59e0b" : "#6b7280"}
                            strokeWidth="3"
                            strokeDasharray={`${rec.confidence_score * 100}, 100`}
                            strokeLinecap="round"
                          />
                        </svg>
                        <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-gray-700">
                          {(rec.confidence_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="mt-3 ml-11">
                    <MarkdownContent content={rec.match_reason} className="text-gray-600" />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </PageShell>
  );
}
