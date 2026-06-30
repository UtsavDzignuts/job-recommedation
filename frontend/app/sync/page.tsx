"use client";

import PageShell from "@/components/PageShell";
import LoadingSpinner from "@/components/LoadingSpinner";
import ErrorAlert from "@/components/ErrorAlert";
import { useApiRequest } from "@/hooks/useApiRequest";
import { syncFull } from "@/lib/api-client";
import type { SyncReport } from "@/lib/types";

export default function SyncPage() {
  const { data, error, loading, execute, dismiss } = useApiRequest<SyncReport>();

  const handleSync = () => {
    execute(() => syncFull());
  };

  return (
    <PageShell title="Sync Embeddings" subtitle="Trigger a full re-sync of all job board data into the vector database for up-to-date AI features.">
      {/* Info card */}
      <div className="rounded-xl border border-indigo-100 bg-indigo-50/50 p-4 mb-6">
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-indigo-100 flex items-center justify-center">
            <svg className="w-4 h-4 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-indigo-800">What does sync do?</p>
            <p className="text-xs text-indigo-600 mt-1 leading-relaxed">
              This operation generates vector embeddings for all job posts, companies, and candidates in the database, 
              then stores them in ChromaDB. This enables semantic search, RAG question answering, and AI recommendations.
              The operation is idempotent — running it multiple times is safe.
            </p>
          </div>
        </div>
      </div>

      {/* Data summary */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 mb-6">
        <p className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3">Current Data</p>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-900">25</p>
            <p className="text-xs text-gray-500">Job Posts</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-900">8</p>
            <p className="text-xs text-gray-500">Companies</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-900">12</p>
            <p className="text-xs text-gray-500">Candidates</p>
          </div>
        </div>
      </div>

      <button
        onClick={handleSync}
        disabled={loading}
        className="rounded-xl bg-gradient-to-r from-pink-600 to-rose-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-pink-200/50 hover:shadow-pink-300/60 hover:scale-[1.02] disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 transition-all duration-200"
      >
        {loading ? "Syncing..." : "🔄 Trigger Full Sync"}
      </button>

      {loading && <LoadingSpinner label="Syncing embeddings... this may take a minute." />}

      {error && <ErrorAlert message={error} onDismiss={dismiss} onRetry={handleSync} />}

      {data && (
        <div className="mt-8 animate-slide-up">
          <div className="rounded-2xl border border-emerald-100 bg-gradient-to-br from-emerald-50 to-teal-50/50 p-6">
            <div className="flex items-center gap-2 mb-5">
              <div className="w-7 h-7 rounded-lg bg-emerald-500 flex items-center justify-center shadow-sm">
                <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h2 className="text-sm font-bold text-emerald-800">Sync Complete</h2>
              <span className="ml-auto text-xs text-emerald-600 font-medium">
                {data.duration_seconds.toFixed(1)}s
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <StatItem label="Total Processed" value={data.total_entities} icon="📊" />
              <StatItem label="Created" value={data.created} icon="✨" color="text-emerald-700" />
              <StatItem label="Updated" value={data.updated} icon="🔄" color="text-indigo-700" />
              <StatItem label="Deleted" value={data.deleted} icon="🗑️" color="text-amber-700" />
              <StatItem label="Failed" value={data.failed} icon="❌" color={data.failed > 0 ? "text-red-600" : "text-gray-400"} />
              <StatItem
                label="Success Rate"
                value={data.total_entities > 0 ? `${(((data.total_entities - data.failed) / data.total_entities) * 100).toFixed(0)}%` : "N/A"}
                icon="✅"
                color="text-emerald-700"
              />
            </div>
          </div>
        </div>
      )}
    </PageShell>
  );
}

function StatItem({
  label,
  value,
  icon,
  color = "text-gray-900",
}: {
  label: string;
  value: number | string;
  icon: string;
  color?: string;
}) {
  return (
    <div className="bg-white/60 rounded-xl p-3.5 border border-emerald-100/50">
      <div className="flex items-center gap-1.5 mb-1">
        <span className="text-sm">{icon}</span>
        <p className="text-xs text-gray-500 font-medium">{label}</p>
      </div>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
    </div>
  );
}
