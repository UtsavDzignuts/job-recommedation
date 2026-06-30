import FeatureCard from "@/components/FeatureCard";

const FEATURES = [
  {
    title: "Ask AI",
    description: "Ask natural language questions about jobs, companies, and candidates in your platform.",
    href: "/ask-ai",
    icon: "🤖",
    gradient: "bg-gradient-to-br from-blue-500 to-cyan-400",
  },
  {
    title: "Job Recommendations",
    description: "Paste your resume and get AI-powered job recommendations with confidence scores.",
    href: "/recommend",
    icon: "💼",
    gradient: "bg-gradient-to-br from-violet-500 to-purple-400",
  },
  {
    title: "Improve Description",
    description: "Rewrite job descriptions with AI in different styles — short, formal, or marketing.",
    href: "/improve-description",
    icon: "✍️",
    gradient: "bg-gradient-to-br from-amber-500 to-orange-400",
  },
  {
    title: "AI Agent",
    description: "Give complex tasks to an autonomous agent that reasons step-by-step using tools.",
    href: "/agent",
    icon: "🧠",
    gradient: "bg-gradient-to-br from-emerald-500 to-teal-400",
  },
  {
    title: "Sync Embeddings",
    description: "Trigger a full sync of job board data into the vector database for up-to-date AI.",
    href: "/sync",
    icon: "🔄",
    gradient: "bg-gradient-to-br from-pink-500 to-rose-400",
  },
];

export default function Dashboard() {
  return (
    <main className="mx-auto max-w-6xl px-4 py-12 sm:px-6 lg:px-8">
      {/* Hero section */}
      <div className="text-center mb-12">
        <div className="inline-flex items-center gap-2 rounded-full bg-indigo-50 border border-indigo-100 px-4 py-1.5 text-xs font-semibold text-indigo-700 mb-4">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse-slow" />
          AI-Powered Platform
        </div>
        <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-5xl">
          AI Intelligence{" "}
          <span className="gradient-text">Layer</span>
        </h1>
        <p className="mt-4 text-lg text-gray-500 max-w-2xl mx-auto leading-relaxed">
          Your AI-powered command center for the Job Board Platform. 
          Search, recommend, improve, and automate — all in one place.
        </p>
      </div>

      {/* Feature grid */}
      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {FEATURES.map((feature) => (
          <FeatureCard key={feature.href} {...feature} />
        ))}
      </div>

      {/* Bottom stats bar */}
      <div className="mt-16 flex flex-wrap items-center justify-center gap-8 text-center">
        <div>
          <p className="text-2xl font-bold text-gray-900">25</p>
          <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Job Posts</p>
        </div>
        <div className="w-px h-8 bg-gray-200" />
        <div>
          <p className="text-2xl font-bold text-gray-900">8</p>
          <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Companies</p>
        </div>
        <div className="w-px h-8 bg-gray-200" />
        <div>
          <p className="text-2xl font-bold text-gray-900">12</p>
          <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">Candidates</p>
        </div>
        <div className="w-px h-8 bg-gray-200" />
        <div>
          <p className="text-2xl font-bold text-gray-900">5</p>
          <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">AI Features</p>
        </div>
      </div>
    </main>
  );
}
