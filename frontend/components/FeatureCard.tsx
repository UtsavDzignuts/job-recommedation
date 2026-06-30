import Link from "next/link";

interface FeatureCardProps {
  title: string;
  description: string;
  href: string;
  icon: string;
  gradient: string;
}

export default function FeatureCard({ title, description, href, icon, gradient }: FeatureCardProps) {
  return (
    <Link
      href={href}
      className="group relative block rounded-2xl border border-gray-200/60 bg-white p-6 shadow-sm hover:shadow-xl hover:shadow-gray-200/40 hover:-translate-y-1 transition-all duration-300 overflow-hidden"
    >
      {/* Gradient accent top bar */}
      <div className={`absolute top-0 left-0 right-0 h-1 ${gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />

      <div className={`inline-flex items-center justify-center w-12 h-12 rounded-xl ${gradient} mb-4 shadow-lg`}>
        <span className="text-2xl filter drop-shadow-sm">{icon}</span>
      </div>

      <h3 className="text-lg font-semibold text-gray-900 group-hover:text-indigo-700 transition-colors">
        {title}
      </h3>
      <p className="mt-2 text-sm text-gray-500 leading-relaxed">{description}</p>

      <div className="mt-4 flex items-center text-sm font-medium text-indigo-600 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
        <span>Explore</span>
        <svg className="ml-1 w-4 h-4 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </div>
    </Link>
  );
}
