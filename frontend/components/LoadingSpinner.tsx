export default function LoadingSpinner({ label = "Loading..." }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 py-6 animate-fade-in">
      <div className="relative">
        <div className="w-5 h-5 border-2 border-indigo-200 rounded-full" />
        <div className="absolute top-0 left-0 w-5 h-5 border-2 border-indigo-600 rounded-full border-t-transparent animate-spin" />
      </div>
      <span className="text-sm text-gray-500 font-medium">{label}</span>
    </div>
  );
}
