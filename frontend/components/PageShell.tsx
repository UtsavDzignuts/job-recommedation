interface PageShellProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}

export default function PageShell({ title, subtitle, children }: PageShellProps) {
  return (
    <main className="mx-auto max-w-4xl px-4 py-10 sm:px-6 lg:px-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
          {title}
        </h1>
        {subtitle && (
          <p className="mt-2 text-base text-gray-500">{subtitle}</p>
        )}
      </div>
      {children}
    </main>
  );
}
