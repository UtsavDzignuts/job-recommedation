"use client";

import ReactMarkdown from "react-markdown";

interface MarkdownContentProps {
  content: string;
  className?: string;
}

export default function MarkdownContent({ content, className = "" }: MarkdownContentProps) {
  return (
    <div className={`prose prose-sm max-w-none ${className}`}>
      <ReactMarkdown
        components={{
          h1: ({ children }) => (
            <h1 className="text-xl font-bold text-gray-900 mt-4 mb-2">{children}</h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-lg font-bold text-gray-900 mt-3 mb-2">{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-base font-semibold text-gray-800 mt-3 mb-1">{children}</h3>
          ),
          p: ({ children }) => (
            <p className="text-sm text-gray-700 leading-relaxed mb-2">{children}</p>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-gray-900">{children}</strong>
          ),
          em: ({ children }) => (
            <em className="italic text-gray-600">{children}</em>
          ),
          ul: ({ children }) => (
            <ul className="list-disc list-inside space-y-1 mb-3 text-sm text-gray-700">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-inside space-y-1 mb-3 text-sm text-gray-700">{children}</ol>
          ),
          li: ({ children }) => (
            <li className="leading-relaxed">{children}</li>
          ),
          code: ({ children }) => (
            <code className="bg-gray-100 text-gray-800 rounded px-1.5 py-0.5 text-xs font-mono">
              {children}
            </code>
          ),
          pre: ({ children }) => (
            <pre className="bg-gray-50 border border-gray-100 rounded-lg p-3 overflow-x-auto text-xs mb-3">
              {children}
            </pre>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-indigo-200 pl-4 py-1 my-2 bg-indigo-50/30 rounded-r-lg">
              {children}
            </blockquote>
          ),
          a: ({ href, children }) => (
            <a href={href} className="text-indigo-600 hover:text-indigo-800 underline" target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
