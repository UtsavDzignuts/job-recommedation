"use client";

interface ErrorAlertProps {
  message: string;
  onDismiss: () => void;
  onRetry?: () => void;
}

export default function ErrorAlert({ message, onDismiss, onRetry }: ErrorAlertProps) {
  return (
    <div className="rounded-xl border border-red-100 bg-red-50/50 p-4 my-4 animate-fade-in">
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-red-100 flex items-center justify-center">
          <svg className="w-4 h-4 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-red-800">Something went wrong</p>
          <p className="mt-1 text-sm text-red-600">{message}</p>
        </div>
        <button
          onClick={onDismiss}
          className="flex-shrink-0 p-1 rounded-md text-red-400 hover:text-red-600 hover:bg-red-100 transition-colors"
          aria-label="Dismiss error"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      {onRetry && (
        <div className="mt-3 ml-11">
          <button
            onClick={onRetry}
            className="rounded-lg bg-red-600 px-3.5 py-1.5 text-xs font-semibold text-white hover:bg-red-700 shadow-sm shadow-red-200 transition-all"
          >
            Try Again
          </button>
        </div>
      )}
    </div>
  );
}
