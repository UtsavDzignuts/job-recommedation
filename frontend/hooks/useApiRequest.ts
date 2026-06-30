"use client";

import { useCallback, useState } from "react";

interface ApiRequestState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

export function useApiRequest<T>() {
  const [state, setState] = useState<ApiRequestState<T>>({
    data: null,
    error: null,
    loading: false,
  });

  const execute = useCallback(async (fn: () => Promise<T>) => {
    setState({ data: null, error: null, loading: true });
    try {
      const data = await fn();
      setState({ data, error: null, loading: false });
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "An unexpected error occurred.";
      setState({ data: null, error: message, loading: false });
    }
  }, []);

  const dismiss = useCallback(() => {
    setState((prev) => ({ ...prev, error: null }));
  }, []);

  return { ...state, execute, dismiss };
}
