import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "AI Intelligence Layer - Job Board Platform",
  description:
    "AI-powered job board with RAG Q&A, recommendations, description improvement, and autonomous agent.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-indigo-50/30 text-gray-900">
        <Navbar />
        {children}
      </body>
    </html>
  );
}
