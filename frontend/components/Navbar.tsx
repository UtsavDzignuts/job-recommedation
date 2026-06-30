"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const NAV_LINKS = [
  { href: "/", label: "Dashboard", icon: "◆" },
  { href: "/ask-ai", label: "Ask AI", icon: "◎" },
  { href: "/recommend", label: "Recommendations", icon: "◈" },
  { href: "/improve-description", label: "Improve", icon: "◇" },
  { href: "/agent", label: "Agent", icon: "◉" },
  { href: "/sync", label: "Sync", icon: "◌" },
];

export default function Navbar() {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 border-b border-gray-200/60 bg-white/80 backdrop-blur-xl">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-600 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-200/50 group-hover:shadow-indigo-300/60 transition-shadow">
              <span className="text-white text-sm font-bold">AI</span>
            </div>
            <span className="text-lg font-bold bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent">
              JobBoard
            </span>
          </Link>

          {/* Desktop nav */}
          <div className="hidden md:flex md:items-center md:gap-1">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`relative rounded-lg px-3.5 py-2 text-sm font-medium transition-all duration-200 ${
                  pathname === link.href
                    ? "text-indigo-700 bg-indigo-50/80"
                    : "text-gray-500 hover:text-gray-900 hover:bg-gray-100/60"
                }`}
              >
                {link.label}
                {pathname === link.href && (
                  <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-4 h-0.5 bg-indigo-600 rounded-full" />
                )}
              </Link>
            ))}
          </div>

          {/* Mobile hamburger */}
          <button
            className="md:hidden p-2 rounded-lg text-gray-500 hover:bg-gray-100 transition-colors"
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label="Toggle navigation menu"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              {menuOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>
      </div>

      {/* Mobile menu panel */}
      {menuOpen && (
        <div className="md:hidden border-t border-gray-100 bg-white/95 backdrop-blur-xl animate-fade-in">
          <div className="px-4 py-3 space-y-1">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setMenuOpen(false)}
                className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                  pathname === link.href
                    ? "bg-indigo-50 text-indigo-700"
                    : "text-gray-600 hover:bg-gray-50"
                }`}
              >
                <span className="text-xs opacity-50">{link.icon}</span>
                {link.label}
              </Link>
            ))}
          </div>
        </div>
      )}
    </nav>
  );
}
