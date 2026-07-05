import { useCallback, useEffect, useMemo, useState } from "react";

import { configApi } from "@/api/configApi";
import { markdownToHtml, sanitizeHtml } from "@/utils/markdown";

interface WikiSection {
  title: string;
  pages: string[];
}

function formatPageName(slug: string): string {
  return slug.replace(/-/g, " ").replace("F.A.Q.", "FAQ");
}

const FALLBACK_SECTIONS: WikiSection[] = [
  {
    title: "Getting Started",
    pages: ["Home", "Onboarding-Guide-for-Newcomers", "The-Program"],
  },
  {
    title: "Configuration",
    pages: ["General", "Model", "Data", "Concepts", "Training"],
  },
  {
    title: "Training",
    pages: ["Optimizers", "Advanced-Optimizers", "Custom-Scheduler"],
  },
  {
    title: "Methods",
    pages: ["LoRA", "Embedding", "Additional-Embeddings"],
  },
  {
    title: "Guides & FAQ",
    pages: ["F.A.Q.", "Lessons-Learnt-and-Tutorials"],
  },
];

export default function HelpPage() {
  const [sections, setSections] = useState<WikiSection[]>([]);
  const [activePage, setActivePage] = useState<string>("Home");
  const [content, setContent] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch the page list on mount
  useEffect(() => {
    let cancelled = false;
    const fetchPages = async () => {
      try {
        const data = await configApi.wikiPages();
        if (!cancelled) {
          setSections(data as WikiSection[]);
        }
      } catch {
        if (!cancelled) {
          setSections(FALLBACK_SECTIONS);
        }
      }
    };
    fetchPages();
    return () => {
      cancelled = true;
    };
  }, []);

  // Fetch a specific page
  const fetchPage = useCallback(async (slug: string) => {
    setLoading(true);
    setError(null);
    setActivePage(slug);
    try {
      const data = await configApi.wikiPage(slug);
      setContent(data.content);
    } catch {
      setError(`Failed to load "${formatPageName(slug)}". Check your connection and try again.`);
      setContent("");
    } finally {
      setLoading(false);
    }
  }, []);

  // Load the default page on mount (or when sections arrive)
  useEffect(() => {
    if (sections.length > 0 && sections[0].pages.length > 0) {
      fetchPage(sections[0].pages[0]);
    }
  }, [sections, fetchPage]);

  // Convert markdown content to HTML, then sanitize to prevent XSS
  const renderedHtml = useMemo(() => {
    if (!content) return "";
    const rawHtml = markdownToHtml(content);
    return sanitizeHtml(rawHtml);
  }, [content]);

  return (
    <div className="flex gap-0 min-h-[calc(100vh-200px)]">
      <nav className="flex-shrink-0 border-r w-[260px] border-[var(--color-border-subtle)] bg-[var(--color-surface-raised)] rounded-l-[var(--radius-sm)] overflow-y-auto">
        <div className="py-4">
          <h3 className="text-sm font-semibold uppercase tracking-wider px-4 pb-3 m-0 text-[var(--color-on-surface-secondary)] border-b border-[var(--color-border-subtle)]">
            Documentation
          </h3>
          {sections.map((section) => (
            <div key={section.title} className="mt-3">
              <div className="text-xs font-semibold uppercase tracking-wider px-4 py-1 pb-1.5 text-[var(--color-on-surface-secondary)] opacity-70">
                {section.title}
              </div>
              <ul className="list-none m-0 p-0">
                {section.pages.map((slug) => (
                  <li key={slug}>
                    <button
                      onClick={() => fetchPage(slug)}
                      className={`help-nav-link text-sm block w-full text-left py-[5px] px-4 pl-6 border-none cursor-pointer transition-[background-color,color] duration-200 ease-out ${
                        activePage === slug ? "help-nav-link-active" : ""
                      }`}
                    >
                      {formatPageName(slug)}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </nav>

      <div className="flex-1 bg-[var(--color-surface-raised)] rounded-r-[var(--radius-sm)] px-8 py-6 overflow-y-auto min-w-0">
        {loading && (
          <div className="text-center py-12">
            <div className="skeleton w-[200px] h-6 mx-auto mb-4 rounded-md" />
            <div className="skeleton w-4/5 h-4 mx-auto mb-2 rounded" />
            <div className="skeleton w-3/5 h-4 mx-auto mb-2 rounded" />
            <div className="skeleton w-[70%] h-4 mx-auto rounded" />
          </div>
        )}

        {error && !loading && (
          <div className="p-6 bg-[var(--color-error-500-alpha-08)] border border-[var(--color-error-500-alpha-20)] rounded-[var(--radius-sm)] text-[var(--color-error-500)] text-center">
            <p className="m-0 mb-3 font-medium">{error}</p>
            <button onClick={() => fetchPage(activePage)} className="action-button text-caption">
              Retry
            </button>
          </div>
        )}

        {!loading && !error && renderedHtml && (
          <div className="wiki-content" dangerouslySetInnerHTML={{ __html: renderedHtml }} />
        )}

        {!loading && !error && !renderedHtml && (
          <div className="text-center py-12 text-[var(--color-on-surface-secondary)]">
            <p>Select a page from the sidebar to get started.</p>
          </div>
        )}
      </div>
    </div>
  );
}
