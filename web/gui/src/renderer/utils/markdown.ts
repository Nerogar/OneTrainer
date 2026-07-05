import DOMPurify from "dompurify";
import { marked } from "marked";

marked.setOptions({ gfm: true, breaks: false });

const SANITIZE_CONFIG: DOMPurify.Config = {
  ALLOWED_TAGS: [
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "a",
    "img",
    "ul",
    "ol",
    "li",
    "code",
    "pre",
    "em",
    "strong",
    "del",
    "blockquote",
    "table",
    "tr",
    "td",
    "th",
    "thead",
    "tbody",
    "br",
    "hr",
    "div",
    "span",
  ],
  ALLOWED_ATTR: ["href", "src", "alt", "class", "id", "loading", "target", "rel"],
  ALLOW_DATA_ATTR: false,
};

DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A") {
    const href = node.getAttribute("href") ?? "";
    const isExternal = href.startsWith("http://") || href.startsWith("https://") || href.startsWith("//");
    if (isExternal) {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener noreferrer");
    }
  }
});

export function sanitizeHtml(dirty: string): string {
  return DOMPurify.sanitize(dirty, SANITIZE_CONFIG as Parameters<typeof DOMPurify.sanitize>[1]);
}

export function markdownToHtml(md: string): string {
  return marked.parse(md, { async: false });
}
