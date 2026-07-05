import DOMPurify from "dompurify";

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

export function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

export function isSafeHref(uri: string): boolean {
  // eslint-disable-next-line no-control-regex -- Intentionally strip ASCII control chars (0x00-0x1F) and space to prevent URI obfuscation attacks
  const normalized = uri.replace(/[\x00-\x20]+/g, "").toLowerCase();
  return /^https?:\/\//.test(normalized) || /^mailto:/.test(normalized) || normalized.startsWith("#");
}

export function isSafeSrc(uri: string): boolean {
  // eslint-disable-next-line no-control-regex -- Intentionally strip ASCII control chars (0x00-0x1F) and space to prevent URI obfuscation attacks
  const normalized = uri.replace(/[\x00-\x20]+/g, "").toLowerCase();
  return (
    /^https?:\/\//.test(normalized) ||
    /^data:image\/(png|jpeg|jpg|gif|webp|avif|bmp)[;,]/.test(normalized) ||
    normalized.startsWith("/api/")
  );
}

export function inlineMarkdown(line: string): string {
  const parts = line.split(/(<[^>]+>)/g);
  let result = parts
    .map((part) => {
      if (/^<[^>]+>$/.test(part)) return part;
      return escapeHtml(part);
    })
    .join("");

  result = result.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (_match, alt: string, url: string) =>
    isSafeSrc(url) ? `<img src="${url}" alt="${alt}" loading="lazy" />` : "",
  );

  result = result.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_match, text: string, url: string) =>
    isSafeHref(url) ? `<a href="${url}" target="_blank" rel="noopener noreferrer">${text}</a>` : text,
  );

  result = result.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
  result = result.replace(/___(.+?)___/g, "<strong><em>$1</em></strong>");

  result = result.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  result = result.replace(/__(.+?)__/g, "<strong>$1</strong>");

  result = result.replace(/\*(.+?)\*/g, "<em>$1</em>");
  result = result.replace(/(?<!\w)_(.+?)_(?!\w)/g, "<em>$1</em>");

  result = result.replace(/~~(.+?)~~/g, "<del>$1</del>");

  result = result.replace(/`([^`]+)`/g, "<code>$1</code>");

  return result;
}

function isHtmlLine(line: string): boolean {
  return /^\s*<(?:img|div|p|br|hr|details|summary|table|tr|td|th|thead|tbody|figcaption|figure|picture|source|video|iframe)\b/i.test(
    line.trim(),
  );
}

export function markdownToHtml(md: string): string {
  const lines = md.split("\n");
  const html: string[] = [];
  let i = 0;

  const pushParagraph = (text: string) => {
    if (text.trim()) {
      html.push(`<p>${inlineMarkdown(text.trim())}</p>`);
    }
  };

  while (i < lines.length) {
    const line = lines[i];

    if (line.startsWith("```")) {
      const lang = line.slice(3).trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) {
        codeLines.push(escapeHtml(lines[i]));
        i++;
      }
      i++;
      const langAttr = lang ? ` class="language-${escapeHtml(lang)}"` : "";
      html.push(`<pre><code${langAttr}>${codeLines.join("\n")}</code></pre>`);
      continue;
    }

    if (isHtmlLine(line)) {
      const htmlBlock: string[] = [];
      while (
        i < lines.length &&
        (isHtmlLine(lines[i]) || (htmlBlock.length > 0 && lines[i].trim() !== "" && !lines[i].startsWith("#")))
      ) {
        htmlBlock.push(lines[i]);
        i++;
        if (htmlBlock.length > 1 && /^\s*<\//.test(lines[i - 1])) break;
      }
      html.push(htmlBlock.join("\n"));
      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      html.push(`<h${level}>${inlineMarkdown(headingMatch[2])}</h${level}>`);
      i++;
      continue;
    }

    if (/^(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      html.push("<hr />");
      i++;
      continue;
    }

    if (line.trim().startsWith("|") && line.trim().endsWith("|")) {
      const tableRows: string[][] = [];
      let hasHeader = false;

      while (i < lines.length && lines[i].trim().startsWith("|") && lines[i].trim().endsWith("|")) {
        const row = lines[i].trim();

        if (/^\|[\s:]*-{2,}[\s:]*(\|[\s:]*-{2,}[\s:]*)*\|$/.test(row)) {
          hasHeader = true;
          i++;
          continue;
        }

        const cells = row
          .slice(1, -1)
          .split("|")
          .map((c) => c.trim());
        tableRows.push(cells);
        i++;
      }

      if (tableRows.length > 0) {
        html.push("<table>");
        tableRows.forEach((cells, rowIdx) => {
          const isHeaderRow = hasHeader && rowIdx === 0;
          const tag = isHeaderRow ? "th" : "td";
          const rowHtml = cells.map((c) => `<${tag}>${inlineMarkdown(c)}</${tag}>`).join("");
          if (isHeaderRow) {
            html.push(`<thead><tr>${rowHtml}</tr></thead><tbody>`);
          } else {
            html.push(`<tr>${rowHtml}</tr>`);
          }
        });
        if (hasHeader) {
          html.push("</tbody>");
        }
        html.push("</table>");
      }
      continue;
    }

    if (line.startsWith("> ") || line === ">") {
      const quoteLines: string[] = [];
      while (i < lines.length && (lines[i].startsWith("> ") || lines[i] === ">")) {
        quoteLines.push(lines[i].replace(/^>\s?/, ""));
        i++;
      }
      html.push(`<blockquote>${markdownToHtml(quoteLines.join("\n"))}</blockquote>`);
      continue;
    }

    if (/^(\s*)([-*+])\s+/.test(line)) {
      const listItems: string[] = [];
      while (i < lines.length && /^(\s*)([-*+])\s+/.test(lines[i])) {
        const match = lines[i].match(/^(\s*)([-*+])\s+(.*)$/);
        if (match) {
          listItems.push(inlineMarkdown(match[3]));
        }
        i++;
      }
      html.push("<ul>");
      listItems.forEach((item) => html.push(`<li>${item}</li>`));
      html.push("</ul>");
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const listItems: string[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        const match = lines[i].match(/^\s*\d+\.\s+(.*)$/);
        if (match) {
          listItems.push(inlineMarkdown(match[1]));
        }
        i++;
      }
      html.push("<ol>");
      listItems.forEach((item) => html.push(`<li>${item}</li>`));
      html.push("</ol>");
      continue;
    }

    if (line.trim() === "") {
      i++;
      continue;
    }

    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !lines[i].startsWith("#") &&
      !lines[i].startsWith("```") &&
      !lines[i].startsWith("> ") &&
      !lines[i].startsWith("| ") &&
      !/^(\s*)([-*+])\s+/.test(lines[i]) &&
      !/^\s*\d+\.\s+/.test(lines[i]) &&
      !/^(-{3,}|\*{3,}|_{3,})\s*$/.test(lines[i]) &&
      !isHtmlLine(lines[i])
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    if (paraLines.length > 0) {
      pushParagraph(paraLines.join(" "));
    }
  }

  return html.join("\n");
}
