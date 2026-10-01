import { Fragment, type ReactNode } from "react";

/** A deliberately small Markdown subset for model replies: headings,
 * paragraphs, bullet/numbered lists, pipe tables, fenced code, and inline
 * **bold**, *italic*, `code`. Builds React elements only — never
 * innerHTML — so model output can't inject markup. */

const INLINE = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g;

function renderInline(text: string): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return <code key={i}>{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      return <em key={i}>{part.slice(1, -1)}</em>;
    }
    return <Fragment key={i}>{part}</Fragment>;
  });
}

const BULLET = /^\s*[-*•]\s+/;
const NUMBERED = /^\s*\d+[.)]\s+/;
const HEADING = /^(#{1,4})\s+(.*)$/;
const TABLE_ROW = /^\s*\|.*\|\s*$/;
const TABLE_DIVIDER = /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/;

const splitRow = (line: string) =>
  line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());

export function Markdown({ text }: { text: string }) {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (line.trim() === "") {
      i++;
      continue;
    }

    if (line.trim().startsWith("```")) {
      const code: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        code.push(lines[i++]);
      }
      i++;
      blocks.push(<pre key={blocks.length}><code>{code.join("\n")}</code></pre>);
      continue;
    }

    const heading = HEADING.exec(line);
    if (heading) {
      const Tag = (["h3", "h3", "h4", "h5"] as const)[heading[1].length - 1];
      blocks.push(<Tag key={blocks.length}>{renderInline(heading[2])}</Tag>);
      i++;
      continue;
    }

    if (TABLE_ROW.test(line) && i + 1 < lines.length && TABLE_DIVIDER.test(lines[i + 1])) {
      const header = splitRow(line);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && TABLE_ROW.test(lines[i])) {
        rows.push(splitRow(lines[i++]));
      }
      blocks.push(
        <div className="md-table" key={blocks.length}>
          <table>
            <thead>
              <tr>{header.map((cell, c) => <th key={c}>{renderInline(cell)}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((row, r) => (
                <tr key={r}>{row.map((cell, c) => <td key={c}>{renderInline(cell)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }

    const listPattern = BULLET.test(line) ? BULLET : NUMBERED.test(line) ? NUMBERED : null;
    if (listPattern) {
      const items: string[] = [];
      while (i < lines.length && listPattern.test(lines[i])) {
        items.push(lines[i++].replace(listPattern, ""));
      }
      const List = listPattern === BULLET ? "ul" : "ol";
      blocks.push(
        <List key={blocks.length}>
          {items.map((item, n) => <li key={n}>{renderInline(item)}</li>)}
        </List>,
      );
      continue;
    }

    const paragraph: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !HEADING.test(lines[i]) &&
      !BULLET.test(lines[i]) &&
      !NUMBERED.test(lines[i]) &&
      !lines[i].trim().startsWith("```") &&
      !(TABLE_ROW.test(lines[i]) && i + 1 < lines.length && TABLE_DIVIDER.test(lines[i + 1]))
    ) {
      paragraph.push(lines[i++]);
    }
    blocks.push(
      <p key={blocks.length}>
        {paragraph.map((l, n) => (
          <Fragment key={n}>
            {n > 0 && <br />}
            {renderInline(l)}
          </Fragment>
        ))}
      </p>,
    );
  }

  return <>{blocks}</>;
}
