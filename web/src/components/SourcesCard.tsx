interface Props {
  sources: string[];
}

function basename(path: string): string {
  return path.split("/").pop() || path;
}

/** 引用来源：便签底部一排图钉小标签 */
export function SourcesCard({ sources }: Props) {
  return (
    <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-dashed border-amber-800/25 pt-2.5">
      <span className="text-xs font-bold text-ink-soft">📎 引用便签：</span>
      {sources.map((src, i) => (
        <span
          key={`${src}-${i}`}
          title={src}
          className="inline-flex items-center gap-1 rounded-sm bg-white/55 px-2 py-0.5 text-xs text-amber-900 shadow-sm ring-1 ring-amber-900/10"
        >
          <span className="font-bold">[{i + 1}]</span>
          {basename(src)}
        </span>
      ))}
    </div>
  );
}
