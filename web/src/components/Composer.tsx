import { useRef, useState, type KeyboardEvent } from "react";

interface Props {
  onSend: (question: string) => void;
  disabled: boolean;
}

export function Composer({ onSend, disabled }: Props) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
    requestAnimationFrame(() => {
      if (textareaRef.current) textareaRef.current.style.height = "auto";
    });
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="px-4 pb-6 pt-2 sm:px-8">
      <div className="mx-auto max-w-3xl">
        <div className="note note-yellow relative px-5 pb-3 pt-4">
          <span className="tape" aria-hidden />
          <div className="lined-paper mt-1 rounded-sm">
            <textarea
              ref={textareaRef}
              value={value}
              disabled={disabled}
              onChange={(e) => {
                setValue(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
              }}
              onKeyDown={handleKeyDown}
              rows={2}
              placeholder="把问题写在这张便签上…（Enter 贴上，Shift+Enter 换行）"
              className="w-full resize-none bg-transparent px-3 py-0.5 text-[15px] text-ink outline-none placeholder:text-ink-soft/60 disabled:opacity-50"
              style={{ lineHeight: "34px" }}
            />
          </div>
          <div className="mt-2 flex items-center justify-between">
            <span className="text-xs text-ink-soft">
              {disabled ? "正在书写回答…" : "Enter 发送 · Shift+Enter 换行"}
            </span>
            <button
              type="button"
              onClick={submit}
              disabled={disabled || !value.trim()}
              className="group flex items-center gap-1.5 rounded-full bg-red-600 px-4 py-1.5 text-sm font-bold text-amber-50 shadow-pin transition hover:scale-105 hover:bg-red-500 active:scale-95 disabled:cursor-not-allowed disabled:bg-stone-400 disabled:shadow-none"
              title="贴到问答板上"
            >
              <span className="text-base leading-none">📌</span>
              贴上
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
