import { useEffect, type ReactNode } from "react";

interface ModalProps {
  open: boolean;
  onClose: () => void;
  /** 无障碍标签（aria-label） */
  label: string;
  /** 便签标题 */
  title: ReactNode;
  /** 标题下的副标题说明 */
  subtitle?: ReactNode;
  /** 外壳宽度类，默认 640px；知识库等宽场景传 w-[780px] */
  widthClass?: string;
  /** 外壳最大高度类 */
  maxHeightClass?: string;
  /** 弹窗体内容（原两个 Modal 的 body / footer 区域） */
  children: ReactNode;
}

/**
 * 便签风格弹窗外壳：半透明遮罩 + 纸条容器 + 胶带 + 标题栏 + 关闭按钮。
 * 点击遮罩或按 Esc 关闭；内容区由调用方自行组织。
 */
export function Modal({
  open,
  onClose,
  label,
  title,
  subtitle,
  widthClass = "w-[640px]",
  maxHeightClass = "max-h-[86vh]",
  children,
}: ModalProps) {
  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-amber-950/50 p-4 backdrop-blur-[2px]"
      onClick={onClose}
    >
      <div
        className={`note note-yellow relative flex ${maxHeightClass} ${widthClass} max-w-full flex-col !p-0 shadow-note-lg`}
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label={label}
      >
        <span className="tape" aria-hidden />

        <div className="flex items-start justify-between px-7 pb-2 pt-6">
          <div>
            <h2 className="font-hand text-3xl text-ink">{title}</h2>
            {subtitle && (
              <p className="mt-0.5 text-xs text-amber-900/70">{subtitle}</p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭"
            className="flex h-7 w-7 items-center justify-center rounded-full bg-red-700/80 text-sm font-bold text-amber-50 shadow transition hover:scale-110 hover:bg-red-800"
          >
            ×
          </button>
        </div>

        {children}
      </div>
    </div>
  );
}
