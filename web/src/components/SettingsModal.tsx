import { useEffect, useMemo, useState } from "react";
import {
  getConfig,
  saveConfig,
  type ConfigField,
} from "../api/admin";
import { Modal } from "./Modal";

interface Props {
  open: boolean;
  onClose: () => void;
}

const GROUP_ORDER = ["对话模型", "检索与重排", "嵌入模型", "历史压缩"];
const GROUP_EMOJI: Record<string, string> = {
  对话模型: "🤖",
  检索与重排: "📎",
  嵌入模型: "🧬",
  历史压缩: "🗜️",
};

export function SettingsModal({ open, onClose }: Props) {
  const [fields, setFields] = useState<ConfigField[]>([]);
  /** key → 表单字符串；secret 未触碰时为空串（保存时不下发） */
  const [draft, setDraft] = useState<Record<string, string>>({});
  const [touchedSecrets, setTouchedSecrets] = useState<Set<string>>(
    () => new Set(),
  );
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedTip, setSavedTip] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    setError(null);
    setSavedTip(null);
    getConfig()
      .then((fs) => {
        setFields(fs);
        const init: Record<string, string> = {};
        for (const f of fs) {
          init[f.key] =
            f.type === "secret" ? "" : f.value == null ? "" : String(f.value);
        }
        setDraft(init);
        setTouchedSecrets(new Set());
      })
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  }, [open]);

  const groups = useMemo(() => {
    const map = new Map<string, ConfigField[]>();
    for (const f of fields) {
      const list = map.get(f.group) ?? [];
      list.push(f);
      map.set(f.group, list);
    }
    return GROUP_ORDER.filter((g) => map.has(g)).map((g) => ({
      name: g,
      fields: map.get(g)!,
    }));
  }, [fields]);

  if (!open) return null;

  const setValue = (key: string, value: string) =>
    setDraft((prev) => ({ ...prev, [key]: value }));

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSavedTip(null);
    const values: Record<string, string> = {};
    for (const f of fields) {
      if (f.type === "secret") {
        // 密钥：留空 = 不修改；只有用户输入了新值才下发
        if (touchedSecrets.has(f.key) && draft[f.key] !== "") {
          values[f.key] = draft[f.key];
        }
        continue;
      }
      values[f.key] = draft[f.key] ?? "";
    }
    try {
      const res = await saveConfig(values);
      setSavedTip(
        `已生效：${res.applied.length} 项配置，重置 ${res.caches_cleared} 个组件缓存，已写入 .env`,
      );
      setTouchedSecrets(new Set());
      // 刷新一遍（拿回脱敏后的最新值与类型）
      const fs = await getConfig();
      setFields(fs);
      setDraft((prev) => {
        const next = { ...prev };
        for (const f of fs) if (f.type === "secret") next[f.key] = "";
        return next;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      label="调试设置"
      title="调试设置"
      subtitle="改完点「贴上新设置」即时生效，同时写回 .env；下一次提问即用新配置"
    >
      <div className="flex-1 overflow-y-auto px-7 py-3">
          {loading && (
            <p className="py-10 text-center text-sm text-amber-900/70">
              正在翻找当前设置…
            </p>
          )}
          {error && (
            <div className="mb-3 rounded-sm border-2 border-dashed border-red-700/60 bg-red-100/70 px-3 py-2 text-sm text-red-900">
              {error}
            </div>
          )}
          {savedTip && !error && (
            <div className="mb-3 rounded-sm border-2 border-dashed border-emerald-700/60 bg-emerald-100/70 px-3 py-2 text-sm text-emerald-900">
              ✅ {savedTip}
            </div>
          )}

          {groups.map((group) => (
            <section key={group.name} className="mb-5">
              <h3 className="mb-2 border-b-2 border-dashed border-amber-900/30 pb-1 font-hand text-xl text-ink">
                {GROUP_EMOJI[group.name]} {group.name}
              </h3>
              <div className="grid grid-cols-1 gap-x-4 gap-y-2.5 sm:grid-cols-2">
                {group.fields.map((f) => (
                  <label
                    key={f.key}
                    className={`block text-xs ${f.type === "bool" ? "flex items-center gap-2 pt-4" : ""}`}
                  >
                    {f.type === "bool" ? (
                      <>
                        <input
                          type="checkbox"
                          checked={draft[f.key] === "true"}
                          onChange={(e) =>
                            setValue(f.key, e.target.checked ? "true" : "false")
                          }
                          className="h-4 w-4 accent-amber-700"
                        />
                        <span className="text-sm font-semibold text-ink">
                          {f.label}
                        </span>
                      </>
                    ) : (
                      <>
                        <span className="mb-0.5 block font-semibold text-ink">
                          {f.label}
                          {f.optional && (
                            <span className="ml-1 font-normal text-amber-900/50">
                              （可空）
                            </span>
                          )}
                        </span>
                        {f.type === "select" ? (
                          <select
                            value={draft[f.key] ?? ""}
                            onChange={(e) => setValue(f.key, e.target.value)}
                            className="w-full rounded-sm border-b-2 border-amber-900/40 bg-white/80 px-2 py-1 text-sm text-ink outline-none focus:border-amber-700"
                          >
                            {f.options?.map((opt) => (
                              <option key={opt} value={opt}>
                                {opt}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type={
                              f.type === "secret"
                                ? "password"
                                : f.type === "int" || f.type === "float"
                                  ? "number"
                                  : "text"
                            }
                            step={f.type === "float" ? "0.1" : undefined}
                            value={draft[f.key] ?? ""}
                            placeholder={
                              f.type === "secret"
                                ? f.secret_set
                                  ? `已设置 ${f.value ?? ""}（留空不修改）`
                                  : "尚未设置"
                                : undefined
                            }
                            onChange={(e) => {
                              setValue(f.key, e.target.value);
                              if (f.type === "secret") {
                                setTouchedSecrets(
                                  (prev) => new Set(prev).add(f.key),
                                );
                              }
                            }}
                            className="w-full rounded-sm border-b-2 border-amber-900/40 bg-white/80 px-2 py-1 text-sm text-ink outline-none focus:border-amber-700"
                          />
                        )}
                        {f.help && (
                          <span className="mt-0.5 block text-[10px] leading-4 text-amber-900/60">
                            {f.help}
                          </span>
                        )}
                      </>
                    )}
                  </label>
                ))}
              </div>
            </section>
          ))}
        </div>

        <div className="flex items-center justify-end gap-3 border-t-2 border-dashed border-amber-900/30 px-7 py-4">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-sm font-bold text-amber-900/70 transition hover:text-ink"
          >
            取消
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={saving || loading}
            className="note note-mint relative px-5 py-2 text-sm font-bold shadow-note-sm transition hover:-translate-y-0.5 disabled:opacity-50"
          >
            {saving ? "粘贴中…" : "📌 贴上新设置"}
          </button>
        </div>
    </Modal>
  );
}
