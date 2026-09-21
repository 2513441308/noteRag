import { useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { ChatWindow } from "./components/ChatWindow";
import { Composer } from "./components/Composer";
import { SettingsModal } from "./components/SettingsModal";
import { KnowledgeModal } from "./components/KnowledgeModal";
import { EvalModal } from "./components/EvalModal";
import { useChat } from "./hooks/useChat";

export default function App() {
  const {
    threads,
    currentId,
    messages,
    streaming,
    loading,
    error,
    newThread,
    removeThread,
    renameThread,
    selectThread,
    send,
  } = useChat();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [knowledgeOpen, setKnowledgeOpen] = useState(false);
  const [evalOpen, setEvalOpen] = useState(false);

  return (
    <div className="relative z-10 flex h-screen overflow-hidden">
      <Sidebar
        threads={threads}
        currentId={currentId}
        disabled={streaming}
        onNew={newThread}
        onSelect={selectThread}
        onDelete={removeThread}
        onRename={renameThread}
        onOpenSettings={() => setSettingsOpen(true)}
        onOpenKnowledge={() => setKnowledgeOpen(true)}
        onOpenEval={() => setEvalOpen(true)}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        {error && (
          <div className="mx-auto mt-3 w-fit max-w-[90%] rotate-[-0.8deg] rounded-sm bg-red-100 px-4 py-1.5 text-sm font-semibold text-red-800 shadow-note-sm ring-1 ring-red-300">
            ⚠️ {error}
          </div>
        )}

        {loading ? (
          <div className="flex flex-1 items-center justify-center">
            <div
              className="note note-yellow relative px-8 py-5 font-hand text-2xl text-ink"
              style={{ transform: "rotate(1deg)" }}
            >
              <span className="tape" aria-hidden />
              正在把便签板挂上墙…
            </div>
          </div>
        ) : (
          <>
            <ChatWindow messages={messages} streaming={streaming} />
            <Composer onSend={send} disabled={streaming} />
          </>
        )}
      </main>

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
      <KnowledgeModal
        open={knowledgeOpen}
        onClose={() => setKnowledgeOpen(false)}
      />
      <EvalModal open={evalOpen} onClose={() => setEvalOpen(false)} />
    </div>
  );
}
