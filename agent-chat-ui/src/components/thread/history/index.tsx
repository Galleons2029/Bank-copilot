import { Button } from "@/components/ui/button";
import { useThreads } from "@/providers/Thread";
import { Thread } from "@langchain/langgraph-sdk";
import { useEffect } from "react";

import { getContentString } from "../utils";
import { useQueryState, parseAsBoolean } from "nuqs";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { PanelRightOpen, PanelRightClose } from "lucide-react";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { cn } from "@/lib/utils";

function ThreadList({
  threads,
  onThreadClick,
}: {
  threads: Thread[];
  onThreadClick?: (threadId: string) => void;
}) {
  const [threadId, setThreadId] = useQueryState("threadId");

  return (
    <div className="flex h-full min-h-0 w-full flex-col items-start justify-start gap-2 overflow-y-auto pr-1 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {threads.map((t) => {
        let itemText = t.thread_id;
        if (
          typeof t.values === "object" &&
          t.values &&
          "messages" in t.values &&
          Array.isArray(t.values.messages) &&
          t.values.messages?.length > 0
        ) {
          const firstMessage = t.values.messages[0];
          itemText = getContentString(firstMessage.content);
        }
        return (
          <div
            key={t.thread_id}
            className="w-full px-1.5"
          >
            <Button
              variant="ghost"
              className={cn(
                "w-full items-start justify-start rounded-2xl text-left font-normal shadow-sm transition",
                threadId === t.thread_id
                  ? "bg-emerald-500 text-white shadow-lg shadow-emerald-900/20 hover:bg-emerald-500"
                  : "bg-white/60 text-emerald-950 hover:bg-white",
              )}
              onClick={(e) => {
                e.preventDefault();
                onThreadClick?.(t.thread_id);
                if (t.thread_id === threadId) return;
                setThreadId(t.thread_id);
              }}
            >
              <p className="truncate text-ellipsis">{itemText}</p>
            </Button>
          </div>
        );
      })}
    </div>
  );
}

function ThreadHistoryLoading() {
  return (
    <div className="flex h-full min-h-0 w-full flex-col items-start justify-start gap-2 overflow-y-auto pr-1 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {Array.from({ length: 30 }).map((_, i) => (
        <Skeleton
          key={`skeleton-${i}`}
          className="h-10 w-full rounded-2xl"
        />
      ))}
    </div>
  );
}

export default function ThreadHistory() {
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );

  const { getThreads, threads, setThreads, threadsLoading, setThreadsLoading } =
    useThreads();

  useEffect(() => {
    if (typeof window === "undefined") return;
    setThreadsLoading(true);
    getThreads()
      .then(setThreads)
      .catch(console.error)
      .finally(() => setThreadsLoading(false));
  }, []);

  return (
    <>
      <aside className="shadow-inner-right relative hidden h-full min-h-0 w-[320px] shrink-0 flex flex-col overflow-hidden rounded-tr-3xl bg-gradient-to-b from-white via-emerald-50/20 to-white/95 pb-6 pt-0 text-emerald-950 backdrop-blur-xl lg:flex">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 rounded-tr-3xl border border-white/60"
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 rounded-tr-3xl ring-1 ring-inset ring-emerald-100/60"
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-5 -top-6 h-6 rounded-t-3xl bg-white/90 shadow-[0_25px_60px_rgba(16,185,129,0.25)]"
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-y-4 right-0 w-[2px] rounded-full bg-gradient-to-b from-transparent via-emerald-300/80 to-transparent blur-[0.5px]"
        />
        <div
          aria-hidden
          className="pointer-events-none absolute inset-y-10 -right-6 w-16 bg-gradient-to-b from-emerald-500/15 via-emerald-400/5 to-emerald-500/15 blur-3xl"
        />

        <div className="relative z-10 flex h-full flex-col">
          <div className="flex w-full items-center justify-between border-b border-white/40 px-5 pb-3 pt-3">
            <h1 className="text-lg font-semibold tracking-tight">
              对话历史
            </h1>
            <Button
              className="hover:bg-gray-100"
              variant="ghost"
              onClick={() => setChatHistoryOpen((p) => !p)}
            >
              {chatHistoryOpen ? (
                <PanelRightOpen className="size-5" />
              ) : (
                <PanelRightClose className="size-5" />
              )}
            </Button>
          </div>

          <div className="flex-1 min-h-0 px-5 pt-4">
            {threadsLoading ? (
              <ThreadHistoryLoading />
            ) : (
              <ThreadList threads={threads} />
            )}
          </div>
        </div>
      </aside>
      <div className="lg:hidden">
        <Sheet
          open={!!chatHistoryOpen && !isLargeScreen}
          onOpenChange={(open) => {
            if (isLargeScreen) return;
            setChatHistoryOpen(open);
          }}
        >
          <SheetContent
            side="left"
            className="flex lg:hidden"
          >
            <SheetHeader>
              <SheetTitle>对话历史</SheetTitle>
            </SheetHeader>
            <ThreadList
              threads={threads}
              onThreadClick={() => setChatHistoryOpen((o) => !o)}
            />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
