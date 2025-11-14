'use client';
import { AccountPanel } from '@/app/account/account-panel';
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  type LucideIcon,
  Home,
  Bot,
  Database,
  FileText,
  Settings,
  Bell,
  User,
  Wallet,
  TrendingUp,
  Users,
  ClipboardList,
  LogOut,
  Edit3,
  Plus,
  RefreshCw,
  Trash2,
  Layers,
  GraduationCap,
  CheckCircle2,
  Clock3,
  BookOpenCheck,
  Sparkles,
  ArrowUpRight,
  Target,
  CalendarClock,
  MessageCircle,
} from 'lucide-react';
import { Thread } from "@/components/thread";
import { StreamProvider } from "@/providers/Stream";
import { ThreadProvider } from "@/providers/Thread";
import { ArtifactProvider } from "@/components/thread/artifact";
import { Toaster } from "@/components/ui/sonner";
import React from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import type { KnowledgeBase, KnowledgeChunk } from "@/types/knowledge";
import { Skeleton } from "@/components/ui/skeleton";

type NavigationItem = {
  id: string;
  label: string;
  icon: LucideIcon;
  color: string;
};

type StatDefinition = {
  id: string;
  label: string;
  value: string;
  icon: LucideIcon;
  accent: {
    bg: string;
    icon: string;
  };
};

type FeaturePanel = {
  title: string;
  description?: string;
  render: (context?: FeaturePanelRenderContext) => ReactNode;
};

type FeaturePlaceholderAction = {
  label: string;
  variant?: 'primary' | 'secondary';
};

type AIGuidanceIntent = {
  topic: string;
  prompt: string;
};

type FeaturePanelRenderContext = {
  openPanel: (id: PanelId) => void;
  startAIGuidance: (intent: AIGuidanceIntent) => void;
  aiGuidanceIntent: AIGuidanceIntent | null;
  clearAIGuidance: () => void;
};

const navigationItems = [
  { id: 'ai', label: 'AI 助手', icon: Bot, color: 'bg-emerald-500' },
  { id: 'dashboard', label: '数据看板', icon: Home, color: 'bg-emerald-500' },
  { id: 'knowledge', label: '知识库', icon: Database, color: 'bg-emerald-500' },
  { id: 'accounts', label: '总分查账', icon: FileText, color: 'bg-emerald-500' },
  { id: 'learning', label: '学习进度', icon: BookOpenCheck, color: 'bg-indigo-500' },
  { id: 'training', label: '企业培训', icon: GraduationCap, color: 'bg-amber-500' },
] as const satisfies NavigationItem[];

type PanelId = (typeof navigationItems)[number]['id'];

const dashboardStats: StatDefinition[] = [
  {
    id: 'balance',
    label: '总账户余额',
    value: '¥7,049,371.25',
    icon: Wallet,
    accent: { bg: 'bg-green-100', icon: 'text-green-600' },
  },
  {
    id: 'income',
    label: '本月收入',
    value: '¥45,678.90',
    icon: TrendingUp,
    accent: { bg: 'bg-blue-100', icon: 'text-blue-600' },
  },
  {
    id: 'activeAccounts',
    label: '活跃账户',
    value: '12',
    icon: Users,
    accent: { bg: 'bg-yellow-100', icon: 'text-yellow-600' },
  },
  {
    id: 'pending',
    label: '待处理事务',
    value: '8',
    icon: ClipboardList,
    accent: { bg: 'bg-red-100', icon: 'text-red-600' },
  },
];

const incomeExpenseData = [
  { month: 'Jan', income: 3800, expense: 2500 },
  { month: 'Feb', income: 2800, expense: 1500 },
  { month: 'Mar', income: 2000, expense: 10000 },
  { month: 'Apr', income: 2500, expense: 3500 },
  { month: 'May', income: 2000, expense: 4800 },
  { month: 'Jun', income: 2500, expense: 3200 },
  { month: 'Jul', income: 3200, expense: 4200 },
  { month: 'Aug', income: 4000, expense: 2500 },
  { month: 'Sep', income: 5200, expense: 3000 },
  { month: 'Oct', income: 4800, expense: 4000 },
  { month: 'Nov', income: 5500, expense: 2800 },
  { month: 'Dec', income: 6000, expense: 3500 },
];

const accountDistributionData = [
  { name: '企业基本账户', value: 400 },
  { name: '企业储蓄账户', value: 300 },
  { name: '外汇结算账户', value: 200 },
  { name: '专项基金账户', value: 100 },
];

const ACCOUNT_COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

const realtimeInsightMetrics = [
  { label: '新员工培训完成率', value: '87%', helper: '较上周 +4.2%' },
  { label: '财务异常待处理', value: '3 条', helper: '已派单 2 条' },
  { label: '知识库日均访问', value: '256 次', helper: '环比 +12.4%' },
] as const;

const trainingOverviewStats = [
  { label: '平均完成率', value: '86%', helper: '环比提升 6.5%' },
  { label: '重点课程合格率', value: '94%', helper: '本周新增 2 位通过' },
  { label: '未完成知识点', value: '37 个', helper: '集中在风控法规模块' },
] as const;

const trainingKnowledgeTracks = [
  {
    id: 'risk',
    title: '风控与合规',
    completed: 26,
    total: 30,
    owner: '风控部',
    highlight: '新版监管指引',
  },
  {
    id: 'cash',
    title: '现金管理·企业网银',
    completed: 18,
    total: 24,
    owner: '对公条线',
    highlight: '费用精细化核算',
  },
  {
    id: 'ai',
    title: 'AI+业务操作规范',
    completed: 12,
    total: 20,
    owner: '运营管理部',
    highlight: '流程机器人联动',
  },
] as const;

const trainingRecords = [
  {
    id: 'li-jing',
    name: '李婧',
    role: '对公客户经理',
    focus: '跨境业务培训',
    progress: 92,
    lastUpdate: '今日 09:15',
    status: '领先 4 天',
    completedTopics: ['跨境收支合规指引', '贸易背景核验清单', '涉外收入备案流程'],
    pendingTopics: ['外汇风险预警响应'],
  },
  {
    id: 'wang-hao',
    name: '王浩',
    role: '财务共享中心',
    focus: '账务自动化专项',
    progress: 74,
    lastUpdate: '昨日 18:20',
    status: '需督促',
    completedTopics: ['账簿影像归档', '自动分录回溯', '业务口径与会计口径对账'],
    pendingTopics: ['跨系统异常联调', '审批串联策略'],
  },
  {
    id: 'zhang-yu',
    name: '张瑜',
    role: '运营支持',
    focus: '网点作业风险控制',
    progress: 61,
    lastUpdate: '本周二 11:40',
    status: '按计划推进',
    completedTopics: ['双录流程质检', '票据影像留存', '异常凭证复核'],
    pendingTopics: ['柜面智能校验', '压力场景演练'],
  },
] as const;

const personalLearningProfile = {
  targetRole: '数字风控分析师',
  cycle: '第 3 周 / 共 6 周',
  progress: 68,
  hours: '26.5 小时',
  mastered: 42,
  total: 58,
  nextCheckpoint: '5 月 28 日 · 案例实战',
  mentor: '沈敏 · 风控专家',
};

const personalLearningHighlights = [
  { label: '本周累计学习', value: '12.5 小时', helper: '含 3 次实操演练', trend: '+2.3 小时' },
  { label: '知识点掌握', value: '24 / 31', helper: '9 个待巩固', trend: '+3' },
  { label: '实战题正确率', value: '88%', helper: '近三天 20 道题', trend: '+5%' },
] as const;

const personalKnowledgePoints = [
  {
    id: 'graph-risk',
    title: '图数据库风控关系解析',
    mastery: 72,
    status: '待巩固',
    trend: '+8%',
    tags: ['风控建模', '图谱'],
    lastReview: '昨日 21:10',
    aiPrompt:
      '请扮演资深风控教练，帮我梳理图数据库风控关系解析的核心方法，重点讲解如何识别隐含关联账户。',
  },
  {
    id: 'cash-flow',
    title: '现金流异常链路诊断',
    mastery: 54,
    status: '需重点复习',
    trend: '-3%',
    tags: ['现金管理', '预警'],
    lastReview: '3 天前',
    aiPrompt:
      '我需要重新理解现金流异常链路诊断，请拆解排查步骤，并给出 2 个场景演练题。',
  },
  {
    id: 'ai-governance',
    title: 'AI 助手在审批链路的治理策略',
    mastery: 81,
    status: '保持强化',
    trend: '+11%',
    tags: ['AI 治理', '审批'],
    lastReview: '今日 09:35',
    aiPrompt:
      '请结合 AI 助手在审批链路的治理策略，帮我列出需要重点监控的 3 个风险点及应对方案。',
  },
] as const;

const personalPlanTimeline = [
  {
    id: 'today',
    title: '今日冲刺 · 120 分钟',
    focus: '交易异常特征 + AI 辅助分析',
    tasks: ['复盘典型案例库', '完成 6 道演练题', '整理复盘笔记'],
    due: '今日 21:00',
    aiPrompt: '基于交易异常特征和 AI 辅助分析，帮我制定 2 组练习题并给出解析。',
  },
  {
    id: 'tomorrow',
    title: '明日预习 · 90 分钟',
    focus: '跨境收支合规 + 情景问答',
    tasks: ['阅读监管更新', '准备情景问答模版', '沉淀 FAQ'],
    due: '明日 19:30',
    aiPrompt: '我需要跨境收支合规的最新监管点，请生成情景问答脚本并附加追问建议。',
  },
] as const;

const featurePanels: Record<PanelId, FeaturePanel> = {
  dashboard: {
    title: '数据看板',
    description: '核心指标、收入支出趋势与账户分布总览',
    render: () => <DashboardOverview />,
  },
  ai: {
    title: 'AI 助手',
    description: '构建智能问答流程，辅助内部员工完成业务咨询',
    render: (context) => (
      <AIChatPanel guidanceIntent={context?.aiGuidanceIntent} onClearGuidance={context?.clearAIGuidance} />
    ),
  },
  knowledge: {
    title: '知识库',
    description: '沉淀文档、政策与规范，供其他模块调用',
    render: () => <KnowledgeBasePanel />,
  },
  accounts: {
    title: '总分查账',
    description: '统一的账户视图，支持按条件筛选与导出',
    render: () => <AccountPanel variant="dashboard"/>,
  },
  learning: {
    title: '个人学习进度',
    description: '记录个人学习轨迹、知识点掌握情况，并联动 AI 助手生成计划。',
    render: (context) => (
      <PersonalLearningPanel
        onGuide={(intent) => context?.startAIGuidance(intent)}
      />
    ),
  },
  training: {
    title: '企业培训',
    description: '记录员工对企业培训知识点的学习进度，区分已完成与未完成重点。',
    render: () => <TrainingPanel />,
  },
};

type AIChatPanelProps = {
  guidanceIntent?: AIGuidanceIntent | null;
  onClearGuidance?: () => void;
};

function AIChatPanel({ guidanceIntent, onClearGuidance }: AIChatPanelProps) {
  const [copied, setCopied] = useState(false);
  const topic = guidanceIntent?.topic ?? '';
  const prompt = guidanceIntent?.prompt ?? '';

  useEffect(() => {
    setCopied(false);
  }, [prompt]);

  const handleCopyPrompt = useCallback(async () => {
    if (!prompt) {
      return;
    }
    if (typeof navigator === 'undefined' || !navigator.clipboard) {
      toast.error('当前环境不支持复制');
      return;
    }
    try {
      await navigator.clipboard.writeText(prompt);
      setCopied(true);
      toast.success('已复制到剪贴板');
    } catch (error) {
      const message = error instanceof Error ? error.message : '复制失败';
      toast.error(message);
    }
  }, [prompt]);

  return (
    <section className="relative flex h-full min-h-0 flex-col bg-gradient-to-b from-white via-emerald-50/40 to-emerald-50/5">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-10 top-0 h-8 -translate-y-6 rounded-t-3xl bg-white shadow-[0_30px_65px_-40px_rgba(5,150,105,0.75)]"
      />

      <React.Suspense
        fallback={
          <div className="flex h-full items-center justify-center bg-white/60">
            <div className="text-center">
              <Bot className="mx-auto mb-4 h-12 w-12 animate-pulse text-emerald-500" />
              <p className="text-gray-500">AI 助手正在加载...</p>
            </div>
          </div>
        }
      >
        <Toaster />
        <ThreadProvider>
          <StreamProvider>
            <ArtifactProvider>
              <div className="flex flex-1 min-h-0 px-6 pb-6 pt-4">
                <div
                  className={cn(
                    'relative flex h-full min-h-0 w-full overflow-hidden rounded-3xl border border-emerald-100/70 bg-white shadow-[0_25px_90px_-45px_rgba(4,120,87,0.8)]',
                    guidanceIntent ? 'pt-28' : '',
                  )}
                >
                  {guidanceIntent ? (
                    <div className="absolute left-6 right-6 top-6 z-20">
                      <div className="rounded-2xl border border-emerald-100 bg-white/95 p-4 shadow-lg shadow-emerald-900/10">
                        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                          <div>
                            <p className="text-sm font-semibold text-emerald-900">AI 指导主题：{topic}</p>
                            <p className="mt-1 text-xs text-gray-500">建议提问：</p>
                            <p className="mt-1 text-sm text-gray-700">{prompt}</p>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            <Button variant="outline" size="sm" onClick={handleCopyPrompt}>
                              {copied ? '已复制' : '复制引导语'}
                            </Button>
                            <Button size="sm" onClick={onClearGuidance}>
                              完成指导
                            </Button>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : null}
                  <Thread className="!h-full min-h-0" />
                </div>
              </div>
            </ArtifactProvider>
          </StreamProvider>
        </ThreadProvider>
      </React.Suspense>
    </section>
  );
}

export default function DashboardPage() {
  const [activeMenu, setActiveMenu] = useState<PanelId>('ai');
  const [aiGuidanceIntent, setAIGuidanceIntent] = useState<AIGuidanceIntent | null>(null);
  const [isSettingsDialogOpen, setSettingsDialogOpen] = useState(false);
  const router = useRouter();

  const handleSelectPanel = useCallback(
    (id: PanelId) => {
      if (id !== 'ai') {
        setAIGuidanceIntent(null);
      }
      setActiveMenu(id);
    },
    [],
  );

  const handleStartGuidance = useCallback(
    (intent: AIGuidanceIntent) => {
      setAIGuidanceIntent(intent);
      handleSelectPanel('ai');
    },
    [handleSelectPanel],
  );

  const handleClearGuidance = useCallback(() => setAIGuidanceIntent(null), []);

  const handleOpenSettingsDialog = useCallback(() => setSettingsDialogOpen(true), []);

  const handleLogout = useCallback(() => {
    router.push('/login');
  }, [router]);

  const activePanel = featurePanels[activeMenu] ?? featurePanels.dashboard;
  const panelContext: FeaturePanelRenderContext = {
    openPanel: handleSelectPanel,
    startAIGuidance: handleStartGuidance,
    aiGuidanceIntent,
    clearAIGuidance: handleClearGuidance,
  };

  return (
    <>
      <div className="flex h-screen bg-gray-50">
        <Sidebar
          items={navigationItems}
          active={activeMenu}
          onSelect={handleSelectPanel}
          onOpenSettings={handleOpenSettingsDialog}
          onLogout={handleLogout}
        />

        <div className="flex-1 flex flex-col">
          <DashboardHeader title={activePanel.title} description={activePanel.description} />
          <main className={`flex-1 ${activeMenu === 'ai' ? 'p-0 overflow-hidden' : 'p-6 overflow-y-auto'}`}>
            {activePanel.render(panelContext)}
          </main>
        </div>
      </div>

      <SystemSettingsDialog open={isSettingsDialogOpen} onOpenChange={setSettingsDialogOpen} />
    </>
  );
}

type SidebarProps = {
  items: readonly NavigationItem[];
  active: PanelId;
  onSelect: (id: PanelId) => void;
  onOpenSettings: () => void;
  onLogout: () => void;
};

function Sidebar({ items, active, onSelect, onOpenSettings, onLogout }: SidebarProps) {
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const userMenuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!isUserMenuOpen) {
      return;
    }
    const handleClickOutside = (event: MouseEvent) => {
      if (!userMenuRef.current?.contains(event.target as Node)) {
        setIsUserMenuOpen(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsUserMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isUserMenuOpen]);

  type UserMenuItem = {
    id: string;
    label: string;
    description?: string;
    icon: LucideIcon;
    action: () => void;
    destructive?: boolean;
  };

  const userMenuItems: UserMenuItem[] = [
    {
      id: 'plan',
      label: '升级套餐',
      description: '解锁更高的调用配额',
      icon: Sparkles,
      action: () => toast('升级套餐功能即将上线'),
    },
    {
      id: 'personalize',
      label: '个性化偏好',
      description: '定制主题与快捷方式',
      icon: Edit3,
      action: () => toast('个性化能力正在接入'),
    },
    {
      id: 'settings',
      label: '系统设置',
      description: '组织、权限与安全策略',
      icon: Settings,
      action: onOpenSettings,
    },
    {
      id: 'logout',
      label: '退出登录',
      icon: LogOut,
      destructive: true,
      action: onLogout,
    },
  ];

  return (
    <div className="relative flex w-64 flex-col bg-gradient-to-b from-emerald-600 via-emerald-700 to-emerald-800 text-emerald-50 shadow-2xl">
      <div className="p-6 border-b border-white/10">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-white rounded-lg flex items-center justify-center shadow-lg shadow-emerald-900/30">
            <img src="/logo.svg" alt="Logo" className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">账策云帆</h1>
            <p className="text-sm text-emerald-100/90">Bank-Copilot</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {items.map((item) => (
            <li key={item.id}>
              <button
                type="button"
                onClick={() => onSelect(item.id as PanelId)}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                  active === item.id
                    ? `${item.color} text-white shadow-lg shadow-emerald-900/20`
                    : 'text-emerald-100/80 hover:bg-white/10 hover:text-white'
                }`}
              >
                <item.icon size={20} />
                <span>{item.label}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <div className="relative p-6 border-t border-white/10" ref={userMenuRef}>
        <button
          type="button"
          onClick={() => setIsUserMenuOpen((prev) => !prev)}
          className="flex w-full items-center justify-between rounded-2xl border border-white/10 bg-white/5 px-3 py-3 text-left text-sm transition hover:bg-white/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/70"
        >
          <div className="flex items-center space-x-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white/20 text-white font-semibold">
              Z
            </div>
            <div>
              <p className="font-medium text-white">张会计</p>
              <p className="text-xs text-emerald-100/80">@finance.ops</p>
            </div>
          </div>
          <User size={18} className="text-emerald-100/80" />
        </button>

        {isUserMenuOpen ? (
          <div className="absolute bottom-[calc(100%+12px)] left-6 right-6 z-30 rounded-3xl border border-white/15 bg-emerald-950/90 p-4 text-sm text-white shadow-2xl shadow-emerald-950/40 backdrop-blur-xl">
            <div className="flex items-center gap-3 pb-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-emerald-500 text-base font-semibold text-white shadow-inner shadow-emerald-900/30">
                ZD
              </div>
              <div>
                <p className="text-base font-semibold">张会计</p>
                <p className="text-xs text-emerald-100/70">@finance.ops</p>
              </div>
            </div>
            <div className="my-3 h-px bg-white/10" />
            <div className="space-y-1">
              {userMenuItems.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => {
                    item.action();
                    setIsUserMenuOpen(false);
                  }}
                  className={cn(
                    'flex w-full items-center justify-between rounded-2xl px-3 py-2 text-left transition',
                    item.destructive
                      ? 'text-red-100 hover:bg-red-500/20 hover:text-white'
                      : 'text-emerald-50 hover:bg-white/10',
                  )}
                >
                  <div className="flex items-center gap-3">
                    <item.icon className="h-4 w-4 opacity-80" />
                    <div>
                      <p className="text-sm font-medium">{item.label}</p>
                      {item.description ? (
                        <p className="text-xs text-emerald-100/70">{item.description}</p>
                      ) : null}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

type DashboardHeaderProps = {
  title: string;
  description?: string;
};

function DashboardHeader({ title, description }: DashboardHeaderProps) {
  return (
    <header className="bg-white shadow-sm px-6 py-4 flex items-center justify-between">
      <div>
        <h1 className="text-2xl font-bold text-gray-800">{title}</h1>
        {description ? <p className="text-sm text-gray-500 mt-1">{description}</p> : null}
      </div>

      <div className="flex items-center space-x-4">
        <div className="relative">
          <input
            type="text"
            placeholder="全局搜索..."
            className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <div className="absolute left-3 top-2.5 text-gray-400">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
        </div>

        <div className="relative">
          <Bell size={25} className="text-gray-600 cursor-pointer" />
          <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
            3
          </span>
        </div>
      </div>
    </header>
  );
}

type SystemSettingsDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

function SystemSettingsDialog({ open, onOpenChange }: SystemSettingsDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl border-emerald-100 bg-white/95">
        <DialogHeader>
          <DialogTitle className="text-2xl text-gray-900">系统设置</DialogTitle>
          <DialogDescription className="text-gray-500">
            调整偏好、通知与集成配置，帮助团队在一个面板中完成常用的运维动作。
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5 py-2">
          <section className="rounded-3xl border border-emerald-100/70 bg-emerald-50/40 p-4">
            <p className="text-sm font-semibold text-emerald-900">通知策略</p>
            <p className="text-xs text-emerald-700/80">同步账务异常、AI 工作区状态与审批提醒。</p>
            <div className="mt-4 space-y-4">
              <div>
                <Label htmlFor="notify-email" className="text-xs text-emerald-900/80">
                  通知邮箱
                </Label>
                <Input
                  id="notify-email"
                  defaultValue="ops-team@bankcopilot.cn"
                  className="mt-1 bg-white/90"
                  placeholder="name@example.com"
                />
              </div>
              <div className="flex items-center justify-between rounded-2xl bg-white/70 px-3 py-2">
                <div>
                  <p className="text-sm font-medium text-gray-900">即时提醒</p>
                  <p className="text-xs text-gray-500">新增 AI 会话或知识库写入时同步推送</p>
                </div>
                <Switch defaultChecked className="data-[state=checked]:bg-emerald-500" />
              </div>
              <div className="flex items-center justify-between rounded-2xl bg-white/70 px-3 py-2">
                <div>
                  <p className="text-sm font-medium text-gray-900">周报汇总</p>
                  <p className="text-xs text-gray-500">每周一 09:00 自动发送执行摘要</p>
                </div>
                <Switch defaultChecked={false} className="data-[state=checked]:bg-emerald-500" />
              </div>
            </div>
          </section>

          <section className="rounded-3xl border border-gray-100 bg-gray-50/60 p-4">
            <p className="text-sm font-semibold text-gray-900">安全与集成</p>
            <p className="text-xs text-gray-500">可配置单点登录、Webhook 以及审计策略。</p>
            <div className="mt-4 space-y-3">
              <div>
                <Label htmlFor="webhook" className="text-xs text-gray-500">
                  Webhook Endpoint
                </Label>
                <Input
                  id="webhook"
                  placeholder="https://"
                  defaultValue="https://api.bankcopilot.cn/hooks/accounting"
                  className="mt-1"
                />
              </div>
              <div>
                <Label htmlFor="audit-note" className="text-xs text-gray-500">
                  审计备注
                </Label>
                <Textarea
                  id="audit-note"
                  className="mt-1 min-h-[90px] resize-none"
                  placeholder="记录本次配置变更的背景与执行人"
                  defaultValue="由财务共享中心统一维护访问白名单。"
                />
              </div>
            </div>
          </section>
        </div>

        <DialogFooter className="gap-2">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            取消
          </Button>
          <Button className="bg-emerald-600 hover:bg-emerald-700 text-white">保存设置</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function DashboardOverview() {
  return (
    <section className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {dashboardStats.map((stat) => (
          <StatCard key={stat.id} {...stat} />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ChartCard title="月度收支趋势">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={incomeExpenseData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="month" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
                }}
                labelStyle={{ fontWeight: 'bold' }}
              />
              <Legend
                wrapperStyle={{
                  paddingTop: '20px',
                }}
                formatter={(value) => (
                  <span style={{ color: value === '支出' ? '#ef4444' : '#10b981' }}>{value}</span>
                )}
              />
              <Line
                name="支出"
                type="monotone"
                dataKey="expense"
                stroke="#ef4444"
                strokeWidth={2}
                dot={{ r: 4, fill: '#ef4444' }}
                activeDot={{ r: 6 }}
              />
              <Line
                name="收入"
                type="monotone"
                dataKey="income"
                stroke="#10b981"
                strokeWidth={2}
                dot={{ r: 4, fill: '#10b981' }}
                activeDot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="账户分布">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={accountDistributionData}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {accountDistributionData.map((entry, index) => (
                  <Cell key={`cell-${entry.name}`} fill={ACCOUNT_COLORS[index % ACCOUNT_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #e2e8f0',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
                }}
              />
              <Legend
                layout="horizontal"
                verticalAlign="bottom"
                align="center"
                wrapperStyle={{
                  paddingTop: '20px',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <RealtimeInsights />
    </section>
  );
}

type PersonalLearningPanelProps = {
  onGuide: (intent: AIGuidanceIntent) => void;
};

function PersonalLearningPanel({ onGuide }: PersonalLearningPanelProps) {
  return (
    <section className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-[1.2fr,0.8fr]">
        <div className="rounded-3xl border border-indigo-100 bg-gradient-to-br from-indigo-50 via-white to-white p-6 shadow-sm">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-indigo-600">个人学习</p>
              <h2 className="mt-2 text-2xl font-semibold text-gray-900">我的学习进度与掌握度</h2>
              <p className="mt-2 text-sm text-gray-600">
                针对 {personalLearningProfile.targetRole} 目标，系统实时同步掌握情况与复盘建议。
              </p>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-500">{personalLearningProfile.cycle}</p>
              <p className="text-lg font-semibold text-gray-900">{personalLearningProfile.mentor}</p>
            </div>
          </div>

          <div className="mt-6 grid gap-4 md:grid-cols-3">
            {personalLearningHighlights.map((item) => (
              <div
                key={item.label}
                className="rounded-2xl border border-indigo-100 bg-white/90 p-4 shadow-[0_12px_30px_-20px_rgba(79,70,229,0.7)]"
              >
                <p className="text-xs font-medium text-indigo-600">{item.label}</p>
                <p className="mt-2 text-2xl font-semibold text-gray-900">{item.value}</p>
                <p className="text-xs text-gray-500">{item.helper}</p>
                <p className="text-xs text-emerald-600">{item.trend}</p>
              </div>
            ))}
          </div>

          <div className="mt-6">
            <div className="flex items-center justify-between text-sm text-gray-600">
              <span>学习进度</span>
              <span className="font-semibold text-gray-900">{personalLearningProfile.progress}%</span>
            </div>
            <div className="mt-2 h-3 w-full rounded-full bg-indigo-100">
              <div
                className="h-full rounded-full bg-indigo-500 transition-all"
                style={{ width: `${personalLearningProfile.progress}%` }}
              />
            </div>
            <p className="mt-2 text-xs text-gray-500">
              已掌握 {personalLearningProfile.mastered}/{personalLearningProfile.total} · 累计学习{' '}
              {personalLearningProfile.hours}
            </p>
            <div className="mt-4 flex flex-wrap gap-3 text-sm text-gray-600">
              <span className="inline-flex items-center gap-2 rounded-full bg-indigo-100 px-3 py-1 text-indigo-700">
                <CalendarClock className="h-4 w-4" />
                {personalLearningProfile.nextCheckpoint}
              </span>
              <span className="inline-flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1">
                <Target className="h-4 w-4 text-slate-500" />
                {personalLearningProfile.targetRole}
              </span>
            </div>
          </div>
        </div>

        <div className="rounded-3xl border border-slate-100 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-[0.3em] text-slate-500">量身学习计划</p>
              <h3 className="mt-1 text-lg font-semibold text-gray-900">AI 生成的两日安排</h3>
            </div>
            <Button size="sm" variant="outline" onClick={() => onGuide({ topic: '学习计划优化', prompt: '请根据我当前的学习数据，重新优化未来两天的训练节奏，并给出复盘要点。' })}>
              <Sparkles className="mr-1 h-4 w-4" />
              调整计划
            </Button>
          </div>

          <div className="mt-5 space-y-4">
            {personalPlanTimeline.map((slot) => (
              <div key={slot.id} className="rounded-2xl border border-gray-100 p-4">
                <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <p className="text-sm font-semibold text-gray-900">{slot.title}</p>
                    <p className="text-xs text-gray-500">{slot.focus}</p>
                  </div>
                  <span className="text-xs text-indigo-600">{slot.due}</span>
                </div>
                <ul className="mt-3 space-y-2 text-sm text-gray-600">
                  {slot.tasks.map((task) => (
                    <li key={task} className="flex items-center gap-2">
                      <Sparkles className="h-4 w-4 text-indigo-400" />
                      <span>{task}</span>
                    </li>
                  ))}
                </ul>
                <div className="mt-4 flex flex-wrap gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onGuide({ topic: slot.focus, prompt: slot.aiPrompt })}
                  >
                    <MessageCircle className="mr-1 h-4 w-4" />
                    AI 指导
                  </Button>
                  <Button
                    size="sm"
                    className="bg-indigo-600 text-white hover:bg-indigo-700"
                    onClick={() => onGuide({ topic: `${slot.focus} 检查`, prompt: `帮我检查是否按计划完成：${slot.focus}。如果存在薄弱知识点，请生成补救建议。` })}
                  >
                    <ArrowUpRight className="mr-1 h-4 w-4" />
                    追踪进度
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {personalKnowledgePoints.map((point) => (
          <article key={point.id} className="rounded-3xl border border-gray-100 bg-white p-5 shadow-sm">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-lg font-semibold text-gray-900">{point.title}</p>
                <p className="text-xs text-gray-500">最近复习：{point.lastReview}</p>
              </div>
              <span
                className={cn(
                  'rounded-full px-3 py-1 text-xs font-semibold',
                  point.status.includes('需') ? 'bg-red-50 text-red-600' : point.status.includes('保持') ? 'bg-emerald-50 text-emerald-600' : 'bg-amber-50 text-amber-700',
                )}
              >
                {point.status}
              </span>
            </div>

            <div className="mt-4">
              <div className="flex items-center justify-between text-sm text-gray-600">
                <span>掌握度</span>
                <span className="font-semibold text-gray-900">{point.mastery}%</span>
              </div>
              <div className="mt-2 h-2 rounded-full bg-gray-100">
                <div
                  className={cn(
                    'h-full rounded-full transition-all',
                    point.status.includes('需') ? 'bg-red-400' : 'bg-indigo-500',
                  )}
                  style={{ width: `${point.mastery}%` }}
                />
              </div>
              <p className="mt-1 text-xs text-gray-500">趋势 {point.trend}</p>
            </div>

            <div className="mt-4 flex flex-wrap gap-2 text-xs text-gray-600">
              {point.tags.map((tag) => (
                <span key={tag} className="rounded-full bg-gray-100 px-3 py-1">
                  {tag}
                </span>
              ))}
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => onGuide({ topic: point.title, prompt: point.aiPrompt })}
              >
                <ArrowUpRight className="mr-1 h-4 w-4" />
                前往 AI 指导
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => onGuide({ topic: `${point.title} 错题巩固`, prompt: `请根据 ${point.title} 出 3 道错题巩固题，并给出正确答案与解析。` })}
              >
                再出几题
              </Button>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function TrainingPanel() {
  return (
    <section className="space-y-6">
      <div className="rounded-3xl border border-amber-200 bg-gradient-to-br from-amber-50 via-white to-white p-6 shadow-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-amber-600">企业培训</p>
            <h2 className="mt-2 text-2xl font-semibold text-gray-900">员工学习进度与掌握度一屏掌握</h2>
            <p className="mt-2 text-sm text-gray-600">
              聚焦重点课程，自动同步各岗位已完成、未完成知识点，方便 HR 与部门主管即时跟进。
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Button variant="outline" size="sm">
              导出进度
            </Button>
            <Button size="sm">安排补训</Button>
          </div>
        </div>
        <div className="mt-6 grid gap-4 md:grid-cols-3">
          {trainingOverviewStats.map((item) => (
            <div key={item.label} className="rounded-2xl border border-amber-100 bg-white/90 p-4 shadow-[0_8px_30px_rgba(251,191,36,0.15)]">
              <p className="text-xs font-medium text-amber-600">{item.label}</p>
              <p className="mt-2 text-2xl font-semibold text-gray-900">{item.value}</p>
              <p className="text-xs text-gray-500">{item.helper}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[320px,1fr]">
        <div className="space-y-4">
          {trainingKnowledgeTracks.map((track) => {
            const percent = Math.round((track.completed / track.total) * 100);
            return (
              <div key={track.id} className="rounded-2xl border border-amber-100 bg-white p-4 shadow-sm">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-semibold text-gray-900">{track.title}</p>
                    <p className="text-xs text-gray-500">
                      {track.owner} · {track.highlight}
                    </p>
                  </div>
                  <span className="text-sm font-semibold text-amber-600">
                    {track.completed}/{track.total}
                  </span>
                </div>
                <div className="mt-3 h-2 rounded-full bg-gray-100">
                  <div className="h-full rounded-full bg-amber-400 transition-all" style={{ width: `${percent}%` }} />
                </div>
                <p className="mt-2 text-xs text-gray-500">预计本周完成率 {percent}%</p>
              </div>
            );
          })}

          <div className="rounded-2xl border border-dashed border-amber-200 bg-amber-50/60 p-4 text-sm text-amber-700">
            <p className="font-semibold">跟进建议</p>
            <p className="mt-1 text-xs leading-relaxed">
              系统已识别 3 名员工在「AI+业务操作规范」模块滞后，可直接安排补训并推送通知，避免影响下周期机器人上线。
            </p>
          </div>
        </div>

        <div className="space-y-4">
          {trainingRecords.map((record) => (
            <article key={record.id} className="rounded-2xl border border-gray-100 bg-white p-5 shadow-sm">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <p className="text-lg font-semibold text-gray-900">{record.name}</p>
                  <p className="text-sm text-gray-500">{record.role}</p>
                  <p className="text-xs text-amber-600">{record.focus}</p>
                </div>
                <span
                  className={cn(
                    'inline-flex items-center rounded-full px-3 py-1 text-xs font-medium',
                    record.status === '需督促'
                      ? 'bg-red-50 text-red-600'
                      : record.status.includes('领先')
                        ? 'bg-emerald-50 text-emerald-600'
                        : 'bg-amber-50 text-amber-700',
                  )}
                >
                  {record.status}
                </span>
              </div>

              <div className="mt-4">
                <div className="flex items-center justify-between text-sm text-gray-600">
                  <span>完成度</span>
                  <span className="font-semibold text-gray-900">{record.progress}%</span>
                </div>
                <div className="mt-2 h-2 rounded-full bg-gray-100">
                  <div className="h-full rounded-full bg-emerald-500 transition-all" style={{ width: `${record.progress}%` }} />
                </div>
                <p className="mt-2 text-xs text-gray-500">最后更新：{record.lastUpdate}</p>
              </div>

              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-emerald-600">已完成知识点</p>
                  {record.completedTopics.length ? (
                    <ul className="mt-2 space-y-2">
                      {record.completedTopics.map((topic) => (
                        <li key={topic} className="flex items-start gap-2 text-sm text-gray-700">
                          <CheckCircle2 className="h-4 w-4 flex-shrink-0 text-emerald-500" />
                          <span>{topic}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-2 text-xs text-gray-500">暂无记录</p>
                  )}
                </div>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-amber-600">未完成知识点</p>
                  {record.pendingTopics.length ? (
                    <ul className="mt-2 space-y-2">
                      {record.pendingTopics.map((topic) => (
                        <li key={topic} className="flex items-start gap-2 text-sm text-gray-700">
                          <Clock3 className="h-4 w-4 flex-shrink-0 text-amber-500" />
                          <span>{topic}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="mt-2 text-xs text-gray-500">已全部掌握</p>
                  )}
                </div>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

type CollectionCreatePayload = {
  name: string;
  displayName: string;
  description?: string;
  vectorSize: number;
  distance: string;
};

type CollectionEditPayload = {
  displayName?: string;
  description?: string;
  tags?: string[];
};

type ChunkFormValues = {
  id?: string;
  title?: string;
  text: string;
  source?: string;
  tags: string[];
  metadata: Record<string, unknown>;
};

function KnowledgeBasePanel() {
  const [collections, setCollections] = useState<KnowledgeBase[]>([]);
  const [collectionsLoading, setCollectionsLoading] = useState(true);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [chunks, setChunks] = useState<KnowledgeChunk[]>([]);
  const [chunksLoading, setChunksLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [nextOffset, setNextOffset] = useState<string | number | undefined>();
  const [createOpen, setCreateOpen] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [chunkDialogOpen, setChunkDialogOpen] = useState(false);
  const [chunkDialogMode, setChunkDialogMode] = useState<'create' | 'edit'>('create');
  const [editingChunk, setEditingChunk] = useState<KnowledgeChunk | null>(null);
  const [deletingCollection, setDeletingCollection] = useState<string | null>(null);
  const [deletingChunkId, setDeletingChunkId] = useState<string | null>(null);
  const activeCollectionRef = useRef<string | null>(null);

  const selectedCollection = useMemo(() => {
    if (!selectedName) {
      return null;
    }
    return collections.find((item) => item.name === selectedName) ?? null;
  }, [collections, selectedName]);

  const totalChunks = useMemo(
    () => collections.reduce((sum, item) => sum + item.chunkCount, 0),
    [collections],
  );

  const fetchCollections = useCallback(async () => {
    setCollectionsLoading(true);
    try {
      const response = await fetch('/api/knowledge/collections', {
        cache: 'no-store',
      });
      const json = await response.json();
      if (!response.ok) {
        throw new Error(json?.error ?? '加载知识库失败');
      }
      const data: KnowledgeBase[] = Array.isArray(json.data) ? json.data : [];
      setCollections(data);
      setSelectedName((prev) => {
        if (!data.length) {
          return null;
        }
        if (prev && data.some((item) => item.name === prev)) {
          return prev;
        }
        return data[0]?.name ?? null;
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : '无法加载知识库';
      toast.error(message);
    } finally {
      setCollectionsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCollections();
  }, [fetchCollections]);

  const loadChunks = useCallback(
    async (collectionName: string, options: { append?: boolean; offset?: string | number } = {}) => {
      const append = Boolean(options.append);
      if (append) {
        setLoadingMore(true);
      } else {
        setChunksLoading(true);
      }
      try {
        const params = new URLSearchParams({ limit: '50' });
        if (options.offset !== undefined && options.offset !== null) {
          params.set('offset', String(options.offset));
        }
        const response = await fetch(
          `/api/knowledge/collections/${encodeURIComponent(collectionName)}/chunks?${params.toString()}`,
          { cache: 'no-store' },
        );
        const json = await response.json();
        if (!response.ok) {
          throw new Error(json?.error ?? '加载 Chunk 列表失败');
        }
        if (activeCollectionRef.current !== collectionName) {
          return;
        }
        const incoming: KnowledgeChunk[] = Array.isArray(json.data) ? json.data : [];
        setChunks((prev) => (append ? [...prev, ...incoming] : incoming));
        setNextOffset(json.nextOffset ?? undefined);
      } catch (error) {
        const message = error instanceof Error ? error.message : '加载 Chunk 列表失败';
        toast.error(message);
      } finally {
        if (append) {
          setLoadingMore(false);
        } else {
          setChunksLoading(false);
        }
      }
    },
    [],
  );

  useEffect(() => {
    activeCollectionRef.current = selectedName;
    if (!selectedName) {
      setChunks([]);
      setNextOffset(undefined);
      setChunksLoading(false);
      setLoadingMore(false);
      return;
    }
    loadChunks(selectedName);
  }, [selectedName, loadChunks]);

  const handleCreateCollection = useCallback(
    async (payload: CollectionCreatePayload) => {
      const response = await fetch('/api/knowledge/collections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const json = await response.json();
      if (!response.ok) {
        const message = json?.error ?? '创建知识库失败';
        toast.error(message);
        throw new Error(message);
      }
      const created: KnowledgeBase | undefined = json.data;
      setSelectedName(created?.name ?? payload.name);
      toast.success('知识库创建成功');
      await fetchCollections();
    },
    [fetchCollections],
  );

  const handleUpdateCollection = useCallback(
    async (payload: CollectionEditPayload) => {
      if (!selectedName) {
        return;
      }
      const response = await fetch(`/api/knowledge/collections/${encodeURIComponent(selectedName)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const json = await response.json();
      if (!response.ok) {
        const message = json?.error ?? '更新知识库失败';
        toast.error(message);
        throw new Error(message);
      }
      const updated: KnowledgeBase | undefined = json.data;
      if (updated) {
        setCollections((prev) => prev.map((item) => (item.name === updated.name ? updated : item)));
      } else {
        await fetchCollections();
      }
      toast.success('知识库信息已更新');
    },
    [selectedName, fetchCollections],
  );

  const handleDeleteCollection = useCallback(async () => {
    if (!selectedCollection) {
      return;
    }
    const confirmed = window.confirm(`确认删除知识库「${selectedCollection.metadata.displayName}」？该操作不可恢复。`);
    if (!confirmed) {
      return;
    }
    setDeletingCollection(selectedCollection.name);
    try {
      const response = await fetch(`/api/knowledge/collections/${encodeURIComponent(selectedCollection.name)}`, {
        method: 'DELETE',
      });
      const json = await response.json();
      if (!response.ok) {
        throw new Error(json?.error ?? '删除知识库失败');
      }
      toast.success('知识库已删除');
      setSelectedName(null);
      await fetchCollections();
    } catch (error) {
      const message = error instanceof Error ? error.message : '删除知识库失败';
      toast.error(message);
    } finally {
      setDeletingCollection(null);
    }
  }, [selectedCollection, fetchCollections]);

  const handleSaveChunk = useCallback(
    async (values: ChunkFormValues) => {
      if (!selectedName) {
        throw new Error('请选择知识库');
      }
      const isEdit = Boolean(values.id);
      const url = isEdit
        ? `/api/knowledge/collections/${encodeURIComponent(selectedName)}/chunks/${values.id}`
        : `/api/knowledge/collections/${encodeURIComponent(selectedName)}/chunks`;
      const response = await fetch(url, {
        method: isEdit ? 'PATCH' : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values),
      });
      const json = await response.json();
      if (!response.ok) {
        const message = json?.error ?? '保存 Chunk 失败';
        toast.error(message);
        throw new Error(message);
      }
      toast.success(isEdit ? 'Chunk 已更新' : 'Chunk 已新增');
      setChunkDialogOpen(false);
      setEditingChunk(null);
      await loadChunks(selectedName);
      if (!isEdit) {
        await fetchCollections();
      }
    },
    [selectedName, loadChunks, fetchCollections],
  );

  const handleDeleteChunk = useCallback(
    async (chunk: KnowledgeChunk) => {
      if (!selectedName) {
        return;
      }
      const confirmed = window.confirm('确认删除该 Chunk 吗？');
      if (!confirmed) {
        return;
      }
      setDeletingChunkId(chunk.id);
      try {
        const response = await fetch(
          `/api/knowledge/collections/${encodeURIComponent(selectedName)}/chunks/${chunk.id}`,
          {
            method: 'DELETE',
          },
        );
        const json = await response.json();
        if (!response.ok) {
          throw new Error(json?.error ?? '删除 Chunk 失败');
        }
        toast.success('Chunk 已删除');
        await loadChunks(selectedName);
        await fetchCollections();
      } catch (error) {
        const message = error instanceof Error ? error.message : '删除 Chunk 失败';
        toast.error(message);
      } finally {
        setDeletingChunkId(null);
      }
    },
    [selectedName, loadChunks, fetchCollections],
  );

  const handleLoadMore = useCallback(() => {
    if (!selectedName || nextOffset === undefined) {
      return;
    }
    loadChunks(selectedName, { append: true, offset: nextOffset });
  }, [selectedName, nextOffset, loadChunks]);

  if (collectionsLoading) {
    return <KnowledgePanelSkeleton />;
  }

  if (collections.length === 0) {
    return (
      <FeaturePlaceholder
        icon={Database}
        title="知识库"
        description="集中管理制度、合同、审批流等资料，可与问答机器人和审批服务共享。"
        actions={[
          { label: '上传文档', variant: 'primary' },
          { label: '创建分类', variant: 'secondary' },
        ]}
      />
    );
  }

  return (
    <section className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] text-emerald-600">知识库管理</p>
          <h2 className="mt-2 text-2xl font-semibold text-gray-900">统一的知识资产中心</h2>
          <p className="mt-2 text-sm text-gray-500">
            当前共 {collections.length} 个知识库，累计 {totalChunks} 条 Chunk，可直接映射到 Qdrant 集合。
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="outline" size="sm" onClick={fetchCollections}>
            <RefreshCw className="h-4 w-4" />
            刷新列表
          </Button>
          <Button size="sm" onClick={() => setCreateOpen(true)}>
            <Plus className="h-4 w-4" />
            新建知识库
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-emerald-100 bg-emerald-50/70 p-4">
          <p className="text-xs uppercase tracking-wide text-emerald-700">知识库数量</p>
          <p className="mt-2 text-3xl font-semibold text-gray-900">{collections.length}</p>
          <p className="text-xs text-emerald-700/80">与业务领域一一对应，便于权限隔离</p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
          <p className="text-xs uppercase tracking-wide text-slate-600">Chunk 总量</p>
          <p className="mt-2 text-3xl font-semibold text-gray-900">{totalChunks}</p>
          <p className="text-xs text-slate-600">包括结构化与非结构化文本段落</p>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[340px,1fr]">
        <KnowledgeSidebar
          items={collections}
          selectedName={selectedName}
          onSelect={setSelectedName}
        />

        {selectedCollection ? (
          <CollectionDetailSection
            collection={selectedCollection}
            chunks={chunks}
            chunksLoading={chunksLoading}
            loadingMore={loadingMore}
            nextOffset={nextOffset}
            deletingChunkId={deletingChunkId}
            onRefreshChunks={() => {
              if (selectedCollection) {
                loadChunks(selectedCollection.name);
              }
            }}
            onCreateChunk={() => {
              setChunkDialogMode('create');
              setEditingChunk(null);
              setChunkDialogOpen(true);
            }}
            onEditChunk={(chunk) => {
              setChunkDialogMode('edit');
              setEditingChunk(chunk);
              setChunkDialogOpen(true);
            }}
            onDeleteChunk={handleDeleteChunk}
            onLoadMore={handleLoadMore}
            onEditCollection={() => setEditOpen(true)}
            onDeleteCollection={handleDeleteCollection}
            deletingCollection={deletingCollection === selectedCollection.name}
          />
        ) : (
          <div className="rounded-2xl border border-dashed border-emerald-200 bg-white/60 p-8 text-center text-gray-500">
            请选择左侧的知识库查看详情
          </div>
        )}
      </div>

      <CollectionCreateDialog open={createOpen} onOpenChange={setCreateOpen} onSubmit={handleCreateCollection} />
      <CollectionEditDialog
        open={editOpen}
        onOpenChange={setEditOpen}
        collection={selectedCollection}
        onSubmit={handleUpdateCollection}
      />
      <ChunkEditorDialog
        open={chunkDialogOpen}
        onOpenChange={setChunkDialogOpen}
        mode={chunkDialogMode}
        chunk={editingChunk}
        onSubmit={handleSaveChunk}
      />
    </section>
  );
}

function KnowledgePanelSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-8 w-56" />
      <div className="grid gap-6 lg:grid-cols-[340px,1fr]">
        <Skeleton className="h-[460px] rounded-2xl" />
        <Skeleton className="h-[460px] rounded-2xl" />
      </div>
    </div>
  );
}

type KnowledgeSidebarProps = {
  items: KnowledgeBase[];
  selectedName: string | null;
  onSelect: (name: string) => void;
};

function KnowledgeSidebar({ items, selectedName, onSelect }: KnowledgeSidebarProps) {
  return (
    <aside className="rounded-2xl border border-emerald-100 bg-white shadow-sm">
      <div className="border-b border-emerald-50 p-5">
        <p className="text-sm font-semibold text-gray-900">知识库列表</p>
        <p className="mt-1 text-xs text-gray-500">一个知识库对应 Qdrant 中的一个集合</p>
      </div>
      <div className="max-h-[560px] space-y-3 overflow-y-auto p-4">
        {items.map((item) => (
          <button
            key={item.name}
            type="button"
            onClick={() => onSelect(item.name)}
            className={cn(
              'w-full rounded-xl border p-4 text-left transition hover:border-emerald-300',
              selectedName === item.name ? 'border-emerald-400 bg-emerald-50 shadow-sm' : 'border-gray-100 bg-white',
            )}
          >
            <div className="flex items-center justify-between gap-3">
              <p className="font-medium text-gray-900">{item.metadata.displayName}</p>
              <span
                className={cn(
                  'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold',
                  item.status === 'green'
                    ? 'bg-emerald-100 text-emerald-700'
                    : 'bg-amber-100 text-amber-700',
                )}
              >
                <span className="h-1.5 w-1.5 rounded-full bg-current" />
                {item.status?.toUpperCase() ?? 'UNKNOWN'}
              </span>
            </div>
            <p className="mt-1 min-h-[1.5rem] text-xs text-gray-500">
              {item.metadata.description || '暂无描述'}
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-gray-500">
              <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2 py-0.5">
                <Layers className="h-3 w-3" />
                维度 {item.vectorSize}
              </span>
              <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2 py-0.5">
                <Database className="h-3 w-3" />
                {item.chunkCount} 条
              </span>
            </div>
          </button>
        ))}
      </div>
    </aside>
  );
}

type CollectionDetailProps = {
  collection: KnowledgeBase;
  chunks: KnowledgeChunk[];
  chunksLoading: boolean;
  loadingMore: boolean;
  nextOffset?: string | number;
  deletingChunkId: string | null;
  onRefreshChunks: () => void;
  onCreateChunk: () => void;
  onEditChunk: (chunk: KnowledgeChunk) => void;
  onDeleteChunk: (chunk: KnowledgeChunk) => void | Promise<void>;
  onLoadMore: () => void;
  onEditCollection: () => void;
  onDeleteCollection: () => void | Promise<void>;
  deletingCollection: boolean;
};

function CollectionDetailSection({
  collection,
  chunks,
  chunksLoading,
  loadingMore,
  nextOffset,
  deletingChunkId,
  onRefreshChunks,
  onCreateChunk,
  onEditChunk,
  onDeleteChunk,
  onLoadMore,
  onEditCollection,
  onDeleteCollection,
  deletingCollection,
}: CollectionDetailProps) {
  return (
    <section className="flex flex-col rounded-2xl border border-emerald-100 bg-white shadow-sm">
      <div className="flex flex-col gap-4 border-b border-emerald-50 p-6 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-emerald-600">选中知识库</p>
          <h3 className="mt-2 text-2xl font-semibold text-gray-900">{collection.metadata.displayName}</h3>
          <p className="mt-2 text-sm text-gray-600">
            {collection.metadata.description || '暂无描述，可在「编辑信息」中补充。'}
          </p>
          {collection.metadata.tags?.length ? (
            <div className="mt-3 flex flex-wrap gap-2">
              {collection.metadata.tags.map((tag) => (
                <span key={tag} className="rounded-full bg-emerald-50 px-3 py-0.5 text-xs text-emerald-700">
                  #{tag}
                </span>
              ))}
            </div>
          ) : null}
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm" onClick={onEditCollection}>
            <Edit3 className="h-4 w-4" />
            编辑信息
          </Button>
          <Button
            variant="destructive"
            size="sm"
            onClick={() => void onDeleteCollection()}
            disabled={deletingCollection}
          >
            <Trash2 className="h-4 w-4" />
            {deletingCollection ? '删除中...' : '删除知识库'}
          </Button>
        </div>
      </div>

      <div className="grid gap-4 border-b border-emerald-50 p-6 sm:grid-cols-3">
        <StatPill label="Chunk 数量" value={`${collection.chunkCount} 条`} />
        <StatPill label="向量维度" value={`${collection.vectorSize}`} />
        <StatPill label="距离算法" value={collection.distance ?? 'Cosine'} />
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-emerald-50 px-6 py-4">
        <div>
          <p className="font-semibold text-gray-900">Chunk 列表</p>
          <p className="text-sm text-gray-500">支持新增、编辑与删除，内容会实时写入 Qdrant</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm" onClick={onRefreshChunks}>
            <RefreshCw className="h-4 w-4" />
            刷新
          </Button>
          <Button size="sm" onClick={onCreateChunk}>
            <Plus className="h-4 w-4" />
            新增 Chunk
          </Button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <ChunkList
          chunks={chunks}
          loading={chunksLoading}
          loadingMore={loadingMore}
          nextOffset={nextOffset}
          deletingChunkId={deletingChunkId}
          onEdit={onEditChunk}
          onDelete={onDeleteChunk}
          onLoadMore={onLoadMore}
        />
      </div>
    </section>
  );
}

function StatPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border border-gray-100 bg-gray-50 px-4 py-3">
      <p className="text-xs uppercase tracking-wide text-gray-500">{label}</p>
      <p className="mt-2 text-xl font-semibold text-gray-900">{value}</p>
    </div>
  );
}

type ChunkListProps = {
  chunks: KnowledgeChunk[];
  loading: boolean;
  loadingMore: boolean;
  nextOffset?: string | number;
  deletingChunkId: string | null;
  onEdit: (chunk: KnowledgeChunk) => void;
  onDelete: (chunk: KnowledgeChunk) => void | Promise<void>;
  onLoadMore: () => void;
};

function ChunkList({
  chunks,
  loading,
  loadingMore,
  nextOffset,
  deletingChunkId,
  onEdit,
  onDelete,
  onLoadMore,
}: ChunkListProps) {
  if (loading) {
    return (
      <div className="space-y-3 p-6">
        {[0, 1, 2].map((item) => (
          <div key={item} className="space-y-2 rounded-2xl border border-gray-100 p-4">
            <Skeleton className="h-4 w-40" />
            <Skeleton className="h-6 w-full" />
            <Skeleton className="h-6 w-2/3" />
          </div>
        ))}
      </div>
    );
  }

  if (!chunks.length) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 p-10 text-center text-gray-500">
        <Database className="h-10 w-10 text-emerald-400" />
        <p className="text-sm">暂未添加 Chunk，可点击「新增 Chunk」快速接入内容。</p>
      </div>
    );
  }

  return (
    <div className="space-y-4 p-6">
      {chunks.map((chunk) => {
        const preview = chunk.text?.length > 400 ? `${chunk.text.slice(0, 400)}…` : chunk.text;
        const metadataPreview = formatMetadataPreview(chunk.metadata);

        return (
          <article key={chunk.id} className="space-y-3 rounded-2xl border border-gray-100 p-4 shadow-sm">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <p className="text-lg font-semibold text-gray-900">
                  {chunk.title || `未命名 Chunk (${chunk.id.slice(0, 8)})`}
                </p>
                <p className="text-xs text-gray-500">{formatDateLabel(chunk.updatedAt)}</p>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button variant="ghost" size="sm" onClick={() => onEdit(chunk)}>
                  <Edit3 className="h-4 w-4" />
                  编辑
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-red-600 hover:text-red-700"
                  onClick={() => onDelete(chunk)}
                  disabled={deletingChunkId === chunk.id}
                >
                  <Trash2 className="h-4 w-4" />
                  {deletingChunkId === chunk.id ? '删除中...' : '删除'}
                </Button>
              </div>
            </div>

            <p className="whitespace-pre-line text-sm text-gray-600">{preview}</p>

            <div className="flex flex-wrap items-center gap-3 text-xs text-gray-500">
              {chunk.source ? <span>来源：{chunk.source}</span> : null}
              {chunk.tags?.length ? (
                <span className="inline-flex flex-wrap gap-2">
                  {chunk.tags.map((tag) => (
                    <span key={tag} className="rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-600">
                      #{tag}
                    </span>
                  ))}
                </span>
              ) : null}
            </div>

            {metadataPreview ? (
              <div className="text-xs text-gray-500">
                <p className="font-medium text-gray-600">Metadata</p>
                <p className="mt-1">{metadataPreview}</p>
              </div>
            ) : null}
          </article>
        );
      })}

      {nextOffset !== undefined ? (
        <Button
          variant="outline"
          className="w-full"
          onClick={onLoadMore}
          disabled={loadingMore}
        >
          {loadingMore ? '加载中...' : '加载更多 Chunk'}
        </Button>
      ) : null}
    </div>
  );
}

function formatMetadataPreview(metadata?: Record<string, unknown>) {
  if (!metadata) {
    return '';
  }
  const entries = Object.entries(metadata);
  if (!entries.length) {
    return '';
  }
  return entries
    .slice(0, 3)
    .map(([key, value]) => `${key}: ${stringifyMetadataValue(value)}`)
    .join(' · ');
}

function stringifyMetadataValue(value: unknown) {
  if (value === null || value === undefined) {
    return 'null';
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value);
    } catch {
      return '[object]';
    }
  }
  return String(value);
}

function formatDateLabel(input?: string) {
  if (!input) {
    return '刚刚更新';
  }
  const date = new Date(input);
  if (Number.isNaN(date.getTime())) {
    return input;
  }
  return `${date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })} ${date.toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
  })}`;
}

type CollectionCreateDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSubmit: (payload: CollectionCreatePayload) => Promise<void>;
};

function CollectionCreateDialog({ open, onOpenChange, onSubmit }: CollectionCreateDialogProps) {
  const [form, setForm] = useState({
    name: '',
    displayName: '',
    description: '',
    vectorSize: '1536',
    distance: 'Cosine',
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setForm({
        name: '',
        displayName: '',
        description: '',
        vectorSize: '1536',
        distance: 'Cosine',
      });
      setError(null);
    }
  }, [open]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const normalizedName = form.name.trim();
      if (!normalizedName) {
        throw new Error('集合名称不能为空');
      }
      const dimension = Number(form.vectorSize);
      if (!Number.isFinite(dimension) || dimension <= 0) {
        throw new Error('向量维度必须为正数');
      }
      await onSubmit({
        name: normalizedName,
        displayName: (form.displayName || form.name).trim() || normalizedName,
        description: form.description.trim() || undefined,
        vectorSize: dimension,
        distance: form.distance,
      });
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : '提交失败');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>新建知识库</DialogTitle>
          <DialogDescription>创建后会在 Qdrant 中生成同名集合，可立即同步 Chunk。</DialogDescription>
        </DialogHeader>
        <form className="space-y-4" onSubmit={handleSubmit}>
          <div>
            <Label htmlFor="collection-name">集合名称</Label>
            <Input
              id="collection-name"
              placeholder="如 policy-v1"
              value={form.name}
              onChange={(event) => setForm((prev) => ({ ...prev, name: event.target.value }))}
              required
            />
            <p className="mt-1 text-xs text-gray-500">需符合 Qdrant 集合命名规范，建议使用小写加中划线。</p>
          </div>

          <div>
            <Label htmlFor="collection-display">
              展示名称 <span className="text-gray-400">(可选)</span>
            </Label>
            <Input
              id="collection-display"
              placeholder="如 财务制度知识库"
              value={form.displayName}
              onChange={(event) => setForm((prev) => ({ ...prev, displayName: event.target.value }))}
            />
          </div>

          <div>
            <Label htmlFor="collection-description">
              描述 <span className="text-gray-400">(可选)</span>
            </Label>
            <Textarea
              id="collection-description"
              placeholder="补充用途、权限说明等信息"
              value={form.description}
              onChange={(event) => setForm((prev) => ({ ...prev, description: event.target.value }))}
              rows={3}
            />
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <Label htmlFor="collection-vector">向量维度</Label>
              <Input
                id="collection-vector"
                type="number"
                min={1}
                value={form.vectorSize}
                onChange={(event) => setForm((prev) => ({ ...prev, vectorSize: event.target.value }))}
                required
              />
            </div>
            <div>
              <Label htmlFor="collection-distance">度量方式</Label>
              <select
                id="collection-distance"
                className="mt-1 h-10 w-full rounded-md border border-gray-200 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500"
                value={form.distance}
                onChange={(event) => setForm((prev) => ({ ...prev, distance: event.target.value }))}
              >
                <option value="Cosine">Cosine</option>
                <option value="Dot">Dot</option>
                <option value="Euclid">Euclid</option>
              </select>
            </div>
          </div>

          {error ? <p className="text-sm text-red-500">{error}</p> : null}

          <DialogFooter className="gap-2">
            <Button variant="outline" type="button" onClick={() => onOpenChange(false)} disabled={submitting}>
              取消
            </Button>
            <Button type="submit" disabled={submitting}>
              {submitting ? '创建中...' : '创建知识库'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

type CollectionEditDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  collection: KnowledgeBase | null;
  onSubmit: (payload: CollectionEditPayload) => Promise<void>;
};

function CollectionEditDialog({
  open,
  onOpenChange,
  collection,
  onSubmit,
}: CollectionEditDialogProps) {
  const [form, setForm] = useState({
    displayName: '',
    description: '',
    tags: '',
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open && collection) {
      setForm({
        displayName: collection.metadata.displayName,
        description: collection.metadata.description ?? '',
        tags: collection.metadata.tags?.join(', ') ?? '',
      });
    }
    if (!open) {
      setError(null);
    }
  }, [open, collection]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!collection) {
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const tags = form.tags
        .split(/[,，]/)
        .map((tag) => tag.trim())
        .filter(Boolean);
      await onSubmit({
        displayName: form.displayName.trim() || collection.metadata.displayName,
        description: form.description.trim() || undefined,
        tags: tags.length ? tags : undefined,
      });
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : '保存失败');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>编辑知识库信息</DialogTitle>
          <DialogDescription>更新展示名称、说明与标签，帮助团队快速理解内容范围。</DialogDescription>
        </DialogHeader>
        <form className="space-y-4" onSubmit={handleSubmit}>
          <div>
            <Label htmlFor="edit-display">展示名称</Label>
            <Input
              id="edit-display"
              value={form.displayName}
              onChange={(event) => setForm((prev) => ({ ...prev, displayName: event.target.value }))}
              required
            />
          </div>
          <div>
            <Label htmlFor="edit-description">描述</Label>
            <Textarea
              id="edit-description"
              rows={3}
              value={form.description}
              onChange={(event) => setForm((prev) => ({ ...prev, description: event.target.value }))}
            />
          </div>
          <div>
            <Label htmlFor="edit-tags">
              标签 <span className="text-gray-400">(以逗号分隔)</span>
            </Label>
            <Input
              id="edit-tags"
              placeholder="流程, 合同, 保密"
              value={form.tags}
              onChange={(event) => setForm((prev) => ({ ...prev, tags: event.target.value }))}
            />
          </div>
          {error ? <p className="text-sm text-red-500">{error}</p> : null}
          <DialogFooter className="gap-2">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={submitting}>
              取消
            </Button>
            <Button type="submit" disabled={submitting}>
              {submitting ? '保存中...' : '保存信息'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

type ChunkEditorDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  mode: 'create' | 'edit';
  chunk: KnowledgeChunk | null;
  onSubmit: (values: ChunkFormValues) => Promise<void>;
};

function ChunkEditorDialog({ open, onOpenChange, mode, chunk, onSubmit }: ChunkEditorDialogProps) {
  const [form, setForm] = useState({
    title: '',
    source: '',
    tags: '',
    text: '',
    metadata: '{}',
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open && chunk) {
      setForm({
        title: chunk.title ?? '',
        source: chunk.source ?? '',
        tags: chunk.tags?.join(', ') ?? '',
        text: chunk.text ?? '',
        metadata: chunk.metadata ? JSON.stringify(chunk.metadata, null, 2) : '{}',
      });
    }
    if (open && !chunk && mode === 'create') {
      setForm({
        title: '',
        source: '',
        tags: '',
        text: '',
        metadata: '{}',
      });
    }
    if (!open) {
      setError(null);
    }
  }, [open, chunk, mode]);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      let metadata: Record<string, unknown> = {};
      const trimmed = form.metadata.trim();
      if (trimmed) {
        try {
          metadata = JSON.parse(trimmed);
        } catch {
          throw new Error('Metadata 需为合法 JSON');
        }
      }
      const tags = form.tags
        .split(/[,，]/)
        .map((tag) => tag.trim())
        .filter(Boolean);
      await onSubmit({
        id: chunk?.id,
        title: form.title.trim() || undefined,
        source: form.source.trim() || undefined,
        text: form.text,
        tags,
        metadata,
      });
      onOpenChange(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : '保存失败');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>{mode === 'edit' ? '编辑 Chunk' : '新增 Chunk'}</DialogTitle>
          <DialogDescription>Chunk 会连同 payload 一起写入 Qdrant，可随时调整。</DialogDescription>
        </DialogHeader>
        <form className="space-y-4" onSubmit={handleSubmit}>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <Label htmlFor="chunk-title">标题</Label>
              <Input
                id="chunk-title"
                value={form.title}
                onChange={(event) => setForm((prev) => ({ ...prev, title: event.target.value }))}
                placeholder="如「对公开户流程」"
              />
            </div>
            <div>
              <Label htmlFor="chunk-source">来源</Label>
              <Input
                id="chunk-source"
                value={form.source}
                onChange={(event) => setForm((prev) => ({ ...prev, source: event.target.value }))}
                placeholder="文件、系统或链接"
              />
            </div>
          </div>
          <div>
            <Label htmlFor="chunk-tags">
              标签 <span className="text-gray-400">(逗号分隔)</span>
            </Label>
            <Input
              id="chunk-tags"
              value={form.tags}
              onChange={(event) => setForm((prev) => ({ ...prev, tags: event.target.value }))}
              placeholder="流程, 审批, 风控"
            />
          </div>
          <div>
            <Label htmlFor="chunk-text">Chunk 内容</Label>
            <Textarea
              id="chunk-text"
              value={form.text}
              onChange={(event) => setForm((prev) => ({ ...prev, text: event.target.value }))}
              rows={8}
              required
            />
          </div>
          <div>
            <Label htmlFor="chunk-metadata">
              Payload Metadata (JSON) <span className="text-gray-400">(可选)</span>
            </Label>
            <Textarea
              id="chunk-metadata"
              value={form.metadata}
              onChange={(event) => setForm((prev) => ({ ...prev, metadata: event.target.value }))}
              rows={4}
            />
            <p className="mt-1 text-xs text-gray-500">用于记录业务字段，提交时会校验 JSON 格式。</p>
          </div>
          {error ? <p className="text-sm text-red-500">{error}</p> : null}
          <DialogFooter className="gap-2">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={submitting}>
              取消
            </Button>
            <Button type="submit" disabled={submitting}>
              {submitting ? '保存中...' : mode === 'edit' ? '保存修改' : '创建 Chunk'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function ChartCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="bg-white rounded-xl shadow-sm p-6">
      <h2 className="text-lg font-semibold text-gray-800 mb-4">{title}</h2>
      <div className="h-80">{children}</div>
    </section>
  );
}

function StatCard({ label, value, icon: Icon, accent }: StatDefinition) {
  return (
    <div className="bg-white rounded-xl shadow-sm p-6 hover:shadow-md transition-shadow">
      <div className="flex justify-between items-start">
        <div>
          <p className="text-sm text-gray-500">{label}</p>
          <p className="text-2xl font-bold text-gray-800">{value}</p>
        </div>
        <div className={`w-12 h-12 rounded-full flex items-center justify-center ${accent.bg}`}>
          <Icon size={24} className={accent.icon} />
        </div>
      </div>
    </div>
  );
}

type FeaturePlaceholderProps = {
  icon: LucideIcon;
  title: string;
  description: string;
  actions?: FeaturePlaceholderAction[];
};

function FeaturePlaceholder({ icon: Icon, title, description, actions = [] }: FeaturePlaceholderProps) {
  return (
    <div className="min-h-[420px] bg-white rounded-xl shadow-sm flex flex-col items-center justify-center text-center p-10">
      <div className="w-20 h-20 rounded-full bg-gray-100 flex items-center justify-center mb-6">
        <Icon className="text-gray-500" size={36} />
      </div>
      <h2 className="text-xl font-semibold text-gray-800 mb-3">{title}</h2>
      <p className="text-gray-500 max-w-2xl">{description}</p>

      {actions.length > 0 ? (
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          {actions.map((action) => (
            <button
              key={action.label}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                action.variant === 'primary'
                  ? 'bg-green-500 text-white hover:bg-green-600'
                  : 'border border-gray-200 text-gray-700 hover:border-gray-300'
              }`}
            >
              {action.label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function RealtimeInsights() {
  return (
    <section className="bg-white rounded-2xl border border-emerald-100 p-8 shadow-sm">
      <div className="grid gap-8 lg:grid-cols-2">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-emerald-500">实时洞察</p>
          <h2 className="text-2xl font-semibold text-gray-900 mt-3">关键运营指标</h2>
          <p className="text-gray-500 mt-4 leading-relaxed">
            汇总培训进度、财务预警与知识库访问情况。和前方的收入趋势、账户分布形成闭环，帮助你一次性掌握全局动态。
          </p>
          <div className="mt-6 flex flex-wrap gap-3 text-sm text-gray-500">
            <span className="px-3 py-1 rounded-full bg-emerald-50 text-emerald-700">更新于 5 分钟前</span>
            <span className="px-3 py-1 rounded-full bg-gray-100">AI 异常监测开启</span>
          </div>
        </div>

        <div className="grid gap-4">
          {realtimeInsightMetrics.map((item) => (
            <div
              key={item.label}
              className="rounded-2xl border border-emerald-50 bg-gradient-to-r from-emerald-50 to-white px-6 py-4 shadow-sm"
            >
              <p className="text-sm text-gray-500">{item.label}</p>
              <div className="mt-1 flex items-end justify-between">
                <p className="text-2xl font-semibold text-gray-900">{item.value}</p>
                <span className="text-sm text-emerald-600">{item.helper}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
