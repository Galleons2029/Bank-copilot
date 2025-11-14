'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import {
  Brain,
  Database,
  MessageCircle,
  BookOpen,
  ShieldCheck,
  Layers,
  BarChart3,
  Users,
  Target,
  AlertTriangle,
  Zap,
  Play,
  Menu,
  X,
} from 'lucide-react';
import { BrandMark } from '@/components/brand-mark';

type LogoIconProps = {
  size?: number;
  className?: string;
};

const LogoIcon = ({ size = 32, className }: LogoIconProps) => (
  <Image
    src="/logo.svg"
    alt="账策云帆 Logo"
    width={size}
    height={size}
    className={`object-contain ${className ?? ''}`.trim()}
  />
);

const navLinks = [
  { label: '产品概览', href: '#overview' },
  { label: '方案蓝图', href: '#solutions' },
  { label: '能力矩阵', href: '#capabilities' },
  { label: '场景落地', href: '#scenarios' },
  { label: '安全合规', href: '#security' },
];

export default function HomePage() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [activeTrack, setActiveTrack] = useState(0);

  const summaryStats = [
    { label: '分户余额数据', value: '2 亿条' },
    { label: '会计分录', value: '1 亿条' },
    { label: '交易明细', value: '700 万条' },
    { label: '总分核对结果', value: '60 万条' },
  ];

  const heroHighlights = [
    { label: '部署周期', value: '< 2 周' },
    { label: '培训周期缩短', value: '40%+' },
    { label: '回答准确率', value: '95%' },
  ];

  const challenges = [
    {
      title: '非财会新人入门慢',
      description: '科目、借贷方向等概念分散在大量手册中，依赖资深导师口口相传。',
      metric: '平均熟练周期 3-6 周',
      icon: Users,
    },
    {
      title: '分录穿透跨系统',
      description: '历史/当日检索分属不同系统，分录到账号交易的反向穿透效率低。',
      metric: '跨 4+ 业务系统',
      icon: Layers,
    },
    {
      title: '总分不平定位慢',
      description: '当核对出现异常，需要逐笔排查多维数据，定位耗时且难追溯。',
      metric: '定位耗时 > 2 小时',
      icon: AlertTriangle,
    },
  ];

  const solutionTracks = [
    {
      title: '知识获取与构建',
      subtitle: 'Knowledge Fabric',
      description:
        '连接 Word / Excel / PDF / 图像等异构文档，自动抽取并构建高可信账务知识库。',
      icon: BookOpen,
      bullets: [
        'OCR+结构化解析，统一口径与版本',
        '重复/缺失/过时内容质检，自动回溯源文档',
        '热更新策略，知识随监管政策动态迭代',
      ],
    },
    {
      title: '业务意图记忆引擎',
      subtitle: 'Memory Graph',
      description:
        '面向 20+ 轮长对话的记忆强化网络，追踪意图、维度与上下文，过滤无关闲聊。',
      icon: MessageCircle,
      bullets: [
        '短/长记忆双缓存，按会话角色加权',
        '意图轨迹追踪，可视化对话节点',
        '业务指标驱动的噪声过滤与回溯',
      ],
    },
    {
      title: '推理与决策中枢',
      subtitle: 'Reasoning & Reconcile',
      description:
        '结合多智能体和检索增强推理（RAG+CoT），快速定位总分不平原因并生成结论。',
      icon: Brain,
      bullets: [
        '多维检索：账号 / 机构 / 币种 / 金额',
        '推理链路可追溯，结论可复核',
        '异常原因建议直接回写至核对看板',
      ],
    },
  ];

  const capabilityMatrix = [
    {
      title: '知识导师 · Knowledge Mentor',
      description: 'AI 教师以对话方式讲解行内账务体系，为新人生成个性化学习路径与测评。',
      metrics: [
        { label: '术语覆盖', value: '12,000+' },
        { label: '答复准确率', value: '95%' },
      ],
      icon: BookOpen,
    },
    {
      title: '多层检索 · Ledger Graph',
      description: '从分录到交易明细的多跳检索，支持历史/当日、系统/账号的灵活切换。',
      metrics: [
        { label: '穿透维度', value: '5 层' },
        { label: '查询时延', value: '< 1s' },
      ],
      icon: Database,
    },
    {
      title: '异常定位 · Reconcile Copilot',
      description: '基于推理与决策技术，对总分不平科目追溯日期、机构、币种与金额，输出原因建议。',
      metrics: [
        { label: '定位时间', value: '-70%' },
        { label: '异常归因', value: '多场景' },
      ],
      icon: Brain,
    },
  ];

  const roadmapPhases = [
    {
      title: '数据接入与治理',
      timeframe: 'Week 0-1',
      detail: '多源文档、系统表与日志接入；敏感字段脱敏；知识刻面设计。',
    },
    {
      title: '知识 + 记忆编排',
      timeframe: 'Week 1-2',
      detail: '构建多模态知识库与记忆网络，建立意图识别与维度抽取模型。',
    },
    {
      title: '决策与核对工作流',
      timeframe: 'Week 2-3',
      detail: '多智能体协同完成检索、推理、回填，打通核对与培训看板。',
    },
    {
      title: '持续运营与评估',
      timeframe: 'Week 3+',
      detail: '上线指标、人工审核闭环与版本管理，支持监管稽核。',
    },
  ];

  const useCases = [
    {
      title: '交互式账务术语问答',
      description: '把复杂财务手册变成可对话的知识服务，帮助新人在对话中快速掌握概念。',
      icon: BookOpen,
    },
    {
      title: '多维度账务检索',
      description: '按账号 / 机构 / 科目 / 时间等维度组合查询，跨系统结果统一呈现。',
      icon: Database,
    },
    {
      title: '总分不平原因定位',
      description: '针对异常科目自动回溯交易链路，输出可能原因及建议处理动作。',
      icon: BarChart3,
    },
    {
      title: '培训与测评自动化',
      description: '根据学习进度生成测试题与案例，实时记录培训成绩与风险提示。',
      icon: Target,
    },
  ];

  const securityPillars = [
    {
      title: '数据隔离与审计',
      description: '私有化部署 + 细粒度权限，所有对话与检索可追溯，可输出稽核报告。',
      icon: ShieldCheck,
    },
    {
      title: '合规策略内嵌',
      description: '遵循银行内控与监管要求，提示敏感信息访问并内置审批流。',
      icon: Layers,
    },
    {
      title: '可观测与可控',
      description: '运营看板实时监控延迟、准确率与异常，支持自动或人工干预。',
      icon: BarChart3,
    },
  ];

  useEffect(() => {
    const timer = setInterval(() => {
      setActiveTrack((prev) => (prev + 1) % solutionTracks.length);
    }, 8000);
    return () => clearInterval(timer);
  }, [solutionTracks.length]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50">
      <nav className="sticky top-0 z-50 border-b border-white/5 bg-slate-950/80 backdrop-blur">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex h-16 items-center justify-between">
            <Link href="/" className="flex items-center gap-3">
              <div className="rounded-2xl border border-emerald-400/40 bg-emerald-500/10 p-2">
                <LogoIcon size={28} />
              </div>
              <BrandMark className="text-xl font-semibold text-white" />
            </Link>

            <div className="hidden md:flex items-center gap-8 text-sm font-medium">
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="text-slate-300 hover:text-white transition-colors"
                >
                  {link.label}
                </Link>
              ))}
            </div>

            <div className="hidden md:flex items-center gap-3">
              <Link
                href="/login"
                className="rounded-xl border border-emerald-400/50 px-4 py-2 text-sm font-semibold text-emerald-200 hover:bg-emerald-400/10"
              >
                登录控制台
              </Link>
              <Link
                href="/login"
                className="rounded-xl bg-emerald-400 px-4 py-2 text-sm font-semibold text-slate-900 hover:bg-emerald-300"
              >
                预约演示
              </Link>
            </div>

            <button
              className="md:hidden text-slate-200"
              onClick={() => setIsMenuOpen((prev) => !prev)}
            >
              {isMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>

        {isMenuOpen && (
          <div className="md:hidden border-t border-white/10 bg-slate-900">
            <div className="space-y-2 px-4 py-4">
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setIsMenuOpen(false)}
                  className="block rounded-lg px-3 py-2 text-slate-200 hover:bg-white/5"
                >
                  {link.label}
                </Link>
              ))}
              <Link
                href="/login"
                onClick={() => setIsMenuOpen(false)}
                className="block rounded-lg bg-emerald-400 px-3 py-2 text-center text-slate-900 font-semibold"
              >
                立即体验
              </Link>
            </div>
          </div>
        )}
      </nav>

      <main>
        <section className="relative overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
          <div className="absolute inset-0 opacity-40" aria-hidden>
            <div className="absolute inset-y-0 right-0 w-2/3 bg-[radial-gradient(circle_at_top,_#10b98133,_transparent_55%)]" />
            <div className="absolute -left-32 top-10 h-64 w-64 rounded-full bg-emerald-500/20 blur-3xl" />
          </div>
          <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
            <div className="grid gap-12 lg:grid-cols-2 items-center">
              <div>
                <p className="text-sm uppercase tracking-[0.3em] text-emerald-300/80">
                  LLM 财务智能体 · Bank-Copilot
                </p>
                <h1 className="mt-6 text-4xl md:text-5xl lg:text-6xl font-semibold text-white leading-tight">
                  <span className="text-white">账策云帆</span>
                  <span className="text-emerald-300 block mt-2">面向银行账务的企业级智能助手</span>
                </h1>
                <p className="mt-6 text-lg text-slate-200 leading-relaxed">
                  基于大语言模型与银行账务数据特征训练，打造知识问答、账务检索与异常定位的一体化智能体，帮助培训部、会计管理、IT 运维多角色协同提效。
                </p>
                <div className="mt-8 flex flex-col sm:flex-row gap-4">
                  <Link
                    href="/login"
                    className="inline-flex items-center justify-center rounded-2xl bg-emerald-400 px-8 py-4 text-lg font-semibold text-slate-900 shadow-[0_20px_45px_rgba(16,185,129,0.35)] hover:-translate-y-0.5 transition-transform"
                  >
                    预约私享演示
                  </Link>
                  <button className="inline-flex items-center justify-center rounded-2xl border border-white/30 px-8 py-4 text-lg font-semibold text-white hover:bg-white/5">
                    <Play className="w-5 h-5 mr-2" />
                    查看解决方案 Deck
                  </button>
                </div>
                <div className="mt-10 grid gap-6 sm:grid-cols-3">
                  {heroHighlights.map((item) => (
                    <div key={item.label} className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur">
                      <div className="text-sm text-slate-300">{item.label}</div>
                      <div className="mt-2 text-2xl font-semibold text-white">{item.value}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="relative">
                <div className="rounded-3xl border border-white/10 bg-white/5 p-8 backdrop-blur">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs uppercase tracking-[0.3em] text-emerald-200">
                        Data Footprint
                      </p>
                      <h3 className="text-2xl font-semibold text-white mt-2">账务数据底座</h3>
                    </div>
                    <div className="rounded-2xl bg-emerald-400/20 p-3">
                      <Database className="w-6 h-6 text-emerald-200" />
                    </div>
                  </div>
                  <div className="mt-6 grid gap-4 sm:grid-cols-2">
                    {summaryStats.map((stat) => (
                      <div key={stat.label} className="rounded-2xl bg-white/5 p-4 border border-white/10">
                        <div className="text-sm text-slate-300">{stat.label}</div>
                        <div className="mt-1 text-xl font-semibold text-white">{stat.value}</div>
                      </div>
                    ))}
                  </div>
                  <div className="mt-6 rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-4 text-emerald-100">
                    <p className="text-sm">
                      支持对 2025.06.01-2025.06.10 时间区间内 3 大账务数据域的深度分析，并持续扩展到全量历史。根据角色定制指标看板，保证上线即有价值。
                    </p>
                  </div>
                </div>
                <div className="absolute -right-8 -bottom-10 hidden lg:block">
                  <div className="rounded-3xl border border-white/10 bg-slate-900/70 px-6 py-4 shadow-2xl backdrop-blur">
                    <div className="flex items-center gap-3 text-sm text-slate-200">
                      <Zap className="w-4 h-4 text-emerald-300" />
                      多智能体协同 · 记忆强化 · 合规审计
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="overview" className="bg-white text-slate-900 py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="max-w-3xl">
              <p className="text-sm font-semibold text-emerald-600">赛题背景 · 业务挑战</p>
              <h2 className="mt-4 text-3xl md:text-4xl font-bold">
                对齐银行账务场景的三大关键痛点
              </h2>
              <p className="mt-4 text-lg text-slate-600">
                账策云帆针对苏州银行等金融机构的账务检索、培训、核对场景打造，帮助新人快速入门，资深会计高效检索，核对团队秒速定位异常。
              </p>
            </div>
            <div className="mt-12 grid gap-6 md:grid-cols-3">
              {challenges.map((challenge) => {
                const Icon = challenge.icon;
                return (
                  <div key={challenge.title} className="rounded-2xl border border-slate-200 bg-slate-50 p-6 shadow-sm">
                    <div className="flex items-center justify-between">
                      <div className="rounded-xl bg-emerald-500/15 p-3 text-emerald-600">
                        <Icon className="w-6 h-6" />
                      </div>
                      <span className="text-xs font-medium text-slate-500">{challenge.metric}</span>
                    </div>
                    <h3 className="mt-4 text-xl font-semibold text-slate-900">{challenge.title}</h3>
                    <p className="mt-3 text-slate-600">{challenge.description}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        <section id="solutions" className="bg-slate-900 py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-6">
              <div>
                <p className="text-sm font-semibold text-emerald-300">实施策略 · 方案蓝图</p>
                <h2 className="mt-4 text-3xl md:text-4xl font-bold text-white">
                  三层架构串联知识、记忆与推理
                </h2>
                <p className="mt-4 text-lg text-slate-300 max-w-3xl">
                  通过知识构建、意图记忆以及推理决策三大模块，形成面向银行账务场景的闭环；每一层均可独立部署，也可按需按模块扩展。
                </p>
              </div>
              <div className="text-sm text-slate-400">自动轮播展示 · 鼠标悬停可固定卡片</div>
            </div>
            <div className="mt-10 grid gap-4 lg:grid-cols-3">
              {solutionTracks.map((track, index) => {
                const Icon = track.icon;
                const isActive = index === activeTrack;
                return (
                  <button
                    type="button"
                    key={track.title}
                    onMouseEnter={() => setActiveTrack(index)}
                    onFocus={() => setActiveTrack(index)}
                    className={`text-left rounded-2xl border p-6 transition-all ${
                      isActive
                        ? 'border-emerald-400/60 bg-emerald-400/10 shadow-[0_20px_45px_rgba(16,185,129,0.15)]'
                        : 'border-white/10 bg-white/5 hover:border-emerald-400/40'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`rounded-2xl p-3 ${
                        isActive ? 'bg-emerald-400 text-slate-900' : 'bg-white/10 text-emerald-200'
                      }`}>
                        <Icon className="w-6 h-6" />
                      </div>
                      <div>
                        <p className="text-xs uppercase tracking-[0.3em] text-slate-400">
                          {track.subtitle}
                        </p>
                        <h3 className="text-xl font-semibold text-white">{track.title}</h3>
                      </div>
                    </div>
                    <p className="mt-4 text-slate-300">{track.description}</p>
                  </button>
                );
              })}
            </div>
            <div className="mt-10 rounded-3xl border border-white/10 bg-white/5 p-8">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <p className="text-sm font-semibold text-emerald-300">模块亮点</p>
                  <h3 className="text-2xl font-semibold text-white">{solutionTracks[activeTrack].title}</h3>
                  <p className="text-slate-300 mt-2">{solutionTracks[activeTrack].description}</p>
                </div>
                <div className="flex gap-3 text-xs text-slate-400">
                  <span>RAG</span>
                  <span>记忆网络</span>
                  <span>可追溯链路</span>
                </div>
              </div>
              <ul className="mt-6 grid gap-4 sm:grid-cols-3">
                {solutionTracks[activeTrack].bullets.map((bullet) => (
                  <li key={bullet} className="rounded-2xl border border-white/10 bg-slate-900/40 p-4 text-sm text-slate-200">
                    {bullet}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

        <section id="capabilities" className="bg-slate-50 py-20 text-slate-900">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
              <div>
                <p className="text-sm font-semibold text-emerald-600">能力矩阵</p>
                <h2 className="mt-4 text-3xl md:text-4xl font-bold text-slate-900">从知识到决策的闭环产品力</h2>
                <p className="mt-4 text-lg text-slate-600 max-w-3xl">
                  针对培训、检索、核对三大场景，形成可配置的能力组件，并沉淀指标体系与治理工具，方便与行内现有系统对接。
                </p>
              </div>
              <div className="rounded-2xl border border-emerald-200 bg-white/70 p-5 text-sm text-slate-600">
                <div className="font-semibold text-slate-900">金融级可扩展</div>
                <p className="mt-1">支持 API / SDK / 私有化部署，兼容主流国产化环境。</p>
              </div>
            </div>
            <div className="mt-12 grid gap-6 md:grid-cols-3">
              {capabilityMatrix.map((capability) => {
                const Icon = capability.icon;
                return (
                  <div key={capability.title} className="rounded-2xl border border-slate-200 bg-white p-8 shadow-sm">
                    <div className="flex items-center gap-3">
                      <div className="rounded-2xl bg-emerald-500/15 p-3 text-emerald-600">
                        <Icon className="w-6 h-6" />
                      </div>
                      <h3 className="text-xl font-semibold text-slate-900">{capability.title}</h3>
                    </div>
                    <p className="mt-4 text-slate-600">{capability.description}</p>
                    <div className="mt-6 grid gap-3">
                      {capability.metrics.map((metric) => (
                        <div key={metric.label} className="rounded-xl bg-slate-50 p-3 text-sm">
                          <div className="text-slate-500">{metric.label}</div>
                          <div className="text-lg font-semibold text-slate-900">{metric.value}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        <section className="bg-slate-900 py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="max-w-3xl">
              <p className="text-sm font-semibold text-emerald-300">实施时间线</p>
              <h2 className="mt-4 text-3xl md:text-4xl font-bold text-white">4 周即刻上线 · 快速试点再规模复制</h2>
              <p className="mt-4 text-lg text-slate-300">
                以精益画布为指引，从数据治理到智能体上线再到运营评估，确保每一步可量化、可交付。
              </p>
            </div>
            <div className="mt-12 grid gap-8 lg:grid-cols-4">
              {roadmapPhases.map((phase, index) => (
                <div key={phase.title} className="rounded-2xl border border-white/10 bg-white/5 p-6">
                  <div className="flex items-center justify-between text-slate-400 text-sm">
                    <span>{phase.timeframe}</span>
                    <span>阶段 {index + 1}</span>
                  </div>
                  <h3 className="mt-4 text-xl font-semibold text-white">{phase.title}</h3>
                  <p className="mt-3 text-slate-300 text-sm">{phase.detail}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="scenarios" className="bg-white py-20 text-slate-900">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="max-w-3xl">
              <p className="text-sm font-semibold text-emerald-600">落地场景</p>
              <h2 className="mt-4 text-3xl md:text-4xl font-bold">连接培训部、财务部与风控部的统一工作台</h2>
              <p className="mt-4 text-lg text-slate-600">
                针对行内三个高频用例打造可复制流程，并支持在现有 OA / CRM / 内部知识门户内嵌部署。
              </p>
            </div>
            <div className="mt-12 grid gap-6 md:grid-cols-2">
              {useCases.map((useCase) => {
                const Icon = useCase.icon;
                return (
                  <div key={useCase.title} className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
                    <div className="flex items-center gap-3">
                      <div className="rounded-2xl bg-emerald-500/15 p-3 text-emerald-600">
                        <Icon className="w-6 h-6" />
                      </div>
                      <h3 className="text-xl font-semibold text-slate-900">{useCase.title}</h3>
                    </div>
                    <p className="mt-4 text-slate-600">{useCase.description}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        <section id="security" className="bg-slate-900 py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-8">
              <div className="max-w-2xl">
                <p className="text-sm font-semibold text-emerald-300">安全 · 合规 · 可运营</p>
                <h2 className="mt-4 text-3xl md:text-4xl font-bold text-white">
                  满足金融级合规的技术与运营基线
                </h2>
                <p className="mt-4 text-lg text-slate-300">
                  从模型到数据、从对话到决策，每一步都有可追溯证据链，既满足监管要求，也方便内部审计。
                </p>
              </div>
              <div className="rounded-3xl border border-emerald-400/30 bg-white/5 p-6 text-sm text-emerald-100">
                <p>支持国产化 GPU / 数据中心部署，提供离线安装包与持续升级服务。</p>
              </div>
            </div>
            <div className="mt-12 grid gap-6 md:grid-cols-3">
              {securityPillars.map((pillar) => {
                const Icon = pillar.icon;
                return (
                  <div key={pillar.title} className="rounded-2xl border border-white/10 bg-white/5 p-6">
                    <div className="rounded-2xl bg-emerald-400/15 p-3 text-emerald-200 w-fit">
                      <Icon className="w-6 h-6" />
                    </div>
                    <h3 className="mt-4 text-xl font-semibold text-white">{pillar.title}</h3>
                    <p className="mt-3 text-slate-300 text-sm">{pillar.description}</p>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        <section className="bg-gradient-to-r from-emerald-400 via-emerald-500 to-teal-500 py-20">
          <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-slate-900">
            <p className="text-sm uppercase tracking-[0.3em] text-slate-900/70">Next Step</p>
            <h2 className="mt-4 text-3xl md:text-4xl font-bold">
              让账策云帆成为您账务数字化的第二中枢
            </h2>
            <p className="mt-4 text-lg text-slate-900/80">
              预约 60 分钟定制演示，获取精益画布、指标模板与 PoC 计划，共同验证银行级智能体的价值闭环。
            </p>
            <div className="mt-10 flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href="/login"
                className="rounded-2xl bg-slate-900 px-8 py-4 text-lg font-semibold text-white shadow-lg hover:bg-slate-800"
              >
                安排演示
              </Link>
              <Link
                href="/login"
                className="rounded-2xl border border-slate-900 px-8 py-4 text-lg font-semibold text-slate-900"
              >
                获取技术白皮书
              </Link>
            </div>
          </div>
        </section>
      </main>

      <footer className="bg-slate-950 border-t border-white/5 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid gap-8 md:grid-cols-4">
            <div>
              <div className="flex items-center gap-3">
                <div className="rounded-2xl border border-emerald-400/40 bg-emerald-500/10 p-2">
                  <LogoIcon size={28} />
                </div>
                <BrandMark className="text-xl text-white" />
              </div>
              <p className="mt-4 text-sm text-slate-400">
                账策云帆 Bank-Copilot · 基于 LLM 的账务智能体，为银行提供培训、检索、核对的新范式。
              </p>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white uppercase tracking-wide">产品</h4>
              <ul className="mt-4 space-y-2 text-sm text-slate-400">
                <li><Link href="#overview" className="hover:text-white">产品概览</Link></li>
                <li><Link href="#solutions" className="hover:text-white">方案蓝图</Link></li>
                <li><Link href="#capabilities" className="hover:text-white">能力矩阵</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white uppercase tracking-wide">资源</h4>
              <ul className="mt-4 space-y-2 text-sm text-slate-400">
                <li><Link href="/login" className="hover:text-white">解决方案 Deck</Link></li>
                <li><Link href="/login" className="hover:text-white">精益画布模板</Link></li>
                <li><Link href="/login" className="hover:text-white">PoC 清单</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white uppercase tracking-wide">联系</h4>
              <ul className="mt-4 space-y-2 text-sm text-slate-400">
                <li>商务合作：copilot@bank.com</li>
                <li>售后支持：support@bank.com</li>
                <li>苏州 · 上海 · 北京</li>
              </ul>
            </div>
          </div>
          <div className="mt-12 border-t border-white/5 pt-6 text-center text-sm text-slate-500">
            <p>
              &copy; 2025 <BrandMark className="mx-1 text-base" glow={false} />. All Rights Reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
