'use client';

import { ChangeEvent, FormEvent, useMemo, useState } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { User, Lock, ArrowRight } from 'lucide-react';
import { createClient } from '@/utils/supabase/client';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const supabase = useMemo(() => createClient(), []);
  const router = useRouter();

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);

    const sanitizedUsername = username.trim();
    if (!sanitizedUsername || !password) {
      setError('请输入用户名和密码');
      return;
    }

    setIsLoading(true);

    try {
      const { error: signInError } = await supabase.auth.signInWithPassword({
        email: sanitizedUsername,
        password,
      });

      if (signInError) {
        setError(signInError.message);
        return;
      }

      router.push('/dashboard');
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputChange =
    (setter: (value: string) => void) => (event: ChangeEvent<HTMLInputElement>) =>
      setter(event.target.value);

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col overflow-hidden">
      {/* 顶部导航栏 */}
      <header className="bg-green-700 text-white py-4 px-6 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center space-x-3">
          <div className="w-12 h-12 flex items-center justify-center">
            <Image
              src="/logo.svg"
              alt="苏州银行标志"
              width={48}
              height={48}
              priority
            />
          </div>
          <div>
            <h1 className="text-xl font-bold">苏州银行</h1>
            <p className="text-xs opacity-80">BANK OF SUZHOU</p>
          </div>
        </div>
      </header>

      {/* 主要内容区域 */}
      <main className="relative flex-1 flex items-center justify-center overflow-hidden">
        {/* 背景图片 */}
        <div className="absolute inset-0 z-0">
          <img
            src="https://p6.itc.cn/q_70/images01/20210804/6b2720cdeeda4b499ef43edcab881dbe.jpeg"
            alt="苏州银行大楼"
            className="w-full h-full object-cover"
          />
          {/* 渐变遮罩 */}
          <div className="absolute inset-0 bg-gradient-to-r from-blue-900/30 to-transparent"></div>
        </div>

        {/* 登录表单 */}
        <div className="relative z-10 w-full max-w-md p-8 bg-white rounded-lg shadow-xl backdrop-blur-sm">
          <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">账号登录</h2>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* 用户名输入框 */}
            <div className="relative">
              <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">
                <User size={20} />
              </div>
              <input
                type="text"
                placeholder="用户名"
                value={username}
                onChange={handleInputChange(setUsername)}
                className="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all duration-200"
              />
            </div>

            {/* 密码输入框 */}
            <div className="relative">
              <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">
                <Lock size={20} />
              </div>
              <input
                type="password"
                placeholder="密码"
                value={password}
                onChange={handleInputChange(setPassword)}
                className="w-full pl-12 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all duration-200"
              />
            </div>

            {/* 登录按钮 */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-green-600 hover:bg-green-700 disabled:bg-green-400 disabled:cursor-not-allowed text-white font-medium py-3 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
            >
              {isLoading ? '登录中...' : '登录'}
              <ArrowRight size={18} />
            </button>

            {error && <p className="text-sm text-red-600 text-center">{error}</p>}
          </form>

          {/* 底部链接 */}
          <div className="mt-6 flex justify-between text-sm text-gray-600">
            <Link href="/forgot-password" className="hover:text-green-600 transition-colors">
              忘记密码?
            </Link>
            <Link href="/register" className="hover:text-green-600 transition-colors">
              注册账号
            </Link>
          </div>
        </div>
      </main>

      {/* 底部版权信息 */}
      <footer className="bg-gray-800 text-white py-3 px-6 text-center text-xs flex-shrink-0">
        <p>苏州银行股份有限公司版权所有 苏ICP备10208567号-2 本网站支持IPV6</p>
      </footer>
    </div>
  );
}
