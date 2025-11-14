/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: {
      bodySizeLimit: "10mb",
    },
  },
  // 启用独立输出模式，用于 Docker 部署
  output: "standalone",
  basePath: process.env.NEXT_BASE_PATH || "",
};

export default nextConfig;
