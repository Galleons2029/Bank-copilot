import type { HTMLAttributes } from "react";

import { cn } from "@/lib/utils";

type BrandMarkProps = Omit<HTMLAttributes<HTMLSpanElement>, "children"> & {
  glow?: boolean;
};

export function BrandMark({ className, glow = true, ...props }: BrandMarkProps) {
  return (
    <span
      {...props}
      className={cn(
        "inline-flex items-center whitespace-nowrap bg-gradient-to-r from-emerald-500 via-cyan-500 to-blue-500 bg-clip-text text-transparent font-black tracking-wide",
        glow && "drop-shadow-[0_12px_30px_rgba(16,185,129,0.35)]",
        className,
      )}
    >
      账策云帆
    </span>
  );
}
