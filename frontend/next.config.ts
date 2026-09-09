import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // stop next dev from regenerating AGENTS.md / CLAUDE.md in the repo
  agentRules: false,
  // the floating dev-tools button sits on top of the sidebar's sign-out link
  devIndicators: false,
};

export default nextConfig;
