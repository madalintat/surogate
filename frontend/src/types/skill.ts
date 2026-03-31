export interface Skill {
  id: string;
  name: string;
  displayName: string;
  description: string;
  content: string;
  license: string;
  compatibility: string;
  metadata?: Record<string, string>;
  allowedTools: string[];
  status: string;
  author: string;
  updatedAt: string;
  tags: string[];
  hubRef?: string | null;
}

export interface SchemaField {
  name: string;
  type: string;
  required?: boolean;
  description: string;
}

export interface ToolConfig {
  key: string;
  value: string;
  secret: boolean;
}

export interface AgentRef {
  name: string;
  status: string;
}

export interface Tool {
  id: string;
  name: string;
  displayName: string;
  description: string;
  category: string;
  version: string;
  status: string;
  author: string;
  tags: string[];
  usedByAgents: AgentRef[];
  inputSchema: SchemaField[];
  outputSchema: SchemaField[];
  config: ToolConfig[];
  metrics: { calls24h: number; avgLatency: string; errorRate: string; p99: string };
}

export interface ToolCategory {
  id: string;
  label: string;
  icon: string;
}

export interface McpServer {
  id: string;
  name: string;
  url: string;
  transport: string;
  status: string;
  latency: string;
  description: string;
  toolCount: number;
  connectedAgents: string[];
  tools: string[];
  auth: string;
  updatedAt: string;
}
