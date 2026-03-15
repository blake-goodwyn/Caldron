interface AgentIndicatorProps {
  agent: string | null
}

const AGENT_LABELS: Record<string, string> = {
  'Caldron\nPostman': 'Routing',
  'Research\nPostman': 'Researching',
  'Tavily': 'Searching web',
  'Sleuth': 'Scraping recipes',
  'ModSquad': 'Managing modifications',
  'Spinnaret': 'Tracking development',
  'KnowItAll': 'Answering question',
  'Frontman': 'Preparing response',
}

export function AgentIndicator({ agent }: AgentIndicatorProps) {
  if (!agent) return null

  const label = AGENT_LABELS[agent] || agent

  return (
    <div className="flex items-center gap-2 px-4 py-2 text-sm text-caldron-light/70">
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-caldron-accent opacity-75" />
        <span className="relative inline-flex rounded-full h-2 w-2 bg-caldron-accent" />
      </span>
      {label}...
    </div>
  )
}
