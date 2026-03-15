import { useCallback, useEffect } from 'react'
import {
  ReactFlow,
  useNodesState,
  useEdgesState,
  Background,
  type Node,
  type Edge,
} from '@xyflow/react'
import dagre from '@dagrejs/dagre'
import type { RecipeGraphData } from '../types/messages'

interface RecipeGraphProps {
  graph: RecipeGraphData | null
}

function layoutGraph(nodes: Node[], edges: Edge[]): Node[] {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'TB', nodesep: 50, ranksep: 80 })

  nodes.forEach((node) => {
    g.setNode(node.id, { width: 180, height: 60 })
  })
  edges.forEach((edge) => {
    g.setEdge(edge.source, edge.target)
  })

  dagre.layout(g)

  return nodes.map((node) => {
    const pos = g.node(node.id)
    return { ...node, position: { x: pos.x - 90, y: pos.y - 30 } }
  })
}

export function RecipeGraph({ graph }: RecipeGraphProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([])
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([])

  const updateGraph = useCallback(
    (data: RecipeGraphData) => {
      const newNodes: Node[] = data.nodes.map((n) => ({
        id: n.node_id,
        data: { label: n.recipe?.name || 'Recipe' },
        position: { x: 0, y: 0 },
        style: {
          background: n.node_id === data.foundational_recipe_node ? '#e63946' : '#457b9d',
          color: '#f1faee',
          border: 'none',
          borderRadius: '8px',
          padding: '8px 16px',
          fontSize: '12px',
          fontWeight: n.node_id === data.foundational_recipe_node ? 'bold' : 'normal',
        },
      }))

      const newEdges: Edge[] = data.edges.map((e, i) => ({
        id: `e-${i}`,
        source: e.source,
        target: e.target,
        style: { stroke: '#a8dadc' },
        animated: true,
      }))

      const positioned = layoutGraph(newNodes, newEdges)
      setNodes(positioned)
      setEdges(newEdges)
    },
    [setNodes, setEdges]
  )

  useEffect(() => {
    if (graph) updateGraph(graph)
  }, [graph, updateGraph])

  if (!graph || graph.nodes.length === 0) {
    return (
      <div data-testid="graph-empty" className="h-full flex items-center justify-center text-caldron-light/30 text-sm">
        Recipe graph will appear here
      </div>
    )
  }

  return (
    <div data-testid="recipe-graph" className="h-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#457b9d" gap={20} size={1} />
      </ReactFlow>
    </div>
  )
}
