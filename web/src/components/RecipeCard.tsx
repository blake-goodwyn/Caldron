import type { Recipe } from '../types/messages'

interface RecipeCardProps {
  recipe: Recipe | null
}

export function RecipeCard({ recipe }: RecipeCardProps) {
  if (!recipe) {
    return (
      <div data-testid="recipe-empty" className="h-full flex items-center justify-center text-caldron-light/30 text-sm">
        No recipe yet — start a conversation
      </div>
    )
  }

  return (
    <div data-testid="recipe-card" className="p-4 space-y-4 overflow-y-auto chat-scroll h-full">
      <h3 data-testid="recipe-name" className="text-caldron-cream font-bold text-xl">{recipe.name}</h3>

      {/* Ingredients */}
      {recipe.ingredients?.length > 0 && (
        <div>
          <h4 className="text-caldron-light/70 text-xs uppercase tracking-wider mb-2">Ingredients</h4>
          <ul className="space-y-1">
            {recipe.ingredients.map((ing, i) => (
              <li key={i} className="text-caldron-cream text-sm flex justify-between">
                <span>{ing.name}</span>
                <span className="text-caldron-light/50">
                  {ing.quantity} {ing.unit || ''}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Instructions */}
      {recipe.instructions?.length > 0 && (
        <div>
          <h4 className="text-caldron-light/70 text-xs uppercase tracking-wider mb-2">Instructions</h4>
          <ol className="space-y-2 list-decimal list-inside">
            {recipe.instructions.map((step, i) => (
              <li key={i} className="text-caldron-cream text-sm leading-relaxed">
                {step}
              </li>
            ))}
          </ol>
        </div>
      )}

      {/* Tags */}
      {recipe.tags?.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {recipe.tags.map((tag, i) => (
            <span
              key={i}
              className="px-2 py-1 rounded-full bg-caldron-mid/20 text-caldron-light text-xs"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Sources */}
      {recipe.sources?.length > 0 && (
        <div>
          <h4 className="text-caldron-light/70 text-xs uppercase tracking-wider mb-1">Sources</h4>
          {recipe.sources.map((src, i) => (
            <a
              key={i}
              href={src.startsWith('http') ? src : undefined}
              target="_blank"
              rel="noopener noreferrer"
              className="block text-caldron-mid text-xs hover:underline truncate"
            >
              {src}
            </a>
          ))}
        </div>
      )}
    </div>
  )
}
