import pprint
from class_defs import RecipeGraph, Recipe, Ingredient, RecipeModification, ModsList
import textwrap

class CustomPP(pprint.PrettyPrinter):
    def format(self, obj):
        if isinstance(obj, Ingredient):
            return (f"Ingredient(name={obj.name}, quantity={obj.quantity}, unit={obj.unit})", True, False)
        elif isinstance(obj, RecipeModification):
            return (f"RecipeModification(id={obj._id}, priority={obj.priority}, "
                    f"add_ingredient={obj.add_ingredient}, remove_ingredient={obj.remove_ingredient}, "
                    f"update_ingredient={obj.update_ingredient}, add_instruction={obj.add_instruction}, "
                    f"remove_instruction={obj.remove_instruction}, add_tag={obj.add_tag}, remove_tag={obj.remove_tag})", True, False)
        elif isinstance(obj, Recipe):
            ingredients_str = ", \n".join([f"{ing.name}, {ing.quantity} {ing.unit}" for ing in obj.ingredients])
            instructions_str = "\n".join(obj.instructions)
            tags_str = ", ".join(obj.tags) if obj.tags else "None"
            sources_str = ''.join(['Sources: ' ,", ".join(obj.sources)]) if obj.sources else ""
            return (''.join([f"Recipe:\n{obj.name}\n\n",
                    f"Ingredients:\n{ingredients_str}\n\n"
                    f"Instructions:\n{instructions_str}\n\n"
                    f"{sources_str}"]), True, False)
        elif isinstance(obj, RecipeGraph):
            return (f"RecipeGraph(size={obj.get_graph_size()}, foundational_recipe_node={obj.foundational_recipe_node})", True, False)
        elif isinstance(obj, ModsList):
            mods_list_str = "\n".join([f"{mod._id}: Priority {mod.priority}" for mod in obj.get_mods_list()])
            return (f"ModsList:\n{mods_list_str}", True, False)
        elif isinstance(obj, str):
            return (textwrap.fill(obj, width=32), True, False)
        return super().pformat(obj)
    
printer = CustomPP(width=32)
