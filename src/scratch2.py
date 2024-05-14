from sqlalchemy import create_engine, text

# Establish a connection to your SQLite database
engine = create_engine('sqlite:///mydatabase.db')

# Define the SQL for creating the various views
views_sql = {
    'v_recipe_summary_by_tag': """
    CREATE VIEW IF NOT EXISTS v_recipe_summary_by_tag AS
    SELECT json_each.value AS tag, COUNT(*) AS recipe_count
    FROM recipes, json_each(recipes.tags)
    GROUP BY tag;
    """,
    'v_top_ingredients': """
    CREATE VIEW IF NOT EXISTS v_top_ingredients AS
    SELECT json_each.value AS ingredient, COUNT(*) AS usage_count
    FROM recipes, json_each(recipes.ingredients)
    GROUP BY ingredient
    ORDER BY usage_count DESC;
    """,
    'v_recipes_by_ingredient': """
    CREATE VIEW IF NOT EXISTS v_recipes_by_ingredient AS
    SELECT json_each.value AS ingredient, json_group_array(name) AS recipes
    FROM recipes, json_each(recipes.ingredients)
    GROUP BY ingredient;
    """,
    'v_recipe_complexity': """
    CREATE VIEW IF NOT EXISTS v_recipe_complexity AS
    SELECT name, json_array_length(instructions) AS step_count
    FROM recipes;
    """,
    'v_detailed_recipes': """
    CREATE VIEW IF NOT EXISTS v_detailed_recipes AS
    SELECT id, name, url, json(ingredients) AS ingredients_formatted,
          json(instructions) AS instructions_formatted,
          json(tags) AS tags_formatted
    FROM recipes;
    """,
    'v_searchable_recipes': """
    CREATE VIEW IF NOT EXISTS v_searchable_recipes AS
    SELECT id, name,
          (name || ' ' || (SELECT group_concat(json_each.value, ' ')
                           FROM json_each(recipes.ingredients)) ||
           ' ' || (SELECT group_concat(json_each.value, ' ')
                   FROM json_each(recipes.instructions))) AS searchable_text
    FROM recipes;
    """
}

# Execute the SQL to create each view
with engine.connect() as connection:
    for view_name, sql in views_sql.items():
        # Use the text() function to ensure the SQL is treated as a literal SQL command
        connection.execute(text(sql))
        print(f"View '{view_name}' created successfully.")
