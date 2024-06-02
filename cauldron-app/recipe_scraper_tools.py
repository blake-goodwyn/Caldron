from recipe_scrapers import scrape_me


@tool
def get_recipe_ingredients(url: str) -> dict:
    scraper = scrape_me(url, wild_mode=True)
    return scraper.ingredients()

@tool
def get_recipe_instructions(url: str) -> dict:
    scraper = scrape_me(url, wild_mode=True)
    return scraper.instructions_list()

@tool
def get_recipe_name(url: str) -> str:
    scraper = scrape_me(url, wild_mode=True)
    return scraper.title()

@tool
def get_recipe_keywords(url: str) -> str:
    scraper = scrape_me(url, wild_mode=True)
    return scraper.keywords()