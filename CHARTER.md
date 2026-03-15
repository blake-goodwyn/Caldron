# Caldron — Application Charter

> A living document capturing the intent, principles, and trajectory of the Caldron project.

---

## 1. Mission

Caldron is an AI-assisted recipe development platform. It exists to close the gap between _having an idea for a dish_ and _holding a tested, refined recipe_ — by making that journey conversational, iterative, and informed.

The core premise: recipe development is not a single creative act. It's a feedback loop. You research, draft, cook, taste, adjust, and repeat. Caldron encodes that loop into a multi-agent system that can research on your behalf, track every iteration you've explored, manage a queue of proposed changes, and meet you wherever you are in the process — whether that's "I want a spicy miso ramen" or "swap the kombu for shiitake dashi and bump the chili oil."

The platform is designed for depth over breadth. It doesn't aim to serve a million users a million recipes. It aims to help one cook develop one recipe _really well_.

---

## 2. Design Philosophy & Guiding Principles

### Conversational-first
Interaction should feel like working with a knowledgeable sous chef, not filling out a form. The system should understand natural language intent, ask clarifying questions when it's unsure, and present results in a way that invites continued dialogue.

### Iterative by nature
Recipes evolve. Caldron tracks that evolution in a versioned graph — not as a single document that gets overwritten. Every iteration is preserved, every branch is navigable, and you can always go back.

### Human-in-the-loop
AI proposes; humans refine. The system generates, suggests, and researches, but the cook makes the decisions. Sensory feedback — taste, texture, aroma — is information that only a human (or a physical sensor) can provide. Caldron is built to receive that input and act on it.

### Modular agents, clear responsibilities
Each agent in the system has a well-defined role. The architecture scales by adding new specialists (flavor analysis, nutrition, cost estimation) rather than making existing agents more complex. An agent should do one thing and do it well.

### State transparency
The user should always be able to ask: _What's my current recipe? What modifications are pending? What have I tried before?_ The recipe graph, modification queue, and staging area are first-class concepts, not hidden internals.

### Hardware-aware
Caldron is designed to eventually extend beyond the screen — integrating physical sensors for real-time feedback and thermal printers for recipe card output. The architecture should not assume that all interaction happens through a keyboard.

---

## 3. Non-Goals

- **Not a recipe database.** Caldron develops recipes; it doesn't catalog them at scale. It may search the web for inspiration, but it's not a replacement for Allrecipes or Serious Eats.
- **Not a social platform.** There are no user profiles, feeds, or sharing mechanics in scope. Collaboration may come later, but virality is never a goal.
- **Not a meal planner or grocery app.** Caldron cares about the recipe itself — its structure, flavor, and evolution — not what's in your fridge or what you're eating on Tuesday.
- **Not a replacement for culinary expertise.** It's a tool that augments a cook's process. It won't tell you your soufflé fell because you opened the oven door. It can help you reformulate the recipe so it's more forgiving next time.

---

## 4. User Personas

### The Home Cook Explorer
Curious, adventurous, and idea-rich but time-poor. Wants to say "I'm craving a Thai-inspired barbecue sauce" and get a working recipe they can riff on. Values speed of iteration and the ability to say "more heat, less sweet" without starting over.

### The Culinary Student
Learning technique and flavor theory. Wants to understand _why_ a modification works — not just _what_ to change. Benefits from the recipe graph's history: seeing how a dish evolved from version to version builds intuition.

### The Recipe Developer
Needs structured iteration, version tracking, and eventually export. Might be developing recipes for a blog, cookbook, or product line. Cares about precision (weights, temperatures, timing) and the ability to branch and compare approaches.

### The Hardware Tinkerer
Interested in the sensor bowl, thermal printer, and other physical integrations. Wants to feed real-world data (capacitance readings, temperature logs) back into the development loop. Sees Caldron as a platform for culinary IoT experimentation.

---

## 5. Roadmap

### Phase 0 — Foundation ✓
_Status: Complete_

Multi-agent backend with LangChain/LangGraph. Nine agents in a supervisor pattern handling research, recipe generation, modification management, version tracking, and user interaction. CLI interaction loop. JSON-based state persistence. 68 tests across unit and integration suites.

### Phase 1 — Test-Driven Hardening
_Status: Next_

Establish comprehensive TDD discipline before expanding surface area. This phase ensures the foundation is trustworthy before building on top of it.
- Expand unit test coverage across all agent tools and data models to a measurable baseline
- Add contract tests for agent routing (supervisor decisions are deterministic given inputs)
- Integration tests for full conversation flows (prompt → agent chain → state mutation → output)
- Set up CI pipeline with coverage gates — no merge without passing tests
- Formalize test fixtures for recipe state (graph, mods queue, pot) to support reproducible scenarios

### Phase 2 — Conversational Web Front-End
Expose the agent system through an API layer and build a chat-based web interface. The front-end should support:
- Natural language conversation with the agent system
- Recipe display with structured formatting (ingredients, instructions, tags)
- Visual representation of the recipe graph (version history)
- Modification queue inspection and management
- Real-time streaming of agent responses

### Phase 3 — Performance Hardening
Profile and optimize the system under realistic workloads before scaling features.
- Benchmark agent chain latency end-to-end (prompt to final response)
- Identify and reduce unnecessary LLM round-trips in the supervisor routing
- Optimize state serialization/deserialization (JSON → database transition lives here)
- Load-test the API layer for concurrent sessions
- Establish performance baselines and regression tests
- Introduce user sessions and persistent storage so recipe state survives across conversations

### Phase 4 — Hardware Integration
Connect the sensor bowl (capacitive touch input) and thermal printer (recipe card output) to the web interface. Define a protocol for sensor data ingestion and a rendering pipeline for printed output.

### Phase 5 — Extended Agent Roster
Activate planned specialist agents:
- **Bookworm** — SQL-backed nutritional database queries
- **Remy** — Flavor profile analysis and pairing suggestions
- **HealthNut** — Nutritional analysis and dietary constraint checking
- **MrKrabs** — Cost estimation and ingredient sourcing
- **Critic** — Aggregated feedback from taste tests and sensory input
- **Glutton** — Food validation (ensuring requests stay in the culinary domain)

### Phase 6 — Voice & Multimodal Input
Voice interface for hands-free interaction while cooking. Image input for identifying ingredients or plated dishes. Foundation already laid with voice dependencies and audio processing modules.

### Phase 7 — Culinary ML Research
A dedicated research and development phase exploring machine learning applications for understanding recipe dynamics across multiple dimensions.

**Research inputs:**
- Culinary science texts — _The Flavor Matrix_ (James Briscione), _The Flavor Bible_, _On Food and Cooking_ (Harold McGee)
- Existing flavor compound databases and food pairing datasets
- Recipe corpus analysis (structure, ingredient co-occurrence, technique patterns)

**ML exploration areas:**
- **Flavor affinity modeling** — learn ingredient compatibility from compound overlap and pairing data; move beyond lookup tables to predictive pairing suggestions
- **Technique-outcome prediction** — model how preparation methods (roast vs. braise, emulsify vs. reduce) affect flavor, texture, and structure
- **Substitution intelligence** — given an ingredient and a recipe context, suggest substitutions that preserve the role (acid, fat, umami, binding) rather than just matching category
- **Recipe coherence scoring** — evaluate whether a recipe "makes sense" as a whole — balanced flavor profile, compatible techniques, reasonable proportions
- **Development trajectory learning** — analyze the recipe graph across many users/sessions to identify common refinement patterns (e.g., "most users who start with X end up adjusting Y")

**Outcome:** trained models or embeddings that feed back into the agent system — making Remy, HealthNut, and Critic meaningfully smarter than prompt-engineered wrappers around a general-purpose LLM.

---

## 6. Success Metrics

These are qualitative benchmarks, not KPIs. Caldron succeeds when:

- **A single session gets you somewhere.** A user can go from a vague idea ("something with miso and pork belly") to a concrete, refined recipe in one conversational session — without leaving the interface to Google things.
- **The history tells a story.** The recipe graph is legible and navigable. You can look at the development path and understand why each change was made and what it produced.
- **Ambiguity is handled gracefully.** When the system doesn't understand, it asks — it doesn't guess badly. When it makes assumptions, it says so.
- **Adding agents doesn't break things.** A new specialist (e.g., Remy for flavor analysis) can be wired into the supervisor graph without restructuring existing agents or tools.
- **The front-end disappears.** The interface is invisible in the best sense — the user thinks about their recipe, not about the tool. Interaction feels natural, not procedural.
- **The system knows something about food.** Agent suggestions are grounded in learned flavor dynamics and culinary science — not just LLM pattern-matching on recipe text. Substitutions preserve the _role_ of an ingredient, not just its category.

---

## 7. Technical Boundaries

| Dimension | Current (Phase 0) | Target |
|-----------|-------------------|--------|
| **LLM** | OpenAI (gpt-3.5-turbo) | Swappable — architecture should not be OpenAI-specific |
| **Agent framework** | LangChain 0.2 / LangGraph 0.1 | Upgrade as needed; agent patterns are more important than framework version |
| **State persistence** | JSON files (recipe_graph, mods_list, recipe_pot) | Database-backed (Phase 3) |
| **API layer** | None (direct Python invocation) | REST API, likely FastAPI |
| **Front-end** | CLI interaction loop | Web-based conversational UI (Phase 2) |
| **Hardware** | Serial logs, PyQt GUI prototype | Integrated sensor protocol + print pipeline (Phase 4) |
| **Testing** | pytest (68 tests, unit + integration) | TDD baseline with CI gates (Phase 1), perf regression tests (Phase 3) |
| **ML / Research** | None | Flavor embeddings, coherence models, substitution engine (Phase 7) |

---

_This charter is a living document. It should be updated as the project evolves — not preserved as an artifact of what we once intended._
