# Overview

Caldron is an AI-assisted recipe development platform that allows users to generate and quickly iterate on new recipes. Caldron leverages multi-agent generative artificial intelligence tools to quickly iterate a desired foundational recipe from a provided prompt and optionally provided context. Caldron then provides channels for both human and machine sensory feedback to iteratively refine the recipe.

## Table of Contents

- [Overview](#overview)
- [System Workflow](#system-workflow)
- [Files and Modules](#files-and-modules)
  - [agent_defs.py](#agent_defspy)
  - [agent_tools.py](#agent_toolspy)
  - [cauldron_app.py](#cauldron_apppy)
  - [class_defs.py](#class_defspy)
  - [langchain_util.py](#langchain_utilpy)
  - [logging_util.py](#logging_utilpy)
  - [main.py](#mainpy)
  - [util.py](#utilpy)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## System Workflow

The system works as follows:
1. User provides a prompt and optionally context (i.e. custom ingredients, dietary restrictions, recipe sources, etc.)
2. Caldron parses the request and determines what type of workflow to execute on a resources given (i.e. local database of recipes, web search, available APIs)
3. Caldron executes the query and returns a list of relevant information.
4. Caldron formats the returned information into a context for recipe generation.
5. Caldron generates a foundational recipe based on the context and prompt.
6. Caldron provides the foundational recipe to the user for feedback.
7. The user provides feedback on the recipe and the foundational recipe is tweaked until the user is satisfied.
8. Caldron provides interaction points for human feedback (in the form of written and spoken language) and machine feedback (in the form of sensory data provided by IoT devices).
9. The user is free to cook/bake the recipe and provide feedback on the final product.
10. Caldron intelligently uses this feedback to further refine the recipe as well as save a "snapshot" of the recipe attempt for future reference.

## Files and Modules

### agent_defs.py

This module contains the definitions of various agents used in the application. Agents are core components that interact with different parts of the system to perform designated tasks.

### agent_tools.py

This module integrates various tools required by the agents to function correctly. It includes tool initialization and management functionalities.

### cauldron_app.py

This module is the main application logic that coordinates between different components. It acts as a central hub, orchestrating the flow of data and control.

### class_defs.py

This module contains class definitions used throughout the application. It includes data models, helper classes, and other structures essential for the application's operation.

### langchain_util.py

This module provides utilities related to language processing and chaining tasks together. It includes functions for handling language-specific operations and chaining processes.

### logging_util.py

This module is responsible for setting up and managing the logging functionality of the application. It ensures that all actions and events within the system are properly logged for debugging and monitoring purposes.

### main.py

This is the entry point of the application. It initializes the necessary components and starts the application, managing the overall workflow and execution.

### util.py

This module contains various utility functions that are used across the application. These functions perform common tasks that do not belong to any specific module but are essential for the application's functionality.

## Installation

To install the necessary dependencies and set up the application, run the following commands:

```bash
pip install -r requirements.txt
```

## Usage

To start the current demo visualization of the application, run:

```bash
python main.py
```

Ensure that all necessary configuration files are in place and properly configured before starting the application.

## Contributing

Contributions are welcome! Please follow the standard GitHub flow for contributing:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for more details.