## FEATURE:

- Pocket Flow multi agent project.
- The goal of the project is to retrieve emails from the past month, filter the email by title to determine if they are job application, the provide a summary table with the state of all job applications
- Research Agent to try to find the related job posting on Linkedin,Glassdoor,etc.
- CLI to interact with the agent.
- Gmail for the email agent, Brave API for the research agent.

## EXAMPLES:

In the `examples/` folder, there is a README for you to read to understand what the example is all about and also how to structure your own README when you create documentation for the above feature.

- `examples/cli.py` - use this as a template to create the CLI
- `examples/agent/` - read through all of the files here to understand best practices for creating Pocket Flow Nodes that support different providers and LLMs, handling agent dependencies, and adding tools to the agent.

Don't copy any of these examples directly, it is for a different project entirely. But use this as inspiration and for best practices.

## DOCUMENTATION:

Pocket Flow documentation: https://the-pocket.github.io/PocketFlow/

## OTHER CONSIDERATIONS:

- Include a .env.example, README with instructions for setup including how to configure Gmail and Brave.
- Include the project structure in the README.
- Virtual environment has already been set up with the necessary dependencies.
- Use python_dotenv and load_env() for environment variables
