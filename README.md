# Alduin: Build Your Own CLI Coding Agent from Scratch

This is the starting codebase for the [Build Your Own CLI Coding Agent from Scratch](https://luma.com/5yt57yep) workshop. It's a 2.5-hour hands-on session where we'll build a small but practical CLI coding agent in Python, starting from a basic LLM loop and iteratively adding capabilities like file reading, file writing, and shell command execution.

The session is mostly live coding with minimal lecture. By the end, you'll have a working coding agent that you built yourself, that you understand completely, and that you can customize however you want.

## Pre-event Environment Setup

Please go through these steps before the workshop so we can jump straight into coding when we start. If you run into issues, reach out and I'll help you sort it out.

### 1. Install uv

uv is a fast Python package manager that we'll use throughout the workshop. If you don't have it installed already, follow the instructions on the [official uv installation page](https://docs.astral.sh/uv/getting-started/installation/).

You can verify it's working by running:

```bash
uv --version
```

### 2. Get an Anthropic API Key

We'll be standardizing on Anthropic's API for this workshop. If you don't have an API key yet, head over to the [Anthropic Console](https://console.anthropic.com/settings/keys) and create one. Make sure you have at least $5-10 of API credits loaded up. We'll be making a bunch of API calls during the session and you don't want to hit a limit halfway through.

### 3. Clone the Repository

```bash
git clone https://github.com/primaprashant/alduin.git
cd alduin
```

### 4. Set Up Your Environment File

The repo includes a `.env.sample` file. Copy it to create your own `.env` file and add your API key:

```bash
cp .env.sample .env
```

Then open `.env` in your editor and fill in your Anthropic API key.

### 5. Install Dependencies

Run the following command from the project root:

```bash
make dep
```

This uses uv under the hood, so it will automatically download Python 3.12 if you don't have it, create a virtual environment, and install all the project dependencies. You don't need to manage any of that yourself.

### 6. Verify Everything Works

Finally, let's make sure your API key is set up correctly and you can actually talk to the Anthropic API:

```bash
make check-api-key
```

This sends a simple test request to the API. If you see a response and a green checkmark, you're all set. If it errors out, double-check that your `.env` file has the right key and that you have credits on your account.

After finishing all these steps, **go to [Pre-event Environment Setup Poll on GitHub](https://github.com/primaprashant/alduin/discussions/1) and vote** so that I know how many people are ready to go. If you have any issues with the setup, you can also comment on that poll and I'll help you out.

GitHub poll link: https://github.com/primaprashant/alduin/discussions/1

That's it. See you at the workshop!
