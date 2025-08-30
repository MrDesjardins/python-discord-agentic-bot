# python-discord-agentic-bot

The Agentic Bot allows user to ask questions and having the bot answer as the best it could with the information it could find from several sources.

The agentic bot works with a custom dynamic workflow using the LangGraph framework. The AI agents employs several sources like a SQL database, APIs, web blog posts, other bots and Discord messages (via API).

## Python Environment and Dependencies

Here are few commands to set up the environment and dependencies.

```sh
# Install Packages that Python will need
sudo apt update
sudo apt install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    uuid-dev \
    wget \
    curl \
    llvm \
    make \
    git

# Add Python to a specific version
pyenv install 3.12.10
pyenv local 3.12.10

# Create Environment
python3 -m venv .venv

# Activate Environment
source .venv/bin/activate

# Update PIP
python3 -m pip install --upgrade pip

# Install Requirements
python3 -m pip install -r requirements.txt

# Save Requirements
python3 -m pip freeze > requirements.txt
```


## Running the bot in Dev:

To run the bot in development:

```sh
source .venv/bin/activate
python3 bot.py # or ./bot.py
```

## SQLite3

Add in your `~/.zshrc` or `~/.bashrc` the following line to have the SQLite3 headers and columns configuration from the root .sqliterc file.

```sh
alias sqlite3="sqlite3 --init .sqliterc"


# Unit Tests

```sh
pytest -v -s ./tests/*_unit_test.py
```

# Integration Tests

```sh
pytest -v -s ./tests/*_integration_test.py
```

# Coverage Tests on Unit Tests

```sh
coverage run --omit="./tests/*" -m pytest -v -s ./tests/*unit_test.py && coverage html
```


# Discord

## Discord Configuration

### Set Up a Discord Bot Account

Create a Discord Application:

1. https://discord.com/developers/applications
1. Click on "New Application" and give it a name.
1. Currently, this application has two applications: `AgenticSiegeBot` and `AgenticSiegeBot_Dev`. The first one is for production and the second one is for development.

### OAuth2

Change the redirect to `https://discord.com/oauth2/authorize`

Navigate to the `OAuth2` tab and then `URL Generator`.
Under `Scopes` select:

1. `bot`
1. `applications.commands`

Under "Bot Permissions", select the necessary permissions:

1. View Audit Log
1. Manage Roles
1. Manage Channels
1. View Channels
1. Send Messages
1. Manage Messages
1. Read Message History
1. Add Reactions

Copy the generated URL and open it in your browser to invite the bot to your desired server.

Example: 
```
https://discord.com/oauth2/authorize?client_id=<CLIENT_ID>&permissions=277032144&integration_type=0&scope=bot+applications.commands
```

### Bot

Under Privileges Gateway Intents, select:

1. Prensence Intent
1. Server Members Intent
1. Message Content Intent


### Create a Bot:

In your application, navigate to the "Bot" tab.

1. Click "Add Bot" and confirm.
1. Under "TOKEN", click "Copy" to get your bot's token. Keep this token secure!

### Invite the Bot to Your Server:

Navigate to the `OAuth2` tab and then `URL Generator`.
Under `Scopes` select:

1. `bot`
1. `applications.commands`

Under "Bot Permissions", select the necessary permissions:

1. View Audit Log
1. Manage Roles
1. Manage Channels
1. View Channels
1. Send Messages
1. Manage Messages
1. Read Message History
1. Add Reactions

Copy the generated URL and open it in your browser to invite the bot to your desired server.