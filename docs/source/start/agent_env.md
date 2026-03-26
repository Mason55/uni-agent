# Launch an Agent Environment

Long-horizon agent tasks, such as software engineering, need a **persistent sandbox**: a place where the agent can run commands, modify the environment, and preserve state across many steps. This document shows how to start a sandbox, install tools, and run a simple script.

The runnable code referenced in this document lives under `examples/agent_env`. You can run the example with:

```bash
DEPLOYMENT=<local|vefaas> DEBUG_MODE=1 python examples/agent_env/demo.py
```

Setting `DEBUG_MODE=1` prints all intermediate startup and runtime output to the current terminal.

---

## Start a Sandbox

The first step is to start a sandbox. You can do this either locally or on a remote FaaS platform.

### Local deployment

*Coming soon. You will be able to run the sandbox on your machine or on your own cluster.*

### Remote deployment (VEFAAS)

VEFAAS is a Volcengine FaaS platform. For workloads with many concurrent runs, it is often more stable and scales better than self-hosted local instances.

**Dependencies.** Install the required packages:

```bash
pip install volcenginesdkvefaas swerex
```

**Environment variables.** Set your credentials and optional function settings in the environment.

```bash
export VOLCE_ACCESS_KEY=xxxxxxxxxx
export VOLCE_SECRET_KEY=xxxxxxxxxx
export VEFAAS_FUNCTION_ID=xxxxxxxxxx
export VEFAAS_FUNCTION_ROUTE=xxxxxxxxxx
```

| Variable | Description |
|----------|-------------|
| `VOLCE_ACCESS_KEY` or `VOLCENGINE_ACCESS_KEY` | Volcengine access key |
| `VOLCE_SECRET_KEY` or `VOLCENGINE_SECRET_KEY` | Volcengine secret key |
| `VEFAAS_REGION` | Region (optional, default `cn-beijing`) |

You can also set `VEFAAS_FUNCTION_ID` and `VEFAAS_FUNCTION_ROUTE` directly in the config below as `function_id` and `function_route`. Get your access credentials from the [Volcengine](https://www.volcengine.com/) console, and get the function ID and route from the VEFAAS console.

**Config and start.** Build the config, create the environment, and start it:

```python
import os
import uuid
from uni_agent.interaction import AgentEnv, AgentEnvConfig

run_id = str(uuid.uuid4())
env_config = {
    "deployment": {
        "type": "vefaas",
        "image": "enterprise-public-2-cn-beijing.cr.volces.com/vefaas-public/python:3.12",
        "command": "curl -fsSL https://pjw-test-empty.tos-cn-beijing.ivolces.com/bin/tos_swe_rex.sh | bash -s -- {token}",
        "timeout": 300.0,
        "startup_timeout": 180.0,
        "function_id": os.getenv("VEFAAS_FUNCTION_ID"),
        "function_route": os.getenv("VEFAAS_FUNCTION_ROUTE"),
    },
    "env_variables": {
        "PIP_PROGRESS_BAR": "off",
    }
}
env_config = AgentEnvConfig(**env_config)
env = AgentEnv(run_id=run_id, env_config=env_config)
env.start()
```

- **`run_id`**: A unique identifier for the run, used in logging and deployment.
- **`deployment`**: `type` must be `"vefaas"`.
  - `image` is the sandbox Docker image.
  - `command` is the startup command.
  - `timeout` and `startup_timeout` are specified in seconds.
  - `function_id` and `function_route` identify your VEFAAS function.
- **`env_variables`**: Environment variables exported into the sandbox shell after startup.

---

## Install Tools

Once the sandbox is running, install tools so the agent can execute bash commands and edit files:

```python
from uni_agent.tools import ToolConfig

tools_config = [
    {"name": "execute_bash"},
    {"name": "str_replace_editor"},
]
tools = [ToolConfig(**tool_config).get_tool() for tool_config in tools_config]
env.install_tools(tools)
```

We provide common tool implementations for tasks such as running bash commands and editing files. You can also customize and integrate your own tools.

You can verify that the tools were installed successfully by running:

```python
print(env.communicate("which str_replace_editor"))
```

This command returns:
```python
/usr/local/bin/str_replace_editor
```
This indicates that the installation succeeded.

---

## Run the Demo

The demo runs a few simple steps to show sandbox persistence: install a dependency, create a script, execute it, and read the output.

**1. Install numpy**

```python
env.communicate("pip install numpy -q")
```

The dependency is installed in the current sandbox and persists for the rest of the run.

**2. Create a script and write its output to a file**

Create a small Python script with `str_replace_editor`, then run it and redirect its stdout to a file:

```python
import shlex
_script = "import numpy as np; print(np.array([1,2,3]).sum())"
env.communicate(f"str_replace_editor create --path /tmp/demo.py --file_text {shlex.quote(_script)}")
env.communicate("execute_bash 'python3 /tmp/demo.py > /tmp/demo_out.txt'")
```

**3. View the result**

```python
print(env.communicate("cat /tmp/demo_out.txt"))  # -> 6
```

**4. Close the environment**

```python
env.close()
```

After setting your credentials, you can run the full demo from the repo root with:

```bash
python examples/agent_env/demo.py
```

## Quick Reference

| Step | Code |
|------|------|
| Config & start | `AgentEnvConfig(**env_config)` → `AgentEnv(...)` → `env.start()` |
| Tools | `env.install_tools(tools)` |
| Run command | `env.communicate("...")` |
| Close | `env.close()` |
