from mcp.server.fastmcp import FastMCP
import requests
import json
import sys
import os

webhook = os.environ.get("DISCORD_WEBHOOK_URL")

mcp = FastMCP("discord")


@mcp.tool()
def send_message(msg: str):
    """
    Sends a message to the discord server
    :param msg (str): The input message to send
    """
    data = {"content": msg}
    headers = {"Content-type": "application/json"}
    response = requests.post(webhook, data=json.dumps(data), headers=headers)
    return f"Message sent! Status: {response.status_code}"


if __name__ == "__main__":
    print("Starting MCP server...", file=sys.stderr)
    mcp.run(transport="stdio")
