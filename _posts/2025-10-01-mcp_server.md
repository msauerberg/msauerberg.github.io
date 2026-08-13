---
layout: post
title: "MCP Server - Spotify and Claude"
date: 2025-10-01
description: Building an MCP Server to use Spotify with Claude 
tags: [python]
---

# What is an MCP Server?

According to ChatGPT, an MCP (Model Context Protocol) Server is
"**a specialized system that provides structured data, tools, or workflows to AI models in a standardized way. Instead of hard-coding connections between applications and models, an MCP server acts as a bridge—exposing information or capabilities through a protocol that models can easily consume. This makes it possible to connect large language models (LLMs) with external databases, APIs, or custom business logic without losing consistency or control. In short, an MCP server helps extend what AI can do by safely integrating it with real-world data and operations.**".
To me, it is a powerful framework that allows equipping the AI with tools to execute tasks such as sending emails, checking the weather forcast for the next day, or checking flights and prices on airline webpages. Even though, the framework is rather new, there are already many MCP servers available, e.g., on [https://mcpservers.org/](https://mcpservers.org/) or [https://mcp.so/](https://mcp.so/).

# How to build your own MCP Server in Python?

Getting a simnple MCP server to run, is actually not so hard. A very simple example is provided here [https://modelcontextprotocol.io/examples](https://modelcontextprotocol.io/examples). Also, youtubers such as **NeuralNine** uploaded videos about it (check, for example, **Coding Your Own Custom MCP Server in Python - Full Tutorial** by **NeuralNine**). In this blog post, I will present my own beginner MCP server to you. It will connect Spotify to the AI, enabling the AI to get data on music from Spotify.

# Spotipy - The Spotify Web API

The python library [https://spotipy.readthedocs.io/en/2.25.1/](https://spotipy.readthedocs.io/en/2.25.1/) provides many useful functions to retrieve information on music from Spotify. To use it, however, you would need to sign up to Spotify and demand your developer credentials. This is described here: [https://github.com/spotipy-dev/spotipy/blob/2.22.1/TUTORIAL.md](https://github.com/spotipy-dev/spotipy/blob/2.22.1/TUTORIAL.md). Please note that some function have been deprecated unfortunately. In the past, it was possible to get features such as "danceability", "speechiness" or "energy" for each song. Further, a function that returns similar songs based on a given input song has been deprecated as well. For the MCP server, I will focus on the function named "artist_top_tracks()". It is still available and returns the top 10 songs from a given artist in a given country. For instance, after setting up Spotipy and storing your credentials in the .env-file, you could get the top 10 tracks from Boards of Canada in the US.

```python
import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException
import pandas as pd

# Load variables from .env
load_dotenv()

# Access them
client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")



sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri,
                                               scope="user-top-read"))

def get_top_tracks_country(sp, artist_name: str, country: str = "DE"):

    if country not in spotipy.Spotify.country_codes:
        raise ValueError("country code is not valid")
    
    # search artist since we need the artist ID for the artist_top_tracks() function
    result = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
    if len(result["artists"]["items"]) == 0:
        raise ValueError(f"Artist '{artist_name}' not found on Spotify.")
    artist_id = result["artists"]["items"][0]["id"]

    # get top tracks
    top_tracks = sp.artist_top_tracks(artist_id, country=country)["tracks"]

    # get dataframe
    data = []
    for track in top_tracks:
        data.append({
            "track_name": track["name"],
            "artist_name": ", ".join([a["name"] for a in track["artists"]]),
            "album": track["album"]["name"]
        })

    return pd.DataFrame(data)

df_top_tracks = get_top_tracks_country(sp, artist_name = "Boards of Canada", country="US")
display(df_top_tracks)
```

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/BoC.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

# Coding the MCP Server

I am using the FastMCP library to make the code from above work in a MCP server framework. The line **mcp = FastMCP("Spotify Top Tracks")** defines the server and tools can be provided by writing **@mcp.tool()**, followed by a function (see code below).
For the MCP server, I have adjusted the code slightly to avoid problem with authentification (I dont want the browser popping up and asking for Spotify username and password). Also, I am mapping country names to country codes. You might prompt "What are the top tracks for The Rolling Stones in Germany?". Then, "Germany" needs to be mapped to its country code "DE in order to call the function correctly. Finally, my function does not return a pandas dataframe anymore but a dictionary. This is easier to process for the AI. Please note docstrings (the description of the function) are very important as the AI uses them as guidance. guidance. The clearer and more detailed the docstrings are, the better the model can reason about when and how to call the function.

```python
import os
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from mcp.server.fastmcp import FastMCP

# Load Spotify keys from .env
load_dotenv()

# Access them from .env
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
))

mcp = FastMCP("Spotify Top Tracks")

# Mapping between country code and country name
_ALIASES = {
    "AD": "Andorra",
    "AR": "Argentina",
    "AU": "Australia",
    "AT": "Austria",
    "BE": "Belgium",
    "BO": "Bolivia",
    "BR": "Brazil",
    "BG": "Bulgaria",
    "CA": "Canada",
    "CL": "Chile",
    "CO": "Colombia",
    "CR": "Costa Rica",
    "CY": "Cyprus",
    "CZ": "Czech Republic",
    "DK": "Denmark",
    "DO": "Dominican Republic",
    "EC": "Ecuador",
    "SV": "El Salvador",
    "EE": "Estonia",
    "FI": "Finland",
    "FR": "France",
    "DE": "Germany",
    "GR": "Greece",
    "GT": "Guatemala",
    "HN": "Honduras",
    "HK": "Hong Kong",
    "HU": "Hungary",
    "IS": "Iceland",
    "ID": "Indonesia",
    "IE": "Ireland",
    "IT": "Italy",
    "JP": "Japan",
    "LV": "Latvia",
    "LI": "Liechtenstein",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "MY": "Malaysia",
    "MT": "Malta",
    "MX": "Mexico",
    "MC": "Monaco",
    "NL": "Netherlands",
    "NZ": "New Zealand",
    "NI": "Nicaragua",
    "NO": "Norway",
    "PA": "Panama",
    "PY": "Paraguay",
    "PE": "Peru",
    "PH": "Philippines",
    "PL": "Poland",
    "PT": "Portugal",
    "SG": "Singapore",
    "ES": "Spain",
    "SK": "Slovakia",
    "SE": "Sweden",
    "CH": "Switzerland",
    "TW": "Taiwan",
    "TR": "Turkey",
    "GB": "United Kingdom",
    "US": "United States",
    "UY": "Uruguay",
}

_NAME_TO_CODE = {name.lower(): code for code, name in _ALIASES.items()}

def _normalize_country_code(value: str) -> str:
    """
    Normalize an input (2-letter code or country name) to a valid _ALIASES code.
    Raises ValueError if not found or not supported.
    """
    if not value:
        raise ValueError("country must be provided")

    val = str(value).strip()

    # if the user passed a 2-letter code
    if len(val) == 2 and val.isalpha():
        code = val.upper()
        if code in _ALIASES:
            return code
        raise ValueError("country code is not valid")

    # otherwise try to match by country name
    code = _NAME_TO_CODE.get(val.lower())
    if code:
        return code

    # if not found
    raise ValueError("country code or name is not valid or not available in this market")

# Tool for getting top tracks by artist name and country code 
@mcp.tool()
def get_top_tracks_country(artist_name: str, country: str = "DE") -> List[Dict[str, Any]]:
    """
    Return top tracks for an artist in a country.
    - artist_name: artist (string)
    - country: 2-letter code (e.g., 'DE') or country name (e.g., 'Germany')
    Returns list[dict] with keys: track_name, artist_name, album, popularity, track_id
    """
    artist_name = (artist_name or "").strip()
    if not artist_name:
        raise ValueError("artist_name must be provided")

    code = _normalize_country_code(country)

    # search artist to get the artist ID (input for artist_top_tracks())
    res = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
    artists = res.get("artists", {}).get("items", [])
    if not artists:
        raise ValueError(f"Artist '{artist_name}' not found on Spotify.")
    artist_id = artists[0]["id"]

    top = sp.artist_top_tracks(artist_id, country=code).get("tracks", [])
    out = []
    for t in top:
        out.append({
            "track_name": t.get("name"),
            "artist_name": ", ".join([a.get("name") for a in t.get("artists", []) if a.get("name")]),
            "album": t.get("album", {}).get("name"),
            "popularity": t.get("popularity"),
            "track_id": t.get("id")
        })
    return out


if __name__ == "__main__":
    mcp.run(transport = "stdio")

```

Now, we only need to update the config in Claude Desktop. To do so, click on File -> Settings -> Developer -> config. It should open a JSON file which I have updated to:

```python
{
  "mcpServers": {
    "Demo": {
      "command": "C:\\Python\\MCP_server\\.venv\\Scripts\\python.exe",
      "args": ["-u", "C:\\Python\\MCP_server\\main.py"]
    }
  }
}
```
The path after "command" points to the python.exe in my virtual environment with all required packages installed, whereas the path after "args" points to the file. I needed to shut down claude desktop fully (using the task manager) and then open it again before it worked. If you struggle with this step, please refer to one of the many youtube tutorials out there or ask the AI =)

# Demonstration of the MCP Server

When asking Claude about top tracks by an artist in a given country, it recognizes that there is a tool for it (the MCP server). It will ask you then whether you would like to use it.
After approval, it will use the tool and report its result back to you.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/claude_2.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/claude_3.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/claude_1.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

# More options, more functions

This is just a simple test to see how the MCP server works. There are at least two much better MCP Servers for Spotify available, see [https://github.com/varunneal/spotify-mcp](https://github.com/varunneal/spotify-mcp) and [https://github.com/marcelmarais/spotify-mcp-server](https://github.com/marcelmarais/spotify-mcp-server). They allow not only to get the top tracks but to create playlists, get recently played songs and much more. Thanks for reading.

# Use of AI

I used ChatGPT by OpenAI and Claude by Anthropic to assist with the development of the code. AI was also used in the preparation of this blog post, including brainstorming, improving the wording, translations, and formatting. The content was reviewed, verified, and finalized by me.