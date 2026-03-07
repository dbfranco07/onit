'''
# Copyright 2025 Rowel Atienza. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ToolRegistry - where all tools are registered and managed.
RequestHandler - abstract base class for all tools.
ToolHandler - concrete implementation of RequestHandler that calls a tool via FastMCP client.
'''

import tempfile
import os
import uuid
import base64
import wave
import random
from fastmcp import Client
from abc import ABC, abstractmethod
from mcp.types import ImageContent, TextContent, AudioContent

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def get_tools(tool_registry):
    """Get the tools from the tool registry"""
    tools = []
    for key in tool_registry.handlers:
        tools.append(tool_registry[key].get_tool())
    return tools

class RequestHandler(ABC):
    def __init__(self,
                 url: str = None,
                 tool_item: dict = None,
                 **kwargs: dict):
        super().__init__()
        self.url = url
        self.tool_item = tool_item
        self.kwargs = kwargs

    @abstractmethod
    async def __call__(self, **kwargs: dict) -> str:
        pass

    def get_tool(self) -> dict:
        return self.tool_item
    
    
class ToolHandler(RequestHandler):
    def __init__(self, 
                 url: str = None,
                 tool_item: dict = None,  
                 **kwargs: dict):
        super().__init__(url=url, tool_item=tool_item, **kwargs)

    async def __call__(self, **kwargs: dict) -> str:
        # if there is an image in the kwargs, convert it to base64
        media_types = ['images', 'audios']
        for media_type in media_types:
            if media_type in kwargs:
                if isinstance(kwargs[media_type], list) and len(kwargs[media_type]) == 0:
                    if media_type == 'images':
                        return f"No {media_type} provided. For images, use the camera tool to capture an image first."
                    else:
                        return f"No {media_type} provided. For audios, use the microphone tool to record an audio first."
                kwargs[media_type] = kwargs[media_type][0] if isinstance(kwargs[media_type], list) else kwargs[media_type]
                if isinstance(kwargs[media_type], str):
                    # check if the path is a valid file path
                    if not os.path.exists(kwargs[media_type]):
                        # try removing leading dot
                        kwargs[media_type] = kwargs[media_type].lstrip('.')
                        if not os.path.exists(kwargs[media_type]):
                            logger.error(f"File not found: {kwargs[media_type]}")
                            return f"File not found: {kwargs[media_type]}"
                    with open(kwargs[media_type], 'rb') as image_file:
                        kwargs[media_type] = [base64.b64encode(image_file.read()).decode('utf-8')]
                elif isinstance(kwargs[media_type], bytes):
                    # if the image is already in bytes, convert to base64
                    kwargs[media_type] = [base64.b64encode(kwargs[media_type]).decode('utf-8')]
                # if dictionary, extract the value and encode it
                elif isinstance(kwargs[media_type], dict):
                    for key, value in kwargs[media_type].items():
                        if isinstance(value, str):
                            with open(value, 'rb') as image_file:
                                kwargs[media_type] = [base64.b64encode(image_file.read()).decode('utf-8')]
                        elif isinstance(value, bytes):
                            kwargs[media_type] = [base64.b64encode(value).decode('utf-8')]
                        else:
                            logger.error(f"Unsupported image type: {type(value)}")
                            return "Unsupported image type provided."
                else:
                    logger.error(f"Unsupported images type: {type(kwargs[media_type])}")
                    return f"Unsupported {media_type} type provided. Please provide a list of base64-encoded {media_types} or file paths."
                    
        
        async with Client(self.url) as client:
            keys_kwargs = list(kwargs.keys())
            logger.info(f"Calling tool: {self.tool_item['function']['name']} with arguments: {keys_kwargs}")
            tool_response = await client.call_tool(self.tool_item['function']['name'], kwargs)
        
        if isinstance(tool_response, str):
            return tool_response

        content = tool_response.content
        if not isinstance(content, list):
            content = [content]

        # Handle multi-content responses (e.g., rotate_and_scan returns
        # text + multiple images).  Collect all text parts and save all
        # images to temp files, returning a combined text response.
        text_parts = []
        image_paths = []

        for item in content:
            # MCP data types: https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/types.py
            if isinstance(item, ImageContent):
                image_data = base64.b64decode(item.data)
                # Use the MIME type from MCP to pick the correct file extension
                # so that _maybe_inject_image sends the right Content-Type.
                _mime_to_ext = {
                    "image/jpeg": ".jpg", "image/png": ".png",
                    "image/gif": ".gif", "image/bmp": ".bmp",
                    "image/webp": ".webp",
                }
                ext = _mime_to_ext.get(getattr(item, 'mimeType', ''), '.jpg')
                unique_filename = f"{uuid.uuid4()}{ext}"
                default_image = os.path.join(tempfile.gettempdir(), unique_filename)
                with open(default_image, "wb") as f:
                    f.write(image_data)
                image_paths.append(default_image)
            elif isinstance(item, TextContent):
                text_parts.append(item.text)
            elif isinstance(item, AudioContent):
                audio_data = item.data
                if len(audio_data) == 0:
                    logger.warning("No audio data returned from the tool.")
                    continue
                if isinstance(audio_data, str):
                    logger.info(f"Audio data is a base64 string of length {len(audio_data)}")
                audio_format = item.format if hasattr(item, 'format') else 'wav'
                audio_data = base64.b64decode(item.data)
                meta = item.metadata if hasattr(item, 'metadata') else {}
                logger.info(f"Audio data format: {audio_format}, metadata: {meta}")
                unique_filename = f"{uuid.uuid4()}.{audio_format}"
                default_audio = os.path.join(tempfile.gettempdir(), unique_filename)
                with wave.open(default_audio, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_data)
                text_parts.append(f"{default_audio}")

        # Single-image shortcut: return just the path so _maybe_inject_image
        # can detect it (backward compatible with existing single-image tools).
        if len(image_paths) == 1 and not text_parts:
            return image_paths[0]

        # Multi-content: combine text + image paths
        if image_paths:
            text_parts.append("\n".join(image_paths))

        if text_parts:
            return "\n".join(text_parts)

        return "Undefined content type returned from the tool."

# create a tool registry
class ToolRegistry:
    def __init__(self):
        self.tools = set()
        self.urls = {}
        self.handlers = {}

    def register(self, tool: ToolHandler):
        # many need to modify to support one to many tools
        # self.tools[tool.tool_item['function']['name']] = tool
        
        tool_name = tool.tool_item['function']['name']
        tool_url = tool.url
        self.tools.add(tool_name)
        if tool_name not in self.urls:
            self.urls[tool_name] = [tool_url]
        else:
            self.urls[tool_name].append(tool_url)
        
        key = f"{tool_name}@{tool_url}"
        self.handlers[key] = tool
        
    def get_url(self, tool_name: str):
        """Get a random URL for the tool"""
        if tool_name not in self.urls:
            return None
        urls = self.urls[tool_name]
        index = random.randint(0, len(urls) - 1)
        return urls[index]    
        
    def get_tool_items(self):
        """Get the tool items from the tool registry"""
        tool_items = []
        for key in self.handlers:
            tool_items.append(self.handlers[key].get_tool())
        return tool_items

    def get_handler_by(self, tool_name: str, url: str) -> ToolHandler:
        """Get a tool handler by tool name and URL"""
        if tool_name is None or url is None:
            None
        
        key = f"{tool_name}@{url}"
        if key in self.handlers:
            return self.handlers[key]
        return None
        

    def __getitem__(self, tool_name: str):
        if tool_name not in self.tools:
            return None
        urls = self.urls[tool_name]
        index = random.randint(0, len(urls) - 1)
        key = f"{tool_name}@{urls[index]}"
        return self.handlers[key]

    def __len__(self):
        return len(self.tools)

    def __iter__(self):
        return iter(self.tools)
    
    
