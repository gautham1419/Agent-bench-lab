#from .fastchat_client import FastChatAgent

from .http_agent import HTTPAgent

try:
    from .fastchat_client import FastChatAgent
except ModuleNotFoundError:
    pass