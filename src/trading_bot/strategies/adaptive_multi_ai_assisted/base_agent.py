"""
Modern BaseAgent using Instructor - Clean Implementation
"""
import os
import inspect
from typing import Dict, Optional, Any, Type, TypeVar
from abc import ABC

import structlog
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel
import instructor
import anthropic

from trading_bot.core.settings import settings


logger = structlog.get_logger(__name__)

class ValidationError(Exception):
    pass

class LLMInteractionError(Exception):
    pass

class TemplateNotFoundError(Exception):
    pass

T = TypeVar("T", bound=BaseModel)

class BaseAgent(ABC):
    """Modern base agent class using Instructor for structured outputs."""
    
    def __init__(
        self,
        name: str,
        template_dir: Optional[str] = None,
        system_message_template: str = "system_message.j2",
        system_message_kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        agent_dir: Optional[str] = None,
    ):
        if agent_dir is None:
            agent_dir = os.path.dirname(inspect.getfile(self.__class__))
            
        self.agent_dir = agent_dir
        self.logger = logger.bind(agent=name)
        self.name = name
        self.max_retries = max_retries
        
        # Validate Claude API configuration
        if not settings.claude_api_key:
            raise ValueError("Claude API key is not configured.")
        
        if not settings.claude_api_key.startswith("sk-ant-"):
            raise ValueError("Invalid Claude API key format. Expected key to start with 'sk-ant-'")

        # Initialize Instructor with Claude
        raw_client = anthropic.AsyncAnthropic(
            api_key=settings.claude_api_key,
            timeout=100
        )
        self.client = instructor.from_anthropic(
            raw_client,
            mode=instructor.Mode.ANTHROPIC_TOOLS
        )
        self.model = settings.claude_model

        # Setup templates
        if template_dir is None:
            template_dir = os.path.join(self.agent_dir, "templates")
        
        self.template_dir = template_dir
        self.system_message_template = system_message_template
        template_path = os.path.join(template_dir, system_message_template)
        
        if not os.path.exists(template_path):
            raise TemplateNotFoundError(f"System message template not found: {template_path}")
        
        self._setup_template_environment(template_dir)
        self.sys_msg_kwargs = system_message_kwargs or {}
        self.system_message = self.get_prompt_from_template(
            system_message_template, **self.sys_msg_kwargs
        )

    def _setup_template_environment(self, template_dir: str) -> None:
        """Setup Jinja2 template environment."""
        if not os.path.isdir(self.template_dir):
            raise FileNotFoundError(f"Template directory not found: {self.template_dir}")

        self.template_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def get_prompt_from_template(self, template_name: str, **kwargs) -> str:
        """Render a Jinja2 template with given kwargs."""
        template_path = os.path.join(self.template_dir, template_name)
        if not os.path.exists(template_path):
            raise TemplateNotFoundError(f"Template not found: {template_path}")
        template = self.template_env.get_template(template_name)
        return template.render(**kwargs)

    async def generate_reply(
        self,
        prompt_template: str,
        response_model: Type[T],
        **kwargs,
    ) -> T:
        """Generate structured response using Instructor."""
        prompt = self.get_prompt_from_template(prompt_template, **kwargs)
        response =  await self.generate_reply_with_raw_prompt(
            prompt=prompt,
            response_model=response_model,
        )
        
        # logger.info("Generated response", response=response.model_dump())
        return response

    async def generate_reply_with_raw_prompt(
        self,
        prompt: str,
        response_model: Type[T],
    ) -> T:
        """Generate structured response with Instructor."""
        
        self.logger.info(f"Generating reply for {response_model.__name__}")

        try:
            result = await self.client.chat.completions.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                system=self.system_message,
                messages=[{"role": "user", "content": prompt}],
                response_model=response_model,
                max_retries=self.max_retries
            )
            
            self.logger.info(f"Successfully generated {response_model.__name__}")
            return result
            
        except Exception as e:
            # Catch all exceptions and handle them properly
            error_str = str(e)
            error_type = type(e).__name__
            
            self.logger.error(f"Generation error: {error_type} - {error_str}")
            
            # Check for specific error patterns
            if "validation" in error_str.lower():
                raise ValidationError(f"Response validation failed: {error_str}") from e
            elif "retry" in error_str.lower() or "rate" in error_str.lower():
                raise LLMInteractionError(f"Failed after retries: {error_str}") from e
            elif "authentication" in error_str.lower():
                raise LLMInteractionError(f"Claude API authentication failed: {error_str}") from e
            else:
                raise LLMInteractionError(f"Failed to generate reply: {error_str}") from e
