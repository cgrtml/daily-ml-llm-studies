"""
Prompt Template Generator
------------------------
A tool for generating effective prompts for various common tasks with large language models.
This module provides templates and customization for different prompt engineering techniques.
"""

import argparse
import json
import random
from typing import Dict, List, Optional, Union


class PromptGenerator:
    """A class for generating and managing prompt templates for LLMs."""

    def __init__(self):
        """Initialize the PromptGenerator with predefined templates."""
        # Load template library
        self.templates = {
            "classification": {
                "zero_shot": "Classify the following {content_type} into one of these categories: {categories}.\n\n{content}",
                "few_shot": self._few_shot_classification_template,
                "chain_of_thought": "Classify the following {content_type} into one of these categories: {categories}.\n\nLet's think about this step by step to determine the most appropriate category.\n\n{content}"
            },
            "summarization": {
                "default": "Provide a {length} summary of the following {content_type}:\n\n{content}",
                "bullet_points": "Summarize the main points of the following {content_type} as a bullet point list with {num_points} points:\n\n{content}",
                "eli5": "Explain the following {content_type} in simple terms as if you're explaining to a 5-year-old:\n\n{content}"
            },
            "content_creation": {
                "blog_post": "Write a {tone} blog post about {topic}. The post should be approximately {length} words and include the following sections: {sections}.",
                "product_description": "Create a compelling product description for {product_name}, which is a {product_type}. Highlight its {features} and focus on the benefits for {target_audience}. The description should be {length} and use a {tone} tone.",
                "social_media": "Write a {platform} post about {topic}. The post should be {length}, use a {tone} tone, and include the following hashtags: {hashtags}."
            },
            "code_generation": {
                "function": "Write a {language} function that {functionality}. The function should take {inputs} as input and return {outputs}. Include comments and error handling.",
                "refactor": "Refactor the following {language} code to make it more {goal} (e.g., efficient, readable, maintainable):\n\n```{language}\n{code}\n```",
                "debug": "Debug the following {language} code and fix any issues:\n\n```{language}\n{code}\n```"
            },
            "roleplay": {
                "expert": "You are an expert in {field} with {experience} years of experience. {request}",
                "character": "You are {character}, {character_description}. Respond to the following as this character would:\n\n{query}",
                "debate": "You are an expert who believes {position}. Present the strongest arguments for this position, addressing the following points: {points}."
            },
            "analysis": {
                "data": "Analyze the following data about {topic} and provide insights on {aspects}:\n\n{data}",
                "text": "Perform a detailed analysis of the following text, focusing on {aspects}:\n\n{text}",
                "comparison": "Compare and contrast {subject_1} and {subject_2} based on the following criteria: {criteria}."
            }
        }

        # Common modifiers that can be applied to most prompts
        self.modifiers = {
            "tone": [
                "professional", "casual", "academic", "enthusiastic",
                "informative", "persuasive", "humorous", "formal"
            ],
            "length": [
                "concise (100-150 words)", "detailed (300-500 words)",
                "brief (2-3 sentences)", "comprehensive (1000+ words)"
            ],
            "format": [
                "bullet points", "numbered list", "essay format",
                "Q&A format", "dialogue", "table"
            ]
        }

    def _few_shot_classification_template(self, categories: List[str], content_type: str, content: str) -> str:
        """Generate a few-shot learning template with examples for each category."""
        examples = []
        for category in categories:
            # This would ideally use real examples; using placeholders for demonstration
            examples.append(
                f"Example {content_type}: \"{self._generate_example_for_category(category, content_type)}\"\nCategory: {category}\n")

        few_shot_template = "Classify the following {content_type} into one of these categories: {categories}.\n\n"
        few_shot_template += "Here are some examples:\n\n"
        few_shot_template += "\n".join(examples)
        few_shot_template += "\n\nNow classify this:\n{content}"

        return few_shot_template.format(
            content_type=content_type,
            categories=", ".join(categories),
            content=content
        )

    def _generate_example_for_category(self, category: str, content_type: str) -> str:
        """Generate a simple example for a given category. In a real application, this would use a database of examples."""
        # This is a simplified placeholder - in a real application, you'd use actual examples
        examples_by_category = {
            "positive": "I absolutely love this product! It exceeded all my expectations.",
            "negative": "Terrible experience. The product broke after one use.",
            "neutral": "The product works as expected. Delivery was on time.",
            "technology": "The new processor architecture enables 15% faster computation while reducing power consumption.",
            "politics": "The senate voted 52-48 to pass the new infrastructure bill yesterday.",
            "sports": "The team won their third consecutive championship after an impressive comeback.",
            "entertainment": "The new movie broke box office records during its opening weekend.",
            "science": "Researchers discovered a new species of deep-sea organisms near hydrothermal vents.",
            "health": "The study found that regular exercise can significantly reduce the risk of heart disease."
        }

        return examples_by_category.get(category.lower(), f"This is an example of {category} content.")

    def get_template(self, task_type: str, technique: str = "default") -> Union[str, callable]:
        """Retrieve a specific template based on task type and technique."""
        if task_type not in self.templates:
            raise ValueError(f"Unknown task type: {task_type}. Available types: {list(self.templates.keys())}")

        if technique not in self.templates[task_type]:
            techniques = list(self.templates[task_type].keys())
            # Default to the first technique if the specified one isn't available
            technique = techniques[0]
            print(f"Technique '{technique}' not found for task type '{task_type}'. Using '{techniques[0]}' instead.")

        return self.templates[task_type][technique]

    def list_available_templates(self) -> Dict[str, List[str]]:
        """List all available task types and their techniques."""
        return {task: list(techniques.keys()) for task, techniques in self.templates.items()}

    def get_random_modifier(self, modifier_type: str) -> str:
        """Get a random modifier of the specified type."""
        if modifier_type not in self.modifiers:
            raise ValueError(f"Unknown modifier type: {modifier_type}. Available types: {list(self.modifiers.keys())}")

        return random.choice(self.modifiers[modifier_type])

    def generate_prompt(self, task_type: str, technique: str = "default", **kwargs) -> str:
        """Generate a prompt based on the specified template and parameters."""
        template = self.get_template(task_type, technique)

        # If template is a function (for complex templates like few-shot), call it with kwargs
        if callable(template):
            return template(**kwargs)

        # For standard string templates, format with kwargs
        try:
            return template.format(**kwargs)
        except KeyError as e:
            missing_key = str(e).strip("'")
            raise ValueError(f"Missing required parameter: {missing_key} for {task_type}/{technique} template")

    def save_templates_to_file(self, filename: str) -> None:
        """Save all templates to a JSON file."""
        # Convert callable templates to their __name__ for serialization
        serializable_templates = {}
        for task, techniques in self.templates.items():
            serializable_techniques = {}
            for technique, template in techniques.items():
                if callable(template):
                    serializable_techniques[technique] = f"CALLABLE: {template.__name__}"
                else:
                    serializable_techniques[technique] = template
            serializable_templates[task] = serializable_techniques

        with open(filename, 'w') as f:
            json.dump({
                "templates": serializable_templates,
                "modifiers": self.modifiers
            }, f, indent=2)

        print(f"Templates saved to {filename}")


def main():
    """Command line interface for the PromptGenerator."""
    parser = argparse.ArgumentParser(description="Generate prompts for large language models.")
    parser.add_argument("--task", type=str, help="Type of task (e.g., classification, summarization)")
    parser.add_argument("--technique", type=str, default="default", help="Technique to use for the prompt")
    parser.add_argument("--list", action="store_true", help="List all available task types and techniques")
    parser.add_argument("--save", type=str, help="Save templates to specified JSON file")
    parser.add_argument("--params", type=str, help="JSON string of parameters for the template")

    args = parser.parse_args()
    generator = PromptGenerator()

    if args.list:
        templates = generator.list_available_templates()
        print("Available task types and techniques:")
        for task, techniques in templates.items():
            print(f"\n{task.upper()}:")
            for technique in techniques:
                print(f"  - {technique}")

    elif args.save:
        generator.save_templates_to_file(args.save)

    elif args.task:
        params = {}
        if args.params:
            try:
                params = json.loads(args.params)
            except json.JSONDecodeError:
                print("Error: --params must be a valid JSON string")
                return

        try:
            prompt = generator.generate_prompt(args.task, args.technique, **params)
            print("\n" + "=" * 50 + " GENERATED PROMPT " + "=" * 50)
            print(prompt)
            print("=" * 120)
        except (ValueError, KeyError) as e:
            print(f"Error generating prompt: {e}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()