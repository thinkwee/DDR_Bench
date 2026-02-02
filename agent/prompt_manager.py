#!/usr/bin/env python3
"""
Prompt management for the autonomous data analysis agent
"""

class PromptManager:
    """Manages system prompts and task-specific instructions"""
    
    def __init__(self, auto_finish: bool = True):
        self.auto_finish = auto_finish
        
        # Base system prompt for autonomous operation
        self.base_system_prompt = """You are an autonomous data analysis agent. Your task is to keep exploring and analyzing data for the given task.

IMPORTANT INSTRUCTIONS:
1. always respond in a ReAct style: return what you're thinking and planning to do, and then call the appropriate tool. RETURN BOTH THE TEXT CONTENT AND THE TOOL CALL.
2. you can only call one tool every turn.
3. your reasoning should contain insights derived from last turn's tool call results. BUT DO NOT INCLUDE ANY INSIGHTS OR REASONING IN THE TOOL CALLS. TOOLS ARE ONLY FOR DATA EXPLORE.
4. you should try your best to use the tools to get more information. keep exploring, build more and more complex params as turns go on and you will discover more in the data.
5. first use the tools to check what data is available to you.
"""
        
        # Task completion instruction (conditional)
        self.finish_instruction = """

TASK COMPLETION:
When you can not gather more information, send a message that starts with "FINISH:" followed by your all insights collected from the whole dialogue and tool calls. 
Only use "FINISH:" when you are absolutely certain that no more information can be gathered. 
Carefully use "FINISH:" in your message since it will immediately end the session. Think twice before using it."""
    
    def build_system_prompt_with_task(self, task: str) -> str:
        """Build system prompt with specific task"""
        enhanced_prompt = self.base_system_prompt
        
        # Add finish instruction only if auto_finish is enabled
        if self.auto_finish:
            enhanced_prompt += self.finish_instruction
        
        enhanced_prompt += f"\n\nYOUR TASK: {task}"
        enhanced_prompt += "\n\nAnalyze the task and use the available tools to accomplish it step by step."
        enhanced_prompt += "\n\nALWAYS RETURN in the format of text content with a tool call, no just return the tool call."
        
        return enhanced_prompt
    
    def build_insight_prompt(self, assistant_content: str, user_content: str, task: str) -> str:
        """Build prompt for generating insights from tool execution"""
        return f"""Based on the following tool execution, provide a brief insight about what was discovered or learned:

The reason and action to use the tool:
{assistant_content}

Tool execution result:
{user_content}

Provide a concise insight (1-3 sentences) about what this reveals:
1. It has to be related to the task: {task}.
2. If there is no insight or error in the tool execution, respond with 'NO INSIGHT'.
3. If it only use the data description tools (e.g. tools like list_files, describe_table, get_database_info, get_field_description), respond with 'NO INSIGHT'.
4. The insight from data should answer the question raised in the reason to execute this tool. Focus on this point.
5. Keep all the data or statitics needed in your generated insight.
ONLY respond with the insight."""
    
    def get_insight_system_prompt(self) -> str:
        """Get system prompt for insight generation"""
        return "You are an expert data analyst. Provide concise, actionable insights based on tool execution results."
    
    def build_final_summary_prompt(self, messages: list) -> str:
        """Build prompt for generating final summary from chat message list"""
        # Format conversation history from chat message list
        conversation_text = ""
        for msg in messages:
            if msg.get("role") != "system":
                conversation_text += f"{msg.get('role', '').upper()}: {msg.get('content', '')}\n"
                if msg.get('tool_call'):
                    conversation_text += f"TOOL_CALL: {msg['tool_call']}\n"
                if msg.get('tool_result'):
                    conversation_text += f"TOOL_RESULT: {msg['tool_result']}\n"
        
        return f"""Based on the entire conversation history below, provide a comprehensive final summary of your analysis and findings.

CONVERSATION HISTORY:
{conversation_text}

Please provide a detailed final summary that includes all insights collected from the whole dialogue and tool calls. The summary should be no more than 8192 tokens. Format your response as: "FINISH: [your comprehensive summary here]"
"""
    
    def get_final_summary_system_prompt(self) -> str:
        """Get system prompt for final summary generation"""
        return "You are an expert data analyst. Provide comprehensive final summaries based on complete conversation histories."