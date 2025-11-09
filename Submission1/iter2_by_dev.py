from agent_hive.task import Task
from pydantic import Field
from typing import List
from agent_hive.enum import ContextType
import json
from agent_hive.workflows.base_workflow import Workflow
from reactxen.utils.model_inference import watsonx_llm
import re
from agent_hive.workflows.sequential import SequentialWorkflow
from agent_hive.agents.plan_reviewer_agent import PlanReviewerAgent
from agent_hive.logger import get_custom_logger

logger = get_custom_logger(__name__)

# =========================================================
# TODO: Participants can edit this section ONLY
# Add variable, dict. no more any import just any inline code
# =========================================================
# END OF EDITABLE SECTION


class NewPlanningWorkflow(Workflow):
    """
    Participant Template for Planning Review Workflow.
    ---------------------------------------------------
    📝 Instructions for participants:
    - Only modify the section marked with "TODO: Edit prompt here"
    - Do NOT change any workflow logic, agents, or execution components
    - Keep all retry, memory, and sequential execution intact
    """

    llm: str = Field(description="LLM used by the task planning.")

    def __init__(self, tasks: List[Task], llm: str):
        self.tasks = tasks
        self.memory = []
        self.max_memory = 10
        self.llm = llm
        self.max_retries = 5
        self._verify_tasks()

    def _verify_tasks(self):
        if not isinstance(self.tasks, list):
            raise ValueError("tasks must be a list of Task objects")
        if len(self.tasks) != 1:
            raise ValueError("Planning only supports one task")
        task = self.tasks[0]
        if task.agents is None or len(task.agents) < 1:
            raise ValueError("Task must have at least one agent")

    def run(self, enable_summarization=False):
        generated_steps = self.generate_steps()

        sequential_workflow = SequentialWorkflow(
            tasks=generated_steps, context_type=ContextType.SELECTED
        )

        return sequential_workflow.run()

    def generate_steps(self, save_plan=False, saved_plan_filename=""):
        task = self.tasks[0]
        agent_descriptions = ""

        # =========================================================
        # TODO: Participants can edit this section ONLY
        # 🎨 Purpose: Customize how agent information is collected and formatted
        # ✅ Allowed:
        #     - Change numbering style or bullet points
        #     - Include additional metadata (e.g., agent capabilities, tags)
        #     - Provide examples in a different format
        #     - Add emojis or formatting to make the prompt clearer
        #     - More thinking
        # ❌ Not allowed:
        #     - Modify workflow execution
        #     - Replace the base ReAct agent or Executor
        #     - Change memory or retry logic
        # =========================================================

        # Extract metadata from task if available
        query_category = getattr(task, 'category', None)
        query_type = getattr(task, 'type', None)
        characteristic_form = getattr(task, 'characteristic_form', None)

        logger.info(f"[EDIT SECTION 1] Extracting metadata - Category: {query_category}, Type: {query_type}")

        # Format agent descriptions with clear Markdown for the LLM
        agent_descriptions = "Here are the agents you MUST choose from:\n\n"
        for ii, aagent in enumerate(task.agents):
            agent_descriptions += f"--- (Agent {ii + 1}) ---\n"
            agent_descriptions += f"**Agent Name:** `{aagent.name}`\n"
            agent_descriptions += f"**Description:** {aagent.description}\n"
            logger.info(f"[EDIT SECTION 1] Processing agent {ii + 1}: {aagent.name}")

            if "task_examples" in aagent.__dict__ and aagent.task_examples:
                agent_descriptions += f"**Example Use-Cases (for your inspiration):**\n"
                for idx, task_example in enumerate(aagent.task_examples, start=1):
                    agent_descriptions += f"  {idx}. {task_example}\n"
            agent_descriptions += "\n"  # Add a newline for separation

        logger.info(f"[EDIT SECTION 1] Agent descriptions formatted with {len(task.agents)} agents")

        # =========================================================
        # END OF EDITABLE SECTION
        # 🚫 Participants should not modify code below this line
        # ❌ No new variables, functions, or workflow logic allowed
        # ✅ Only modify the section marked as TODO above
        # =========================================================

        prompt = self.get_prompt(task.description, agent_descriptions)
        logger.info(f"Plan Generation Prompt: \n{prompt}")
        llm_response = watsonx_llm(
            prompt, model_id=self.llm,
        )["generated_text"]
        logger.info(f"Plan: \n{llm_response}")

        final_plan = llm_response
        self.memory = []

        task_pattern = r"#Task\d+: (.+)"
        agent_pattern = r"#Agent\d+: (.+)"
        dependency_pattern = r"#Dependency\d+: (.+)"
        output_pattern = r"#ExpectedOutput\d+: (.+)"

        tasks = re.findall(task_pattern, final_plan)
        agents = re.findall(agent_pattern, final_plan)
        dependencies = re.findall(dependency_pattern, final_plan)
        outputs = re.findall(output_pattern, final_plan)

        if save_plan:
            if not saved_plan_filename.endswith(".txt"):
                saved_plan_filename += ".txt"

            saved_plan_text = f"Question: {task.description}\nPlan:\n{final_plan}"
            with open(saved_plan_filename, "w") as f:
                f.write(saved_plan_text)

        planned_tasks = []
        for i in range(len(tasks)):
            task_description = tasks[i]
            if i == len(agents):
                break
            agent_name = agents[i]
            if i < len(dependencies):
                dependency = dependencies[i]
            else:
                dependency = "None"
            if i < len(outputs):
                expected_output = outputs[i]
            else:
                expected_output = ""

            selected_agent = None
            for agent in task.agents:
                if agent.name == agent_name:
                    selected_agent = agent
                    break
            if selected_agent is None:
                selected_agent = task.agents[0]

            if dependency != "None":
                numbers = re.findall(r"#S(\d+)", dependency)
                numbers = list(map(int, numbers))
                context = [planned_tasks[i - 1] for i in numbers]
            else:
                context = []

            a_task = Task(
                description=task_description,
                expected_output=expected_output,
                agents=[selected_agent],
                context=context,
            )
            planned_tasks.append(a_task)

        logger.info(f"Planned Tasks: \n{planned_tasks}")

        return planned_tasks

    def get_prompt(self, task_description, agent_descriptions):
        # =========================================================
        # TODO: Participants can edit this section ONLY
        # 🎨 Purpose: Improve prompt clarity, formatting, emojis, guidance
        # ✅ Allowed: Wording, structure, examples, emojis
        # ❌ Not allowed: Changing workflow, ReAct agent, Executor, or memory logic
        # =========================================================

        # Extract metadata from task if available
        task = self.tasks[0]
        query_category = getattr(task, 'category', None)
        query_type = getattr(task, 'type', None)
        characteristic_form = getattr(task, 'characteristic_form', None)

        logger.info(f"[EDIT SECTION 2] Building prompt with metadata - Category: {query_category}, Type: {query_type}")

        # Build query info section if metadata is available
        query_info = "### 3. Query Context (If any)\nNo additional query information provided."
        if query_type or query_category or characteristic_form:
            query_info = "### 3. Query Context\n"
            query_info += "Use this information to better understand the user's goal:\n"
            if query_type:
                query_info += f"* **Query Type:** {query_type}\n"
            if query_category:
                query_info += f"* **Query Category:** {query_category}\n"
            if characteristic_form:
                query_info += f"* **Expected Format:** {characteristic_form}\n"

            # Add explicit guidance linked to the metadata
            if query_type == "Inference Query":
                query_info += "-> **Action:** Your plan must focus on analysis and deriving insights, not just data retrieval."
            elif query_type == "Data Query":
                query_info += "-> **Action:** Your plan should focus on retrieving specific data values."
            elif query_type == "Anomaly Detection Query":
                query_info += "-> **Action:** Your plan *must* involve an agent capable of identifying outliers or abnormal patterns."

            logger.info(f"[EDIT SECTION 2] Query information section added to prompt")
        else:
            logger.info(f"[EDIT SECTION 2] No metadata available, using generic prompt")

        prompt = f"""🚀 You are an expert AI-driven Task Planner. Your sole responsibility is to decompose a complex "Problem" into a sequence of actionable "Tasks" that can be executed by the "Available Agents".

Your output *must* be parsed by a script, so it *must* be perfect.

## 1. The Goal
Your plan must, when executed, fully and completely answer the "Problem to Solve".

## 2. Available Agents
{agent_descriptions}

{query_info}

## 4. Problem to Solve
{task_description}

## 5. Your Instructions & Output Format

**Your Thought Process (Internal Monologue - DO NOT WRITE THIS):**
1.  **Analyze Goal:** What is the user's final, desired answer?
2.  **Analyze Agents:** Which agents from the list are needed? (e.g., `DataAgent` for data, `AnalysisAgent` for insights).
3.  **Plan Steps:** Design a step-by-step plan.
4.  **Handle Data:** How will the plan get data? (e.g., "Task 1: Fetch data"). How will it handle file paths? (e.g., "Task 2: Find file").
5.  **Validate:** How will the plan check for errors? (e.g., "Task 3: Validate retrieved data").
6.  **Synthesize:** How will the *last* task combine all results for a final answer?

**Your Output (What you MUST generate):**
Your response *must* be *only* the plan, starting *immediately* with `#Task1`.
Do NOT include *any* other text, conversation, explanations, or reasoning (like "Here is the plan:").

### ⚠️ CRITICAL RULES ⚠️
1.  **AGENT CONSTRAINT (STRICT):** You *must* use *only* agent names from the "Available Agents" list. The `#Agent<N>` name must be an **exact, character-for-character match** for an `**Agent Name:**` (e.g., `SalesDataAgent`). Do not invent or misspell agents. This is a common failure.
2.  **DATA & FILE HANDLING:** When a task involves a file or data (e.g., "read file.csv", "get sensor data"), the task description must be about *the action of retrieving or reading* it. Do not assume a specific file path. The plan must be able to handle "File Not Found" or "Incomplete Data".
3.  **NO HYPOTHETICALS:** Your output *must be a plan*, not an answer. Do not provide hypothetical examples, data, or results. If the problem is "Find the top salesperson," your plan must describe *how* to find them, not output "The top salesperson is Jane Doe (example)."
4.  **ROBUSTNESS & VALIDATION:** Your plan *must* be robust. If #S1 retrieves data, add a task #S2 to *validate* that data (e.g., "Check data for completeness") before #S3 tries to use it. This prevents "Lack of Output Validation" failures.
5.  **DEPENDENCY LOGIC:** Use #Dependency: None for any task that is a "starting point" (i.e., it does not need the output from another task in this plan). You can have multiple tasks with #Dependency: None (e.g., for fetching data from different sources in parallel). For any task that does need the output from a previous step, you must list it (e.g., #Dependency3: #S1, #S2).
6.  **FINAL ANSWER (MANDATORY):** The *last task* (e.g., `#Task<N>`) *must* be the one that provides the final, synthesized answer to the user. It should depend on all previous steps needed for the answer (e.g., `#Dependency<N>: #S1, #S3`). This fixes "Lack of Final Answer" failures.
7.  **EFFICIENCY:** Do not create redundant steps. If #S1 fetches data, re-use it. Do not fetch it again.
8.  **IMMEDIATE OUTPUT:** Your response *must* begin *exactly* with `#Task1:`. Do NOT add any text, spaces, or newlines before it.

**START OF YOUR RESPONSE (Must be #Task1):**
#Task1: <Describe the first sub-task>
#Agent1: <Exact_agent_name_from_Section_2>
#Dependency1: None
#ExpectedOutput1: <What this step will produce>
#Task2: <Describe the second sub-task>
...
"""
        # =========================================================
        # End of participant editable section
        # =========================================================
        return prompt
