# pydantic_util.py

from typing import List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum
from langchain.output_parsers import PydanticOutputParser
from langchain_core.tools import tool

# Define possible task types
class TaskType(str, Enum):
    RECIPE_REFERENCE_RETRIEVAL = "recipe_reference_retrieval"
    #NUTRITIONAL_ANALYSIS = "nutritional_analysis"
    #SOURCING_ANALYSIS = "sourcing_analysis"
    #FLAVOR_PROFILING = "flavor_profiling"
    MODIFICATION_SUGGESTION = "modification_suggestion"
    #PERIPHERAL_INTERPRETATION = "peripheral_interpretation"
    OTHER = "other"

# Define a generic feedback model
class Feedback(BaseModel):
    comments: Optional[str]
    suggestions: Optional[List[str]]
    impact: Optional[str]  # Description of how this feedback impacts the recipe or other tasks

# Define a model for task-specific data (customize as needed for each task)
class TaskData(BaseModel):
    recipe_id: Optional[str]
    task_type: TaskType
    data: Union[dict, str, list]  # Placeholder for actual task-specific data structure

# Define the main message model
class NodeMessage(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the task")
    task_type: TaskType = Field(..., description="Type of the task being performed")
    task_data: TaskData = Field(..., description="Data specific to the task being performed")
    sender: str = Field(..., description="Identifier of the sender of the message")
    next: Optional[List[str]] = Field(None, description="Intended recipient(s) of the message")
    feedback: Optional[Feedback] = Field(None, description="Feedback from the task execution")
    status: str = Field(..., description="Current status of the task (e.g., 'pending', 'completed', 'failed')")
    timestamp: Optional[str] = Field(None, description="Timestamp of the message")
    metadata: Optional[dict] = Field(None, description="Additional metadata or context")

CauldronPydanticParser = PydanticOutputParser(pydantic_object=NodeMessage)
format_instructions = CauldronPydanticParser.get_format_instructions()