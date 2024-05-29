from datetime import datetime
from langchain.tools import Tool
import random
import string

# Define a new tool that returns the current datetime
datetime_tool = Tool(
    name="Datetime",
    func=lambda x: datetime.now().isoformat(),
    description="Returns the current datetime",
)

#Task-ID Generator
task_ID_tool = Tool(
    name="TaskID",
    func=lambda x: ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)),
    description="Returns a random Task ID for the current task",
)

