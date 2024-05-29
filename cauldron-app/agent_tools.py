from datetime import datetime
from langchain.tools import Tool

# Define a new tool that returns the current datetime
datetime_tool = Tool(
    name="Datetime",
    func=lambda x: datetime.now().isoformat(),
    description="Returns the current datetime",
)