import json
import heapq
import uuid
import pickle
import os
from langchain_core.tools import tool
from typing import Annotated

default_mods_list_file = "mods_list.pkl"

class ModsList:
    def __init__(self):
        self.queue = []

    def suggest_mod(self, mod):
        heapq.heappush(self.queue, (-mod['priority'], mod))
    
    def get_mods_list(self):
        return [mod for priority, mod in sorted(self.queue, reverse=True)]
    
    def push_mod(self):
        if self.queue:
            return heapq.heappop(self.queue)[1]
        return None
    
    def rank_mod(self, mod_id, new_priority):
        for i, (priority, mod) in enumerate(self.queue):
            if mod['id'] == mod_id:
                self.queue[i] = (-new_priority, mod)
                heapq.heapify(self.queue)
                break

    def remove_mod(self, mod_id: str) -> bool:
        for i, (priority, mod) in enumerate(self.queue):
            if mod['id'] == mod_id:
                self.queue.pop(i)
                heapq.heapify(self.queue)
                return True
        return False

def fresh_mods_list(filename):
    mods_list = ModsList()
    save_mods_list_to_file(mods_list, filename)

def save_mods_list_to_file(mods_list, filename):
    with open(filename, 'wb') as file:
        pickle.dump(mods_list, file)

def load_mods_list_from_file(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    return ModsList()

@tool
def suggest_mod(
    mod_json: Annotated[str, "The modification JSON to be suggested."],
    mods_list_file: Annotated[str, "The filename for the mods list."] = default_mods_list_file
) -> Annotated[str, "The updated list of modifications."]:
    """Suggest a new modification to be added to the mods list."""
    try:
        mod_data = json.loads(mod_json)
        mods_list = load_mods_list_from_file(mods_list_file)
        mods_list.suggest_mod(mod_data)
        save_mods_list_to_file(mods_list, mods_list_file)
        updated_mods_list = mods_list.get_mods_list()
        return json.dumps(updated_mods_list, indent=2)
    except json.JSONDecodeError:
        return "Failed to suggest modification. Invalid JSON input."

@tool
def get_mods_list(
    mods_list_file: Annotated[str, "The filename for the mods list."] = default_mods_list_file
) -> Annotated[str, "The current list of suggested modifications."]:
    """Get the current list of suggested modifications."""
    try:
        mods_list = load_mods_list_from_file(mods_list_file)
        current_mods_list = mods_list.get_mods_list()
        return json.dumps(current_mods_list, indent=2)
    except json.JSONDecodeError:
        return "Failed to get modifications list. Invalid JSON input."

@tool
def push_mod(
    mods_list_file: Annotated[str, "The filename for the mods list."] = default_mods_list_file
) -> Annotated[str, "The modification that was applied, if available."]:
    """Apply the top modification from the mods list."""
    try:
        mods_list = load_mods_list_from_file(mods_list_file)
        mod_to_apply = mods_list.push_mod()
        save_mods_list_to_file(mods_list, mods_list_file)
        if mod_to_apply:
            return json.dumps(mod_to_apply, indent=2)
        else:
            return "No modifications available to apply."
    except json.JSONDecodeError:
        return "Failed to push modification. Invalid JSON input."

@tool
def rank_mod(
    mod_id: Annotated[str, "The ID of the modification to reprioritize."],
    new_priority: Annotated[int, "The new priority for the modification (1 = highest priority, larger numbers = lower priority)."],
    mods_list_file: Annotated[str, "The filename for the mods list."] = default_mods_list_file
) -> Annotated[str, "The updated list of modifications."]:
    """Reprioritize a given modification within the mods list.

    The priority ranking options are as follows:
    - A lower numerical value indicates higher priority (e.g., 1 is the highest priority).
    - Larger numerical values indicate lower priority.
    """
    try:
        mods_list = load_mods_list_from_file(mods_list_file)
        mods_list.rank_mod(mod_id, new_priority)
        save_mods_list_to_file(mods_list, mods_list_file)
        updated_mods_list = mods_list.get_mods_list()
        return json.dumps(updated_mods_list, indent=2)
    except json.JSONDecodeError:
        return "Failed to reprioritize modification. Invalid JSON input."
    
@tool
def remove_mod(
    mods_list_file: Annotated[str, "The filename for the mods list."],
    mod_id: Annotated[str, "The ID of the modification to remove."]
) -> Annotated[str, "Confirmation message of removal status."]:
    """Remove a modification from the mods list."""
    mods_list = load_mods_list_from_file(mods_list_file)
    success = mods_list.remove_mod(mod_id)
    save_mods_list_to_file(mods_list, mods_list_file)
    if success:
        return f"Modification with ID {mod_id} has been successfully removed."
    else:
        return f"Modification with ID {mod_id} not found in the list."